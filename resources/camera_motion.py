# resources/camera_motion.py
from __future__ import annotations
import os, time, threading, re, base64
from typing import Any, Dict, Tuple

import cv2, numpy as np
from flask import Response
from gpiozero import MotionSensor
from picamera2 import Picamera2

# Optional email notifier (kept here as requested)
from resources.email_notifier import send_email

# ----------------- Shared state -----------------
_started = False
_start_lock = threading.Lock()

_cfg_lock = threading.Lock()
CFG = {
    "intensity_min": float(os.getenv("INTENSITY_MIN", "0.10")),
    "pixel_delta":   int(os.getenv("PIXEL_DELTA", "20")),
    "k_sigma":       float(os.getenv("K_SIGMA", "3.0")),
    "sample_hz":     max(1.0, float(os.getenv("SAMPLE_HZ", "8"))),
    "email_cooldown_s": int(os.getenv("EMAIL_COOLDOWN_S", "120")),
}

# PIR
_pir_pin = 22
_pir = None
_last_state = 0
_last_change_ts = time.time()

def _update_pir_change():
    global _last_state, _last_change_ts
    s = 1 if _pir.motion_detected else 0
    if s != _last_state:
        _last_state = s
        _last_change_ts = time.time()

def pir_snapshot() -> Tuple[int, float]:
    _update_pir_change()
    now = time.time()
    return (1 if _pir.motion_detected else 0, (now - _last_change_ts) * 1000.0)

# Camera
_cam = None
_cam_lock = threading.Lock()

# Motion stats
_motion_lock = threading.Lock()
last_small = None
mu = 0.0
varEMA = 0.0
sigma = 0.02
currentDynThresh = CFG["intensity_min"]
curr_intensity = 0.0
curr_frame_jpeg = None
_last_email_ts = 0.0

# ----------------- Public lifecycle -----------------
def ensure_started(ctx: Dict[str, Any]):
    global _started, _pir_pin, _pir, _cam
    if _started:
        return
    with _start_lock:
        if _started:
            return

        # Resolve PIR pin from CTX if provided
        try:
            _pir_pin = int(ctx.get("pins", {}).get("pir", {}).get("pin", 22))
        except Exception:
            _pir_pin = 22

        _pir = MotionSensor(_pir_pin, sample_rate=4)

        # Camera init
        _cam = Picamera2()
        CAM_W, CAM_H = 640, 480
        config = _cam.create_video_configuration(main={"format": "RGB888", "size": (CAM_W, CAM_H)})
        _cam.configure(config)
        _cam.start()

        # Basic AWB settle + lock
        time.sleep(2.0)
        try:
            try:
                _cam.set_controls({"AwbEnable": True, "AwbMode": 3})
            except Exception:
                _cam.set_controls({"AwbEnable": True, "AwbMode": "Daylight"})
            time.sleep(1.0)
            md = _cam.capture_metadata()
            gains = md.get("ColourGains")
            if gains and isinstance(gains, tuple) and len(gains) == 2:
                _cam.set_controls({"AwbEnable": False, "ColourGains": gains})
            else:
                _cam.set_controls({"AwbEnable": False, "ColourGains": (1.0, 1.6)})
        except Exception:
            pass

        # Start motion loop
        threading.Thread(target=_motion_loop, daemon=True).start()

        _started = True

# ----------------- Config APIs -----------------
def get_config() -> Dict[str, Any]:
    with _cfg_lock:
        return dict(CFG)

def set_config(**body) -> Dict[str, Any]:
    with _cfg_lock:
        if "intensity_min" in body:
            v = float(body["intensity_min"]); CFG["intensity_min"] = max(0.0, min(1.0, v))
        if "pixel_delta" in body:
            v = int(body["pixel_delta"]); CFG["pixel_delta"] = max(1, min(255, v))
        if "k_sigma" in body:
            CFG["k_sigma"] = max(0.0, float(body["k_sigma"]))
        if "sample_hz" in body:
            CFG["sample_hz"] = max(1.0, float(body["sample_hz"]))
        if "email_cooldown_s" in body:
            CFG["email_cooldown_s"] = max(10, int(body["email_cooldown_s"]))
        return dict(CFG)

# ----------------- Status API -----------------
def get_status() -> Dict[str, Any]:
    with _cfg_lock:
        floor = CFG["intensity_min"]; ksig = CFG["k_sigma"]; pdel = CFG["pixel_delta"]
    with _motion_lock:
        inten = float(curr_intensity); thr = float(currentDynThresh)
    val, _since = pir_snapshot()
    return {
        "ok": True,
        "ts": int(time.time() * 1000),
        "pir": int(val),
        "intensity": inten,
        "threshold": thr,
        "floor": floor,
        "k_sigma": ksig,
        "pixel_delta": pdel,
    }

# ----------------- Camera / MJPEG -----------------
def _capture_rgb():
    with _cam_lock:
        rgb = _cam.capture_array("main")
    return rgb

def _jpeg_from_bgr(bgr, quality=80):
    ok, jpeg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return (ok, jpeg.tobytes() if ok else None)

def mjpeg_stream_response(quality=80, max_fps=None, swap=False) -> Response:
    def _gen():
        last = 0.0
        frame_interval = (1.0 / max_fps) if max_fps and max_fps > 0 else 0.0
        while True:
            now = time.time()
            if frame_interval and (now - last) < frame_interval:
                time.sleep(0.001); continue
            rgb = _capture_rgb()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) if swap else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, jpg = _jpeg_from_bgr(bgr, quality=quality)
            if not ok: continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                   jpg + b"\r\n")
            last = now
    resp = Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

# ----------------- Motion loop -----------------
def _gray_small(bgr):
    small = cv2.resize(bgr, (160, 120), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

def _motion_loop():
    global last_small, mu, varEMA, sigma, currentDynThresh, curr_intensity, curr_frame_jpeg, _last_email_ts
    while True:
        with _cfg_lock:
            INTENSITY_MIN = CFG["intensity_min"]
            PIXEL_DELTA   = CFG["pixel_delta"]
            K_SIGMA       = CFG["k_sigma"]
            SAMPLE_HZ     = CFG["sample_hz"]
            EMAIL_COOLDOWN_S = CFG["email_cooldown_s"]
        period = 1.0 / SAMPLE_HZ

        t0 = time.time()
        rgb = _capture_rgb()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        y = _gray_small(bgr)

        if last_small is None:
            last_small = y
            _sleep_rem(t0, period); continue

        diff = cv2.absdiff(y, last_small)
        changed = (diff >= PIXEL_DELTA).sum()
        total = diff.size
        intensity = changed / float(total)

        prevMu = mu
        mu = (1 - 0.02) * mu + 0.02 * intensity
        dev = intensity - prevMu
        varEMA = (1 - 0.02) * varEMA + 0.02 * (dev * dev)
        sigma = max(0.01, float(np.sqrt(varEMA)))
        currentDynThresh = max(INTENSITY_MIN, mu + K_SIGMA * sigma)
        curr_intensity = float(intensity)

        ok, jpeg = _jpeg_from_bgr(bgr, quality=80)
        if ok:
            curr_frame_jpeg = jpeg

        last_small = y

        val, _ = pir_snapshot()
        now_s = time.time()
        if val == 1 and curr_intensity >= currentDynThresh and (now_s - _last_email_ts) >= EMAIL_COOLDOWN_S:
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_s))
            subj = f"Motion alert — PIR=1 intensity={curr_intensity:.2f}"
            html = f"""<div style="font-family:Arial,Helvetica,sans-serif">
            <h3>Motion detected</h3>
            <p><b>Time:</b> {when}</p>
            <p><b>PIR:</b> 1 &nbsp; <b>Intensity:</b> {curr_intensity:.2f} &nbsp; <b>Thresh:</b> {currentDynThresh:.2f}</p>
            <p style="color:#666">Auto-sent by Raspberry Pi.</p>
            </div>"""
            try:
                send_email(subj, html, curr_frame_jpeg)
                _last_email_ts = now_s
            except Exception:
                pass

        _sleep_rem(t0, period)

def _sleep_rem(start, period):
    dt = time.time() - start
    if dt < period:
        time.sleep(period - dt)
