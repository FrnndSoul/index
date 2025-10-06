#!/usr/bin/env python3
"""
Raspberry Pi / USB Cam API @ 1280x720 30fps (robust)
- /api/cam      : MJPEG stream (multipart/x-mixed-replace)
- /api/cam.jpg  : Single JPEG snapshot
- /api/health   : Health probe
"""

import time, threading, signal, sys, logging
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, make_response
from flask_cors import CORS

from picamera2 import Picamera2
import cv2
import numpy as np

# ========= Config (720p @ 30 fps) =========
WIDTH, HEIGHT   = 1280, 720
FPS_TARGET      = 30
JPEG_QUALITY    = 80           # lower (60–75) if CPU is high
HFLIP           = False
VFLIP           = False
ROTATE_DEG      = 0            # 0/90/180/270
USE_OPENCV_FALLBACK = True

# ========= Globals =========
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

picam = None
cap = None
use_opencv = False

_running = True
_frame_lock = threading.Lock()
_latest_jpeg = None
_latest_ts = 0.0

def _encode_jpeg(rgb_array):
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buf.tobytes() if ok else None

def _init_picamera2():
    """Initialize Picamera2; set 1280x720; try to set 30fps only if supported."""
    global picam
    picam = Picamera2()

    # Don’t force FrameRate in config; some UVC paths don’t expose it.
    video_cfg = picam.create_video_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
    picam.configure(video_cfg)

    # Apply only supported controls
    advertised = set(picam.camera_controls.keys())
    to_set = {}
    if "HorizontalFlip" in advertised: to_set["HorizontalFlip"] = bool(HFLIP)
    if "VerticalFlip"   in advertised: to_set["VerticalFlip"]   = bool(VFLIP)
    if "Rotation" in advertised and ROTATE_DEG in (0, 90, 180, 270):
        to_set["Rotation"] = ROTATE_DEG

    picam.start()
    time.sleep(0.4)

    if to_set:
        try:
            picam.set_controls(to_set)
        except Exception as e:
            logging.warning(f"Control set skipped: {e}")

    # Try to set FrameRate AFTER start if the control exists
    if "FrameRate" in advertised:
        try:
            picam.set_controls({"FrameRate": FPS_TARGET})
            logging.info("Picamera2 FrameRate set to %s", FPS_TARGET)
        except Exception as e:
            logging.warning(f"Could not set FrameRate: {e}")

    logging.info("Picamera2 active @ %dx%d (requested), controls: %s",
                 WIDTH, HEIGHT, sorted(advertised))

def _init_opencv():
    """OpenCV fallback for UVC webcams."""
    global cap, use_opencv
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS_TARGET)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open camera index 0")
    use_opencv = True
    # Log what we actually got (drivers may ignore)
    got_w  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    got_h  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    got_fps= cap.get(cv2.CAP_PROP_FPS)
    logging.info("OpenCV backend active (requested 1280x720@30) -> got %.0fx%.0f@%.0f",
                 got_w, got_h, got_fps or 0)

def _init_camera():
    try:
        _init_picamera2()
        logging.info("Using Picamera2 backend.")
    except Exception as e:
        logging.warning(f"Picamera2 init failed: {e}")
        if USE_OPENCV_FALLBACK:
            _init_opencv()
        else:
            raise

def _capture_loop():
    """Capture at ~FPS_TARGET. Even if cam can’t do 30, we pace to avoid overload."""
    global _latest_jpeg, _latest_ts
    frame_interval = 1.0 / max(1, FPS_TARGET)
    next_due = time.time()

    while _running:
        if use_opencv:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                time.sleep(0.005); continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = picam.capture_array("main")

        jpeg = _encode_jpeg(rgb)
        if jpeg:
            with _frame_lock:
                _latest_jpeg = jpeg
                _latest_ts = time.time()

        # pacing
        next_due += frame_interval
        sleep_for = next_due - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_due = time.time()

def _mjpeg_generator():
    boundary = "--frame"
    header_tpl = (
        "Content-Type: image/jpeg\r\n"
        "Content-Length: {length}\r\n\r\n"
    )
    # initial boundary
    yield boundary.encode("ascii") + b"\r\n"
    while True:
        with _frame_lock:
            jpeg = _latest_jpeg
        if jpeg is None:
            time.sleep(0.01)
            continue
        header = header_tpl.format(length=len(jpeg)).encode("ascii")
        yield header + jpeg + b"\r\n" + boundary.encode("ascii") + b"\r\n"

@app.route("/api/cam")
def api_cam_mjpeg():
    resp = Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp

@app.route("/api/cam.jpg")
def api_cam_snapshot():
    with _frame_lock:
        jpeg = _latest_jpeg
        ts = _latest_ts
    if jpeg is None:
        return make_response("Camera warming up", 503)
    r = make_response(jpeg)
    r.headers["Content-Type"] = "image/jpeg"
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    r.headers["X-Timestamp"] = str(ts)
    return r

@app.route("/api/health")
def api_health():
    with _frame_lock:
        ready = _latest_jpeg is not None
        age = (time.time() - _latest_ts) if _latest_ts else None
    return jsonify({
        "ok": True,
        "camera_ready": bool(ready),
        "backend": "opencv" if use_opencv else "picamera2",
        "target": {"w": WIDTH, "h": HEIGHT, "fps": FPS_TARGET},
        "last_frame_age_s": round(age, 3) if age is not None else None,
        "ts": int(datetime.now(timezone.utc).timestamp() * 1000),
    })

def _graceful_exit(*_):
    global _running
    _running = False
    try:
        if use_opencv and cap:
            cap.release()
        if not use_opencv and picam:
            picam.stop(); picam.close()
    finally:
        sys.exit(0)

def main():
    signal.signal(signal.SIGINT,  _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)
    _init_camera()
    t = threading.Thread(target=_capture_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)

if __name__ == "__main__":
    main()
