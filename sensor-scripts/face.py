#!/usr/bin/env python3
from __future__ import annotations
import time, threading
from typing import Dict, Optional, Tuple

try:
    from .camera import CameraManager  # shared camera (prevents device conflicts)
    _HAS_CAM_MGR = True
except Exception:
    _HAS_CAM_MGR = False
    import cv2


class _FaceRuntime:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.stop_evt = threading.Event()
        self.fps: float = 0.0
        self.faces: int = 0
        self.draw_boxes: bool = True   # <-- NEW: overlay toggle

        self.cam_index = 0
        self.size: Tuple[int, int] = (640, 480)
        self._cap = None
        self._cascade = None
        self._cam_mgr = None

    # ---------- camera ----------
    def _open_cam(self):
        if _HAS_CAM_MGR:
            self._cam_mgr = CameraManager(cam_index=self.cam_index,
                                          width=self.size[0], height=self.size[1])
            self._cam_mgr._open()  # ensure settings applied early (safe no-op if already open)
            self._cam_mgr.acquire()
            return
        import cv2
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
            # Ask for MJPG 640x480 @ 30fps (best effort)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.size[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            try: self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception: pass

    def _close_cam(self):
        if _HAS_CAM_MGR:
            try: self._cam_mgr.release()
            except Exception: pass
            return
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _read(self):
        if _HAS_CAM_MGR:
            return self._cam_mgr.read()
        return self._cap.read()

    # ---------- cascade ----------
    def _ensure_cascade(self):
        if self._cascade is not None:
            return
        import cv2
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # ---------- stream ----------
    def gen(self, quality: int = 60, fps_limit: float = 30.0):
        """
        Fast MJPEG with optional face rectangles.
        - Downscale gray for detection, then map boxes back.
        - Detect every N frames, reuse boxes between detections.
        """
        import cv2, time

        DETECT_EVERY = 3   # try 4 if you still need more FPS
        SCALE = 1.15
        MIN_NEIGH = 4
        MIN_SIZE_FULL = (48, 48)   # full-res min face
        DOWNSCALE = 0.5            # detect on 1/2 size

        with self.lock:
            self.stop_evt.clear()
            self.running = True
            self.fps = 0.0
            self.faces = 0

        self._open_cam()
        self._ensure_cascade()

        frame_i = 0
        last_boxes_full = []
        t0 = time.time()
        n = 0
        min_dt = 1.0 / fps_limit if fps_limit and fps_limit > 0 else 0.0
        last_tick = 0.0

        # precompute encode params
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

        try:
            while not self.stop_evt.is_set():
                if min_dt:
                    now = time.time()
                    dt = now - last_tick
                    if dt < min_dt:
                        time.sleep(min_dt - dt)
                    last_tick = time.time()

                ok, frame = self._read()
                if not ok or frame is None:
                    time.sleep(0.002)
                    continue

                if frame_i % DETECT_EVERY == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # downscale for faster detection
                    small = cv2.resize(gray, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA)
                    min_size_small = (max(1, int(MIN_SIZE_FULL[0] * DOWNSCALE)),
                                      max(1, int(MIN_SIZE_FULL[1] * DOWNSCALE)))
                    boxes_small = self._cascade.detectMultiScale(
                        small, scaleFactor=SCALE, minNeighbors=MIN_NEIGH, minSize=min_size_small
                    )
                    # map boxes back to full size
                    last_boxes_full = [(int(x / DOWNSCALE), int(y / DOWNSCALE),
                                        int(w / DOWNSCALE), int(h / DOWNSCALE)) for (x, y, w, h) in boxes_small]
                    self.faces = len(last_boxes_full)
                frame_i += 1

                if self.draw_boxes:
                    for (x, y, w, h) in last_boxes_full:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                ok, jpg = cv2.imencode(".jpg", frame, enc)
                if not ok:
                    continue
                blob = jpg.tobytes()

                n += 1
                dtt = time.time() - t0
                if dtt >= 1.0:
                    self.fps = round(n / dtt, 1)
                    t0 = time.time()
                    n = 0

                yield (b"--frame\r\n"
                      b"Content-Type: image/jpeg\r\n"
                      b"Cache-Control: no-cache\r\n"
                      b"Content-Length: " + str(len(blob)).encode() + b"\r\n\r\n"
                      + blob + b"\r\n")
        finally:
            with self.lock:
                self.running = False
            self._close_cam()

_RT = _FaceRuntime()

# ---------- API for index.py dispatcher ----------
def api_status(ctx: dict, **_) -> Dict[str, object]:
    return {
        "ok": True, "running": _RT.running, "fps": _RT.fps,
        "faces": _RT.faces, "overlay": _RT.draw_boxes   # <-- report overlay state
    }

def api_toggle(ctx: dict, enabled: bool | str | None = None, **_) -> Dict[str, object]:
    if isinstance(enabled, str):
        enabled = enabled.lower() in ("1", "true", "yes", "on")
    if enabled is None:
        return {"ok": False, "error": "missing 'enabled' boolean"}
    if enabled:
        with _RT.lock:
            _RT.stop_evt.clear()
            _RT.running = True
        return {"ok": True, "running": True}
    else:
        _RT.stop_evt.set()
        with _RT.lock:
            _RT.running = False
            _RT.fps = 0.0
            _RT.faces = 0
        return {"ok": True, "running": False}

def api_overlay(ctx: dict, enabled: bool | str | None = None, **_) -> Dict[str, object]:
    """Turn rectangles on/off without stopping the stream."""
    if isinstance(enabled, str):
        enabled = enabled.lower() in ("1", "true", "yes", "on")
    if enabled is None:
        return {"ok": False, "error": "missing 'enabled' boolean"}
    with _RT.lock:
        _RT.draw_boxes = bool(enabled)
    return {"ok": True, "overlay": _RT.draw_boxes}

def api_mjpg(ctx: dict, quality: int | str = 65, fps: float | str = 30.0, **_):
    from flask import Response
    try: q = int(quality)
    except Exception: q = 65
    try: f = float(fps)
    except Exception: f = 30.0
    return Response(
        _RT.gen(quality=q, fps_limit=f),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )
