from __future__ import annotations
from typing import Any, Dict
from resources.camera_motion import ensure_started, mjpeg_stream_response

def api_cam(ctx: Dict[str, Any], quality: str = "80", fps: str | None = None, swap: str = "0", **_):
    """
    Return a Flask Response (multipart/x-mixed-replace) for MJPEG.
    IMPORTANT: index.py must NOT jsonify() this. See route note below.
    """
    ensure_started(ctx)
    q = max(1, min(95, int(float(quality) if quality else 80)))
    f = int(float(fps)) if (fps not in (None, "", "0")) else None
    sw = str(swap).lower() in ("1", "true", "yes")
    return mjpeg_stream_response(quality=q, max_fps=f, swap=sw)
  
def _open(self):
    with self._cap_lock:
        if self._cap is None:
            import cv2
            self._cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
            # Ask for MJPG @ 640x480 (or 320x240 for easier 30fps)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            # Keep only 1 queued frame to reduce lag
            try: self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception: pass

