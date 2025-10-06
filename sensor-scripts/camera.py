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
