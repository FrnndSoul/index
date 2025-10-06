# pir-motion.py
from __future__ import annotations
import time
from flask import Response
from typing import Any, Dict

# All the heavy lifting lives in resources
from resources.camera_motion import (
    ensure_started,
    get_status,
    get_config,
    set_config,
    mjpeg_stream_response,
    pir_snapshot,
)

# --- API surface expected by index.py generic dispatcher ---
# GET  /api/pir/status        -> api_status
# GET  /api/pir               -> api_pir        (current PIR value + since_ms)
# GET  /api/pir/config        -> api_config     (read)
# POST /api/pir/config        -> api_config_set (write)
# GET  /api/pir/cam           -> api_cam        (MJPEG Response passthrough)

def api_status(ctx: Dict[str, Any], **kwargs):
    ensure_started(ctx)
    return get_status()

def api_pir(ctx: Dict[str, Any], **kwargs):
    ensure_started(ctx)
    val, since_ms = pir_snapshot()
    return {
        "ok": True,
        "ts": int(time.time() * 1000),
        "value": int(val),
        "state": "motion" if val else "clear",
        "since_ms": int(since_ms),
    }

def api_config(ctx: Dict[str, Any], **kwargs):
    ensure_started(ctx)
    return {"ok": True, **get_config()}

def api_config_set(ctx: Dict[str, Any], **kwargs):
    # accepts keys: intensity_min, pixel_delta, k_sigma, sample_hz, email_cooldown_s
    ensure_started(ctx)
    updated = set_config(**kwargs)
    return {"ok": True, "cfg": updated}

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
