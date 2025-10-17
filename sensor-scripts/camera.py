from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Tuple, List
import os, time
import numpy as np
import cv2
from flask import Response, make_response

# ========= Camera backend (shared buffer) =========
def _ensure_started(ctx: Dict[str, Any]):
    # your existing worker
    from resources.camera_motion import ensure_started
    ensure_started(ctx)

def _get_latest_bgr() -> Optional[np.ndarray]:
    import resources.camera_motion as cm
    for name in ("get_latest_bgr","get_latest_frame_bgr","get_latest_frame","get_bgr","latest_bgr"):
        g = getattr(cm, name, None)
        if callable(g):
            try:
                f = g()
                if f is not None: return np.asarray(f)
            except Exception:
                continue
        elif g is not None:
            return np.asarray(g)
    return None

# ========= MJPEG helpers =========
_BOUNDARY = "cam_boundary"

def _multipart(gen: Iterable[bytes]) -> Response:
    return Response(gen, mimetype=f"multipart/x-mixed-replace; boundary=--{_BOUNDARY}")

def _yield_part(jpg: bytes):
    yield (
        f"--{_BOUNDARY}\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(jpg)}\r\n\r\n"
    ).encode("utf-8") + jpg + b"\r\n"

def _encode_jpeg(bgr: np.ndarray, quality: int = 80) -> bytes:
    q = max(1, min(95, int(quality)))
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok: raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()

# ========= Engines =========
# ---- Haar face (built-in, light) ----
_HAAR = None
def _haar():
    global _HAAR
    if _HAAR is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _HAAR = cv2.CascadeClassifier(path)
        if _HAAR.empty():
            raise RuntimeError(f"Failed to load Haar cascade at {path}")
    return _HAAR

def _haar_faces(gray: np.ndarray, scale: float, neighbors: int, minsz: Tuple[int,int]):
    rects = _haar().detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors,
                                     flags=cv2.CASCADE_SCALE_IMAGE, minSize=minsz)
    # de-duplicate overlapping hits
    r = rects.tolist()
    if len(r) == 0: return rects
    grouped, _weights = cv2.groupRectangles(r + r, groupThreshold=1, eps=0.2)
    return np.array(grouped) if len(grouped) else rects

# ---- TFLite (optional) ----
_TFL_FACE = None
_TFL_FACE_IN = None
_TFL_OBJ = None
_TFL_OBJ_IN = None
_TFL_OBJ_LABELS = None

def _maybe_load_tflite_face():
    """BlazeFace/MediaPipe-style face detector; set CAM_TFLITE_FACE=/path/model.tflite"""
    global _TFL_FACE, _TFL_FACE_IN
    if _TFL_FACE is not None: return True
    path = os.environ.get("CAM_TFLITE_FACE", "")
    if not path or not os.path.exists(path): return False
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        try:
            from tensorflow.lite import Interpreter  # fallback if full TF available
        except Exception:
            return False
    _TFL_FACE = Interpreter(model_path=path)
    _TFL_FACE.allocate_tensors()
    ins = _TFL_FACE.get_input_details()[0]
    _TFL_FACE_IN = (ins["index"], ins["shape"][1], ins["shape"][2])  # (idx,H,W)
    return True

def _maybe_load_tflite_obj():
    """SSD MobileNet (COCO) style detector; set CAM_TFLITE_OBJ=/path/model.tflite and optional CAM_TFLITE_LABELS"""
    global _TFL_OBJ, _TFL_OBJ_IN, _TFL_OBJ_LABELS
    if _TFL_OBJ is not None: return True
    path = os.environ.get("CAM_TFLITE_OBJ", "")
    if not path or not os.path.exists(path): return False
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        try:
            from tensorflow.lite import Interpreter
        except Exception:
            return False
    _TFL_OBJ = Interpreter(model_path=path)
    _TFL_OBJ.allocate_tensors()
    ins = _TFL_OBJ.get_input_details()[0]
    _TFL_OBJ_IN = (ins["index"], ins["shape"][1], ins["shape"][2])  # (idx,H,W)

    labels_path = os.environ.get("CAM_TFLITE_LABELS", "")
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            _TFL_OBJ_LABELS = [line.strip() for line in f if line.strip()]
    return True

def _run_tflite_face(bgr: np.ndarray, score_th: float) -> List[Tuple[int,int,int,int,float]]:
    """Return [(x,y,w,h,score), ...] in image coordinates."""
    if not _maybe_load_tflite_face(): return []
    idx, H, W = _TFL_FACE_IN
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (W, H))
    inp = np.expand_dims(resized, 0).astype(np.float32) / 255.0
    _TFL_FACE.set_tensor(idx, inp)
    _TFL_FACE.invoke()
    # Common BlazeFace-style outputs (adapt if your model differs):
    # boxes [N,4] in [ymin, xmin, ymax, xmax] normalized; scores [N]; choose first output names robustly
    outs = _TFL_FACE.get_output_details()
    # heuristics: find 2 arrays: one with shape [N,4], one with shape [N]
    o1 = _TFL_FACE.get_tensor(outs[0]["index"])
    o2 = _TFL_FACE.get_tensor(outs[1]["index"])
    boxes = o1 if o1.shape[-1] == 4 else o2
    scores = o2 if o1.shape[-1] == 4 else o1
    h, w = bgr.shape[:2]
    results = []
    for i in range(min(len(scores), len(boxes))):
        s = float(scores[i])
        if s < score_th: continue
        ymin, xmin, ymax, xmax = [float(v) for v in boxes[i]]
        x, y = int(xmin * w), int(ymin * h)
        xe, ye = int(xmax * w), int(ymax * h)
        results.append((x, y, max(0, xe-x), max(0, ye-y), s))
    return results

def _run_tflite_obj(bgr: np.ndarray, score_th: float, classes: Optional[List[str]]=None) -> List[Tuple[int,int,int,int,float,str]]:
    """Return [(x,y,w,h,score,label), ...]. If classes provided, filter by those labels."""
    if not _maybe_load_tflite_obj(): return []
    idx, H, W = _TFL_OBJ_IN
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (W, H))
    inp = np.expand_dims(resized, 0).astype(np.float32)
    _TFL_OBJ.set_tensor(idx, inp)
    _TFL_OBJ.invoke()
    outs = _TFL_OBJ.get_output_details()
    # Typical SSD outputs (adapt if different): boxes[N,4], class_ids[N], scores[N], num[1]
    boxes   = _TFL_OBJ.get_tensor(outs[0]["index"])[0]
    classes = _TFL_OBJ.get_tensor(outs[1]["index"])[0].astype(int)
    scores  = _TFL_OBJ.get_tensor(outs[2]["index"])[0]
    h, w = bgr.shape[:2]
    results = []
    for i, s in enumerate(scores):
        s = float(s)
        if s < score_th: continue
        ymin, xmin, ymax, xmax = [float(v) for v in boxes[i]]
        x, y = int(xmin * w), int(ymin * h)
        xe, ye = int(xmax * w), int(ymax * h)
        label = str(classes[i])
        if _TFL_OBJ_LABELS and 0 <= classes[i] < len(_TFL_OBJ_LABELS):
            label = _TFL_OBJ_LABELS[classes[i]]
        results.append((x, y, max(0, xe-x), max(0, ye-y), s, label))
    return results

# ========= Drawing =========
def _draw_boxes(bgr: np.ndarray, rects, color=(0,200,255), thick=2, label: Optional[str]=None):
    for r in rects:
        if len(r) >= 4:
            x,y,w,h = map(int, r[:4])
        else:
            continue
        cv2.rectangle(bgr, (x,y), (x+w, y+h), color, thick)
        if label or (len(r) >= 5):
            txt = label
            if not txt and len(r) >= 6:  # obj: (x,y,w,h,score,label)
                txt = f"{str(r[5])} {int(100*r[4])}%"
            elif not txt and len(r) == 5:  # face: (x,y,w,h,score)
                txt = f"{int(100*r[4])}%"
            if txt:
                cv2.putText(bgr, str(txt), (x, max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
    return bgr

# ========= Public APIs =========
def api_cam(ctx: Dict[str, Any], quality: str = "80", fps: str | None = None, swap: str = "0", **_):
    """
    Single MJPEG endpoint. If vision.enabled is False -> stream raw (fast path using your backend).
    If enabled -> draw overlay with selected engine (haar_face | tflite_face | tflite_obj).
    Never opens /dev/video*; uses shared camera buffer.
    """
    _ensure_started(ctx)

    vis = (ctx.get("state", {}).get("vision") or {})
    enabled = bool(vis.get("enabled", False))
    engine  = str(vis.get("engine", "haar_face")).lower()
    q = max(1, min(95, int(float(quality) if quality else 80)))
    f = int(float(fps)) if (fps not in (None, "", "0")) else None
    sw = str(swap).lower() in ("1","true","yes")

    # If not enabled -> use your known-good backend MJPEG
    if not enabled:
        from resources.camera_motion import mjpeg_stream_response
        return mjpeg_stream_response(quality=q, max_fps=f, swap=sw)

    minperiod = (1.0 / f) if f else 0.0
    score_th = float(vis.get("score", 0.5))
    scale    = float(vis.get("scale", 1.2))
    neigh    = int(vis.get("neighbors", 5))
    min_w, min_h = vis.get("minsize", [40,40])
    minsz = (int(min_w), int(min_h))

    # warmup for first frame (so <img> doesn't hang)
    t0 = time.time(); bgr0 = None
    while bgr0 is None and (time.time() - t0) < 0.5:
        bgr0 = _get_latest_bgr();  time.sleep(0.01) if bgr0 is None else None
    if bgr0 is None:
        # fallback to raw if buffer isn't ready
        from resources.camera_motion import mjpeg_stream_response
        return mjpeg_stream_response(quality=q, max_fps=f, swap=sw)
    if sw: bgr0 = cv2.flip(bgr0, 1)

    # run engine on first frame
    if engine == "haar_face":
        rects0 = _haar_faces(cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY), scale, neigh, minsz)
        _draw_boxes(bgr0, [(x,y,w,h) for (x,y,w,h) in rects0])
    elif engine == "tflite_face":
        rects0 = _run_tflite_face(bgr0, score_th)
        _draw_boxes(bgr0, rects0)
    elif engine == "tflite_obj":
        rects0 = _run_tflite_obj(bgr0, score_th)  # include labels
        _draw_boxes(bgr0, rects0)
    jpg_first = _encode_jpeg(bgr0, q)

    def gen():
        yield from _yield_part(jpg_first)
        last = time.time(); last_jpg = jpg_first
        while True:
            if minperiod:
                now = time.time(); d = now - last
                if d < minperiod: time.sleep(max(0.0, minperiod - d))
                last = time.time()
            bgr = _get_latest_bgr()
            if bgr is None:
                yield from _yield_part(last_jpg);  continue
            if sw: bgr = cv2.flip(bgr, 1)

            if engine == "haar_face":
                rects = _haar_faces(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), scale, neigh, minsz)
                _draw_boxes(bgr, [(x,y,w,h) for (x,y,w,h) in rects])
            elif engine == "tflite_face":
                _draw_boxes(bgr, _run_tflite_face(bgr, score_th))
            else: # tflite_obj
                _draw_boxes(bgr, _run_tflite_obj(bgr, score_th))

            last_jpg = _encode_jpeg(bgr, q)
            yield from _yield_part(last_jpg)

    return _multipart(gen())

def api_jpg(ctx: Dict[str, Any], quality: str = "80", swap: str = "0", **_) -> Response:
    _ensure_started(ctx)
    vis = (ctx.get("state", {}).get("vision") or {})
    enabled = bool(vis.get("enabled", False))
    engine  = str(vis.get("engine", "haar_face")).lower()

    # wait a moment for a frame
    deadline = time.time() + 1.0
    bgr = None
    while bgr is None and time.time() < deadline:
        bgr = _get_latest_bgr()
        if bgr is None: time.sleep(0.01)
    if bgr is None:
        resp = make_response(b'no frame', 503); resp.headers["Content-Type"]="text/plain"; return resp

    if str(swap).lower() in ("1","true","yes"): bgr = cv2.flip(bgr, 1)

    if enabled:
        score_th = float(vis.get("score", 0.5))
        scale    = float(vis.get("scale", 1.2))
        neigh    = int(vis.get("neighbors", 5))
        min_w, min_h = vis.get("minsize", [40,40]); minsz = (int(min_w), int(min_h))
        if engine == "haar_face":
            rects = _haar_faces(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), scale, neigh, minsz)
            _draw_boxes(bgr, [(x,y,w,h) for (x,y,w,h) in rects])
        elif engine == "tflite_face":
            _draw_boxes(bgr, _run_tflite_face(bgr, score_th))
        else:
            _draw_boxes(bgr, _run_tflite_obj(bgr, score_th))

    jpg = _encode_jpeg(bgr, int(float(quality or 80)))
    resp = make_response(jpg)
    resp.headers["Content-Type"] = "image/jpeg"
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp

def api_health(ctx: Dict[str, Any], **_) -> Dict[str, Any]:
    vis = (ctx.get("state", {}).get("vision") or {})
    return {"ok": True, "vision": vis, "time_ms": int(time.time()*1000)}

# -------- Vision control (toggle / engine / params) --------
def api_status(ctx: Dict[str, Any], **_) -> Dict[str, Any]:
    return {"ok": True, "vision": ctx.get("state", {}).get("vision", {})}

def api_config(ctx: Dict[str, Any],
               enabled: Any = None,
               engine: Any = None,
               score: Any = None,
               scale: Any = None,
               neighbors: Any = None,
               minsize: Any = None,
               **_) -> Dict[str, Any]:
    vis = ctx.setdefault("state", {}).setdefault("vision", {
        "enabled": False, "engine": "haar_face",
        "score": 0.5, "scale": 1.2, "neighbors": 5, "minsize": [40,40]
    })

    if enabled is not None:
        s = str(enabled).strip().lower()
        vis["enabled"] = s in ("1","true","yes","on","enable","enabled")
    if engine is not None:
        e = str(engine).lower()
        if e in ("haar_face","tflite_face","tflite_obj"):
            vis["engine"] = e
    if score is not None:
        try: vis["score"] = float(score)
        except: pass
    if scale is not None:
        try: vis["scale"] = float(scale)
        except: pass
    if neighbors is not None:
        try: vis["neighbors"] = int(neighbors)
        except: pass
    if minsize is not None:
        try:
            if isinstance(minsize, str) and "x" in minsize.lower():
                w,h = (int(x) for x in minsize.lower().split("x",1))
                vis["minsize"] = [w,h]
            elif isinstance(minsize,(list,tuple)) and len(minsize)==2:
                vis["minsize"] = [int(minsize[0]), int(minsize[1])]
        except: pass

    return {"ok": True, "vision": vis}
