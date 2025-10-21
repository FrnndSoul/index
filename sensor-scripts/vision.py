#!/usr/bin/env python3
from __future__ import annotations
import json, time, threading, pathlib, uuid
from typing import Dict, Tuple, Optional, List
import cv2, numpy as np
from flask import Response
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    from tensorflow.lite import Interpreter

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
RES_DIR = ROOT_DIR / "resources"
MODEL_DIR = RES_DIR / "models"
DB_DIR = RES_DIR / "profiles"
for p in (MODEL_DIR, DB_DIR):
    p.mkdir(parents=True, exist_ok=True)

OBJ_MODEL = MODEL_DIR / "detect.tflite"
LBL_PATH = MODEL_DIR / "labelmap.txt"

_cap: Optional[cv2.VideoCapture] = None
_cap_lock = threading.Lock()

def _ensure_capture(width=1280, height=720, fps=30):
    global _cap
    with _cap_lock:
        if _cap is None or (not _cap.isOpened()):
            cap = cv2.VideoCapture(0)
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
                cap.set(cv2.CAP_PROP_FPS, float(fps))
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
            except Exception:
                pass
            _cap = cap

def _read_frame() -> Optional[np.ndarray]:
    global _cap
    _ensure_capture()
    cap = _cap
    if cap is None:
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        time.sleep(0.02)
        with _cap_lock:
            try:
                if _cap is not None:
                    _cap.release()
            except Exception:
                pass
            _cap = None
        _ensure_capture()
        cap = _cap
        if cap is None:
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
    return frame

def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

def _load_labels(p: pathlib.Path) -> List[str]:
    try:
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            parts = s.split(maxsplit=1)
            out.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else s)
        return out
    except Exception:
        return []

OBJ_LABELS: List[str] = _load_labels(LBL_PATH)

def _load_ssd(model_path: pathlib.Path):
    itp = Interpreter(model_path=str(model_path))
    itp.allocate_tensors()
    iinfo = itp.get_input_details()[0]
    oinfo = itp.get_output_details()
    def pick(sub: str, default: int) -> int:
        idx = [i for i, d in enumerate(oinfo) if sub in d["name"].lower()]
        return idx[0] if idx else default
    bix = pick("box", 0)
    cix = pick("class", 1)
    six = pick("score", 2)
    nix = pick("count", 3 if len(oinfo) > 3 else 2)
    ih, iw = iinfo["shape"][1:3]
    idtype = iinfo["dtype"]
    return {"itp": itp, "iinfo": iinfo, "oinfo": oinfo, "bix": bix, "cix": cix, "six": six, "nix": nix, "ih": ih, "iw": iw, "idtype": idtype}

OBJ_SCORE_THRESH = 0.45
OBJ_NMS_IOU = 0.45

def _nms(dets, iou_thr=OBJ_NMS_IOU):
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d[5], reverse=True)
    keep = []
    def iou(a, b):
        ax1, ay1, aw, ah = a[:4]
        bx1, by1, bw, bh = b[:4]
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
        ua = aw * ah + bw * bh - inter
        return inter / ua if ua > 0 else 0.0
    while dets:
        m = dets.pop(0)
        keep.append(m)
        dets = [d for d in dets if iou(m, d) < iou_thr]
    return keep

class Runtime:
    def __init__(self):
        self.lock = threading.Lock()
        self.overlay_enabled = True
        self.allow: List[str] = []
        self.allow_norm: set[str] = set()
        self.fps = 0.0
        self.faces = 0
        self.objects = 0
        self.cascade: Optional[cv2.CascadeClassifier] = None
        self.obj_rt = None
        self.obj_lock = threading.Lock()
        self.profiles: List[Dict] = []
        self._fps_count = 0
        self._t0 = time.time()
    @staticmethod
    def _embed_from_crop(img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.equalizeHist(g)
        f = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype("float32").reshape(-1)
        f = (f - f.mean()) / (f.std() + 1e-6)
        n = np.linalg.norm(f) + 1e-6
        return f / n
    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-6) * (np.linalg.norm(b) + 1e-6)))
    def _load_profiles(self):
        profs = []
        for j in DB_DIR.glob("*.json"):
            try:
                meta = json.loads(j.read_text(encoding="utf-8"))
                vec = np.load(DB_DIR / f"{j.stem}.npy").astype("float32")
                meta["_vec"] = vec
                profs.append(meta)
            except Exception:
                pass
        self.profiles = profs
    def ensure_models(self):
        if self.cascade is None:
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.obj_rt is None and OBJ_MODEL.exists():
            self.obj_rt = _load_ssd(OBJ_MODEL)
        if not self.profiles:
            self._load_profiles()
    def infer_objects(self, frame_bgr: np.ndarray):
        if self.obj_rt is None:
            return []
        rt = self.obj_rt
        itp, iinfo, oinfo = rt["itp"], rt["iinfo"], rt["oinfo"]
        ih, iw = rt["ih"], rt["iw"]
        idtype = rt["idtype"]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
        if idtype == np.uint8:
            inp = inp.astype(np.uint8)
        else:
            inp = (inp.astype(np.float32) / 255.0).astype(idtype)
        with self.obj_lock:
            itp.set_tensor(iinfo["index"], inp[None, ...])
            itp.invoke()
            boxes = itp.get_tensor(oinfo[rt["bix"]]["index"])[0].copy()
            classes = itp.get_tensor(oinfo[rt["cix"]]["index"])[0].copy()
            scores = itp.get_tensor(oinfo[rt["six"]]["index"])[0].copy()
            if len(oinfo) > 3:
                count = int(itp.get_tensor(oinfo[rt["nix"]]["index"])[0])
            else:
                count = len(boxes)
        H, W = frame_bgr.shape[:2]
        raw = []
        for i in range(count):
            sc = float(scores[i])
            if sc < OBJ_SCORE_THRESH:
                continue
            cid = int(classes[i])
            name = OBJ_LABELS[cid] if 0 <= cid < len(OBJ_LABELS) else str(cid)
            if self.allow_norm and (_norm(name) not in self.allow_norm):
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            x, y = int(xmin * W), int(ymin * H)
            w, h = int((xmax - xmin) * W), int((ymax - ymin) * H)
            if w <= 0 or h <= 0:
                continue
            raw.append((x, y, w, h, cid, sc, name))
        return _nms(raw)
    def recognize(self, frame_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]]):
        if not self.profiles:
            return []
        out = []
        for (x, y, w, h) in boxes:
            crop = frame_bgr[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue
            vec = self._embed_from_crop(crop)
            best, name = 0.0, "unknown"
            for p in self.profiles:
                sc = self._cos(vec, p["_vec"])
                if sc > best:
                    best = sc
                    name = p.get("name", "unknown")
            out.append(((x, y, w, h), name, max(0.0, min(1.0, best))))
        return out
    def _draw(self, frame_bgr, faces, names, objs):
        if not self.overlay_enabled:
            return
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for ((x, y, w, h), name, sc) in names:
            lbl = f"{name} {int(sc*100)}%"
            cv2.putText(frame_bgr, lbl, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 220, 255), 2, cv2.LINE_AA)
        for (x, y, w, h, cid, sc, name) in objs:
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            lbl = f"{name} {int(sc*100)}%"
            cv2.putText(frame_bgr, lbl, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 220, 255), 2, cv2.LINE_AA)
    def _tick_fps(self):
        self._fps_count += 1
        dtt = time.time() - self._t0
        if dtt >= 1.0:
            self.fps = round(self._fps_count / dtt, 1)
            self._fps_count = 0
            self._t0 = time.time()

RT = Runtime()
PROFILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

def api_status(ctx: dict, **_) -> Dict[str, object]:
    with RT.lock:
        return {
            "ok": True,
            "fps": RT.fps,
            "faces": RT.faces,
            "objects": RT.objects,
            "overlay": RT.overlay_enabled,
            "allow": list(RT.allow),
            "vision": {"enabled": RT.overlay_enabled, "allow": list(RT.allow)},
        }

def api_overlay(ctx: dict, enabled: bool | str | None = None, **_) -> Dict[str, object]:
    if isinstance(enabled, str):
        enabled = enabled.lower() in ("1", "true", "yes", "on")
    if enabled is None:
        return {"ok": False, "error": "missing 'enabled' boolean"}
    with RT.lock:
        RT.overlay_enabled = bool(enabled)
    return {"ok": True, "overlay": RT.overlay_enabled}

def api_labels(ctx: dict, **_) -> Dict[str, object]:
    return {"ok": True, "labels": OBJ_LABELS}

def api_allow(ctx: dict, allow: List[str] | None = None, **_) -> Dict[str, object]:
    if allow is None:
        return {"ok": True, "allow": list(RT.allow)}
    labelset_norm = {_norm(s) for s in OBJ_LABELS}
    cleaned: List[str] = []
    normed: set[str] = set()
    for s in allow:
        raw = str(s).strip()
        n = _norm(raw)
        if n and n in labelset_norm:
            cleaned.append(raw)
            normed.add(n)
    with RT.lock:
        RT.allow = cleaned
        RT.allow_norm = normed
    return {"ok": True, "allow": list(RT.allow)}

def api_mjpg(ctx: dict, quality: int | str = 65, fps: float | str = 30.0, **_) -> Response:
    q = max(1, min(95, int(float(quality)))) if isinstance(quality, (int, float, str)) else 65
    fps_limit = float(fps) if fps not in (None, "", "0") else 0.0
    min_dt = 1.0 / fps_limit if fps_limit > 0 else 0.0
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(q)]
    boundary = b"--cam_boundary"
    RT.ensure_models()
    def gen():
        last_tick = 0.0
        face_boxes: List[Tuple[int,int,int,int]] = []
        name_tags: List[Tuple[Tuple[int,int,int,int], str, float]] = []
        objs: List[Tuple[int,int,int,int,int,float,str]] = []
        i = 0
        while True:
            if min_dt:
                now = time.time()
                dt = now - last_tick
                if dt < min_dt:
                    time.sleep(min_dt - dt)
                last_tick = time.time()
            frame = _read_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            if (i % 3) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                boxes_small = RT.cascade.detectMultiScale(small, scaleFactor=1.15, minNeighbors=4, minSize=(24, 24))
                face_boxes = [(int(x/0.5), int(y/0.5), int(w/0.5), int(h/0.5)) for (x, y, w, h) in boxes_small]
                RT.faces = len(face_boxes)
                name_tags = RT.recognize(frame, face_boxes)
            if (i % 3) == 0:
                objs = RT.infer_objects(frame)
                RT.objects = len(objs)
            i += 1
            RT._draw(frame, face_boxes, name_tags, objs)
            ok, jpg = cv2.imencode(".jpg", frame, enc)
            if not ok:
                continue
            blob = jpg.tobytes()
            RT._tick_fps()
            yield boundary + b"\r\n"
            yield b"Content-Type: image/jpeg\r\n"
            yield b"Content-Length: " + str(len(blob)).encode("ascii") + b"\r\n\r\n"
            yield blob + b"\r\n"
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=--cam_boundary")

def api_jpg(ctx: dict, quality: int | str = 65, **_) -> Response | Dict[str, object]:
    q = max(1, min(95, int(float(quality)))) if isinstance(quality, (int, float, str)) else 65
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(q)]
    RT.ensure_models()
    frame = _read_frame()
    if frame is None:
        return {"ok": False, "error": "no frame"}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    boxes_small = RT.cascade.detectMultiScale(small, scaleFactor=1.15, minNeighbors=4, minSize=(24, 24))
    faces = [(int(x/0.5), int(y/0.5), int(w/0.5), int(h/0.5)) for (x, y, w, h) in boxes_small]
    names = RT.recognize(frame, faces)
    objs = RT.infer_objects(frame)
    RT._draw(frame, faces, names, objs)
    ok, jpg = cv2.imencode(".jpg", frame, enc)
    if not ok:
        return {"ok": False, "error": "encode failed"}
    return Response(jpg.tobytes(), mimetype="image/jpeg")

def api_cam(ctx: dict, quality: int | str = 65, fps: float | str = 30.0, **_):
    return api_mjpg(ctx, quality=quality, fps=fps)

class _EnrollState:
    def __init__(self):
        self.session_id: Optional[str] = None
        self.tmp_vecs: List[np.ndarray] = []
        self.samples: int = 0
        self.target: int = 10
        self.status: str = "idle"

_EN = _EnrollState()

def _analyze_frame(frame):
    if frame is None:
        return {"ready": False, "lighting": "low", "distance": "closer", "center_ok": False, "pose": "front"}
    RT.ensure_models()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    boxes_small = RT.cascade.detectMultiScale(small, scaleFactor=1.15, minNeighbors=4, minSize=(24, 24))
    if len(boxes_small) == 0:
        m = gray.mean()
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return {"ready": False, "lighting": "low" if m < 70 else ("high" if m > 200 else "ok"), "distance": "closer", "center_ok": False, "pose": "front", "sharp": sharp}
    x0, y0, w0, h0 = max(boxes_small, key=lambda b: b[2] * b[3])
    x, y, w, h = int(x0/0.5), int(y0/0.5), int(w0/0.5), int(h0/0.5)
    H, W = frame.shape[:2]
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    area = (w * h) / float(W * H)
    m = gray.mean()
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lighting = "low" if m < 70 else ("high" if m > 200 else "ok")
    dist = "closer" if area < 0.12 else ("farther" if area > 0.45 else "ok")
    center_ok = (abs(cx - 0.5) < 0.15) and (abs(cy - 0.5) < 0.18)
    crop = frame[max(0, y):y + h, max(0, x):x + w]
    pose = "front"
    if crop is not None and crop.size > 0:
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.equalizeHist(g)
        r1 = PROFILE_CASCADE.detectMultiScale(g, scaleFactor=1.15, minNeighbors=3, minSize=(24, 24))
        r2 = PROFILE_CASCADE.detectMultiScale(cv2.flip(g, 1), scaleFactor=1.15, minNeighbors=3, minSize=(24, 24))
        def _has(rects):
            try:
                return len(rects) > 0
            except TypeError:
                return getattr(rects, "size", 0) > 0
        if _has(r1) or _has(r2):
            pose = "side"
    ready = (lighting == "ok") and (dist == "ok") and center_ok and pose == "front" and sharp > 30.0
    return {"ready": bool(ready), "lighting": lighting, "distance": dist, "center_ok": center_ok, "bbox": [int(x), int(y), int(w), int(h)], "pose": pose}

def api_enroll_start(ctx: dict, target: int | str = 10, **_):
    try:
        t = int(target)
    except Exception:
        t = 10
    _EN.session_id = uuid.uuid4().hex[:8]
    _EN.tmp_vecs = []
    _EN.samples = 0
    _EN.target = max(5, min(20, t))
    _EN.status = "hold"
    return {"ok": True, "session": _EN.session_id, "target": _EN.target}

def api_face_enroll_check(ctx: dict, **_):
    frame = _read_frame()
    res = _analyze_frame(frame)
    _EN.status = "capturing" if res.get("ready") else "hold"
    return {"ok": True, "session": _EN.session_id, "have_frame": frame is not None, "lighting": res.get("lighting", "low"), "distance": res.get("distance", "closer"), "center_ok": res.get("center_ok", False), "pose": res.get("pose", "front"), "ready": res.get("ready", False), "status": _EN.status, "samples": _EN.samples, "target": _EN.target}

def api_face_enroll_capture(ctx: dict, **_):
    if not _EN.session_id:
        return {"ok": False, "error": "no_session"}
    frame = _read_frame()
    res = _analyze_frame(frame)
    if not res.get("ready"):
        return {"ok": False, "error": "not_ready", "why": res}
    x, y, w, h = res["bbox"]
    crop = frame[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return {"ok": False, "error": "empty"}
    vec = RT._embed_from_crop(crop)
    _EN.tmp_vecs.append(vec)
    _EN.samples += 1
    done = _EN.samples >= _EN.target
    if done:
        _EN.status = "done"
    return {"ok": True, "samples": _EN.samples, "done": done}

def api_face_enroll_commit(ctx: dict, name: str | None = None, **_):
    if not _EN.session_id:
        return {"ok": False, "error": "no_session"}
    if not name or not str(name).strip():
        return {"ok": False, "error": "missing_name"}
    if not _EN.tmp_vecs:
        return {"ok": False, "error": "no_data"}
    emb = np.vstack(_EN.tmp_vecs).mean(axis=0).astype("float32")
    slug = "".join(c for c in str(name).strip().lower() if (c.isalnum() or c in "-_")).strip("-_") or "user"
    pid = f"{slug}_{_EN.session_id}"
    meta = {"id": pid, "name": str(name).strip(), "created": int(time.time() * 1000), "samples": int(len(_EN.tmp_vecs))}
    (DB_DIR / f"{pid}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(DB_DIR / f"{pid}.npy", emb)
    RT._load_profiles()
    _EN.session_id = None
    _EN.tmp_vecs = []
    _EN.samples = 0
    _EN.target = 10
    _EN.status = "idle"
    return {"ok": True, "id": pid}
