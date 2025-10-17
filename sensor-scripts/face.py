#!/usr/bin/env python3
from __future__ import annotations
import os, json, uuid, time, threading, pathlib
from typing import Dict, Tuple, Optional, List
import cv2, numpy as np
from tflite_runtime.interpreter import Interpreter

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
RES_DIR = ROOT_DIR / "resources"
MODELS_DIR = RES_DIR / "models"
DB_DIR = RES_DIR / "profiles"
SESS_DIR = RES_DIR / "enroll_sessions"
for p in (DB_DIR, SESS_DIR): p.mkdir(parents=True, exist_ok=True)

try:
    from .camera import CameraManager
    _HAS_CAM_MGR = True
except Exception:
    _HAS_CAM_MGR = False

FACE_DET = MODELS_DIR / "face.tflite"
FACE_EMB = MODELS_DIR / "face_embed.tflite"

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-6) * (np.linalg.norm(b) + 1e-6)))

def _load_profiles() -> List[Dict]:
    out = []
    for j in DB_DIR.glob("*.json"):
        try:
            meta = json.loads(j.read_text(encoding="utf-8"))
            vec = np.load(DB_DIR / f"{j.stem}.npy").astype("float32")
            meta["_vec"] = vec
            out.append(meta)
        except Exception:
            pass
    return out

def _load_ssd_interpreter(path: pathlib.Path):
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    iin = it.get_input_details()[0]
    outs = it.get_output_details()
    if len(outs) >= 4:
        boxes_idx = None
        count_idx = None
        for i, d in enumerate(outs):
            shp = list(d["shape"])
            if len(shp) == 3 and shp[-1] == 4:
                boxes_idx = i
            if int(np.prod(shp)) == 1:
                count_idx = i
        if boxes_idx is None: boxes_idx = 0
        if count_idx is None: count_idx = 3 if len(outs) > 3 else len(outs)-1
        rest = [i for i in range(len(outs)) if i not in (boxes_idx, count_idx)]
        classes_idx = rest[0] if rest else None
        scores_idx  = rest[1] if len(rest) > 1 else (rest[0] if rest else None)
    else:
        boxes_idx, classes_idx, scores_idx, count_idx = 0, 1, 2, 3
    ih, iw = iin["shape"][1:3]
    return it, iin, outs, boxes_idx, classes_idx, scores_idx, count_idx, ih, iw

def _detect_faces_tfl(frame: np.ndarray, det_runtime):
    it, iin, outs, bix, cix, six, nix, ih, iw = det_runtime
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR).astype(np.uint8)[None, ...]
    it.set_tensor(iin["index"], inp)
    it.invoke()
    boxes  = it.get_tensor(outs[bix]["index"])[0]
    scores = it.get_tensor(outs[six]["index"])[0] if six is not None else None
    count  = int(it.get_tensor(outs[nix]["index"])[0]) if nix is not None else len(boxes)
    H, W = frame.shape[:2]
    out = []
    for i in range(count):
        sc = float(scores[i]) if scores is not None else 1.0
        if sc < 0.5: continue
        ymin, xmin, ymax, xmax = boxes[i]
        x, y = int(xmin*W), int(ymin*H)
        w, h = int((xmax-xmin)*W), int((ymax-ymin)*H)
        out.append((x, y, w, h))
    return out

def _load_embed_interpreter(path: pathlib.Path):
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    iin = it.get_input_details()[0]
    o = it.get_output_details()[0]
    eh, ew = iin["shape"][1:3]
    return it, iin, o, eh, ew

def _embed_with_tflite(bgr_crop: np.ndarray, emb_runtime):
    it, iin, o, eh, ew = emb_runtime
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (ew, eh), interpolation=cv2.INTER_AREA).astype("float32")/255.0
    it.set_tensor(iin["index"], inp[None, ...])
    it.invoke()
    v = it.get_tensor(o["index"]).astype("float32").reshape(-1)
    v /= (np.linalg.norm(v) + 1e-6)
    return v

class _EnrollState:
    def __init__(self):
        self.session_id: Optional[str] = None
        self.tmp_vecs: List[np.ndarray] = []
        self.samples: int = 0
        self.target: int = 10
        self.status: str = "idle"

_EN = _EnrollState()

class _FaceRuntime:
    def __init__(self):
        self.target_fps = 30
        self.lock = threading.Lock()
        self.running = False
        self.stop_evt = threading.Event()
        self.fps: float = 0.0
        self.faces: int = 0
        self.draw_boxes: bool = True
        self.cam_index = 0
        self.size: Tuple[int, int] = (640, 480)
        self._cap = None
        self._cam_mgr = None
        self._last_frame = None
        self._profiles: List[Dict] = []
        self._last_names: List[Tuple[Tuple[int,int,int,int],str,float]] = []
        self._det_rt = None
        self._emb_rt = None
        self._det_lock = threading.Lock()

    def _open_cam(self):
        if _HAS_CAM_MGR:
            self._cam_mgr = CameraManager(cam_index=self.cam_index, width=self.size[0], height=self.size[1])
            self._cam_mgr._open()
            self._cam_mgr.acquire()
            return
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            self._cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
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
        return self._cam_mgr.read() if _HAS_CAM_MGR else self._cap.read()

    def _ensure_models(self):
        if self._det_rt is None:
            self._det_rt = _load_ssd_interpreter(FACE_DET)
        if self._emb_rt is None and FACE_EMB.exists():
            self._emb_rt = _load_embed_interpreter(FACE_EMB)

    def _embed(self, crop: np.ndarray) -> np.ndarray:
        if self._emb_rt is not None:
            return _embed_with_tflite(crop, self._emb_rt)
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.equalizeHist(g)
        f = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype("float32").reshape(-1)
        f = (f - f.mean()) / (f.std() + 1e-6)
        f /= (np.linalg.norm(f) + 1e-6)
        return f

    def _recognize(self, frame, boxes):
        if not self._profiles: return []
        out = []
        for (x,y,w,h) in boxes:
            crop = frame[max(0,y):y+h, max(0,x):x+w]
            if crop.size == 0: continue
            vec = self._embed(crop)
            best, name = 0.0, "unknown"
            for p in self._profiles:
                pv = p["_vec"]
                if pv.shape != vec.shape:
                    continue
                sc = _cos(vec, pv)
                if sc > best:
                    best = sc; name = p.get("name","unknown")
            out.append(((x,y,w,h), name, max(0.0, min(1.0, best))))
        return out

    def _faces_from_frame(self, frame: np.ndarray):
        with self._det_lock:
            return _detect_faces_tfl(frame, self._det_rt)

    def gen(self, quality: int = 60, fps_limit: float = 30.0):
        DETECT_EVERY = 3
        with self.lock:
            self.stop_evt.clear()
            self.running = True
            self.fps = 0.0
            self.faces = 0
            self._profiles = _load_profiles()
        self._open_cam()
        self._ensure_models()
        frame_i = 0
        last_boxes_full = []
        t0 = time.time()
        n = 0
        min_dt = 1.0 / fps_limit if fps_limit and fps_limit > 0 else 0.0
        last_tick = 0.0
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        try:
            while not self.stop_evt.is_set():
                if min_dt:
                    now = time.time()
                    dt = now - last_tick
                    if dt < min_dt: time.sleep(min_dt - dt)
                    last_tick = time.time()
                ok, frame = self._read()
                if not ok or frame is None:
                    time.sleep(0.002); continue
                if frame_i % DETECT_EVERY == 0:
                    last_boxes_full = self._faces_from_frame(frame)
                    self.faces = len(last_boxes_full)
                    self._last_names = self._recognize(frame, last_boxes_full)
                frame_i += 1
                if self.draw_boxes:
                    for (x,y,w,h) in last_boxes_full:
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    for ((x,y,w,h), name, sc) in self._last_names:
                        lbl = f"{name} • {int(sc*100)}%"
                        cv2.putText(frame, lbl, (x, max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20,220,255), 2, cv2.LINE_AA)
                self._last_frame = frame.copy()
                ok, jpg = cv2.imencode(".jpg", frame, enc)
                if not ok: continue
                blob = jpg.tobytes()
                n += 1
                dtt = time.time() - t0
                if dtt >= 1.0:
                    self.fps = round(n / dtt, 1)
                    t0 = time.time()
                    n = 0
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\nContent-Length: "
                       + str(len(blob)).encode() + b"\r\n\r\n" + blob + b"\r\n")
        finally:
            with self.lock:
                self.running = False
            self._close_cam()

_RT = _FaceRuntime()

def _analyze(frame):
    if frame is None: return {"ready": False, "reason": "no_frame"}
    dets = _RT._faces_from_frame(frame)
    if not dets:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m = gray.mean()
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return {"ready": False, "reason": "no_face", "lighting": "low" if m < 70 else ("high" if m > 200 else "ok"), "sharp": sharp, "center_ok": False, "distance": "closer"}
    x,y,w,h = max(dets, key=lambda b: b[2]*b[3])
    H, W = frame.shape[:2]
    cx = (x + w/2) / W
    cy = (y + h/2) / H
    area = (w*h) / float(W*H)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    m = gray.mean()
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lighting = "low" if m < 70 else ("high" if m > 200 else "ok")
    dist = "closer" if area < 0.12 else ("farther" if area > 0.45 else "ok")
    center_ok = (abs(cx-0.5) < 0.15) and (abs(cy-0.5) < 0.18)
    ready = (lighting == "ok") and (dist == "ok") and center_ok and sharp > 30.0
    return {"ready": bool(ready), "lighting": lighting, "distance": dist, "center_ok": center_ok, "bbox": [int(x),int(y),int(w),int(h)]}

def api_status(ctx: dict, **_) -> Dict[str, object]:
    return {"ok": True, "running": _RT.running, "fps": _RT.fps, "faces": _RT.faces, "overlay": _RT.draw_boxes}

def api_overlay(ctx: dict, enabled: bool | str | None = None, **_) -> Dict[str, object]:
    if isinstance(enabled, str): enabled = enabled.lower() in ("1","true","yes","on")
    if enabled is None: return {"ok": False, "error": "missing 'enabled' boolean"}
    with _RT.lock: _RT.draw_boxes = bool(enabled)
    return {"ok": True, "overlay": _RT.draw_boxes}

def api_mjpg(ctx: dict, quality: int | str = 65, fps: float | str = 30.0, **_):
    from flask import Response
    try: q = int(quality)
    except Exception: q = 65
    try: f = float(fps)
    except Exception: f = 30.0
    with _RT.lock: _RT.target_fps = int(max(5, min(60, f)))
    return Response(_RT.gen(quality=q, fps_limit=f), mimetype="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})

def api_enroll_start(ctx: dict, target: int | str = 10, **_):
    try: t = int(target)
    except Exception: t = 10
    _EN.session_id = uuid.uuid4().hex[:8]
    _EN.tmp_vecs = []
    _EN.samples = 0
    _EN.target = max(5, min(20, t))
    _EN.status = "hold"
    return {"ok": True, "session": _EN.session_id, "target": _EN.target}

def api_enroll_check(ctx: dict, **_):
    f = _RT._last_frame
    res = _analyze(f)
    if res.get("ready"): _EN.status = "capturing"
    else: _EN.status = "hold"
    blink = True
    return {"ok": True, "session": _EN.session_id, "have_frame": f is not None, "lighting": res.get("lighting","low"), "distance": res.get("distance","closer"), "center_ok": res.get("center_ok", False), "ready": res.get("ready", False), "status": _EN.status, "blink": blink, "samples": _EN.samples, "target": _EN.target}

def api_enroll_capture(ctx: dict, **_):
    if not _EN.session_id: return {"ok": False, "error": "no_session"}
    f = _RT._last_frame
    res = _analyze(f)
    if not res.get("ready"): return {"ok": False, "error": "not_ready", "why": res}
    x,y,w,h = res["bbox"]
    crop = f[max(0,y):y+h, max(0,x):x+w]
    if crop.size == 0: return {"ok": False, "error": "empty"}
    vec = _RT._embed(crop)
    _EN.tmp_vecs.append(vec)
    _EN.samples += 1
    done = _EN.samples >= _EN.target
    if done: _EN.status = "done"
    return {"ok": True, "samples": _EN.samples, "done": done}

def api_enroll_commit(ctx: dict, name: str | None = None, **_):
    if not _EN.session_id: return {"ok": False, "error": "no_session"}
    if not name or not str(name).strip(): return {"ok": False, "error": "missing_name"}
    if not _EN.tmp_vecs: return {"ok": False, "error": "no_data"}
    emb = np.vstack(_EN.tmp_vecs).mean(axis=0).astype("float32")
    slug = "".join(c for c in name.strip().lower() if (c.isalnum() or c in "-_")).strip("-_") or "user"
    pid = f"{slug}_{_EN.session_id}"
    meta = {"id": pid, "name": name.strip(), "created": int(time.time()*1000), "samples": int(len(_EN.tmp_vecs))}
    (DB_DIR / f"{pid}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(DB_DIR / f"{pid}.npy", emb)
    with _RT.lock: _RT._profiles = _load_profiles()
    _EN.session_id = None; _EN.tmp_vecs = []; _EN.samples = 0; _EN.target = 10; _EN.status = "idle"
    return {"ok": True, "id": pid}
