#!/usr/bin/env python3
from __future__ import annotations
import threading, pathlib, time
from typing import Dict, Tuple, Optional, List, Iterable
import cv2, numpy as np
from tflite_runtime.interpreter import Interpreter

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
RES_DIR = ROOT_DIR / "resources"
MODELS_DIR = RES_DIR / "models"
DET_PATH = MODELS_DIR / "detect.tflite"
LBL_PATH = MODELS_DIR / "labelmap.txt"

try:
    from .camera import CameraManager
    _HAS_CAM_MGR = True
except Exception:
    _HAS_CAM_MGR = False

def _load_labels(p: pathlib.Path) -> List[str]:
    try:
        lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines()]
        return [l for l in lines if l]
    except Exception:
        return []

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

def _infer_ssd(frame_bgr: np.ndarray, rt, score_thr: float = 0.5):
    it, iin, outs, bix, cix, six, nix, ih, iw = rt
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR).astype(np.uint8)[None, ...]
    it.set_tensor(iin["index"], inp)
    it.invoke()
    boxes  = it.get_tensor(outs[bix]["index"])[0]
    classes= it.get_tensor(outs[cix]["index"])[0] if cix is not None else None
    scores = it.get_tensor(outs[six]["index"])[0] if six is not None else None
    count  = int(it.get_tensor(outs[nix]["index"])[0]) if nix is not None else len(boxes)
    H, W = frame_bgr.shape[:2]
    dets = []
    for i in range(count):
        sc = float(scores[i]) if scores is not None else 1.0
        if sc < score_thr: continue
        cls = int(classes[i]) if classes is not None else 0
        ymin, xmin, ymax, xmax = boxes[i]
        x, y = int(xmin*W), int(ymin*H)
        w, h = int((xmax-xmin)*W), int((ymax-ymin)*H)
        dets.append((x, y, w, h, cls, sc))
    return dets

class _ObjRuntime:
    def __init__(self):
        self.lock = threading.Lock()
        self.det_lock = threading.Lock()
        self.running = False
        self.stop_evt = threading.Event()
        self.fps: float = 0.0
        self.draw_boxes: bool = True
        self.cam_index = 0
        self.size: Tuple[int, int] = (640, 480)
        self.target_fps = 30
        self._cap = None
        self._cam_mgr = None
        self._det_rt = None
        self._labels: List[str] = _load_labels(LBL_PATH)
        self._labels_lc: List[str] = [s.lower() for s in self._labels]
        self._allowed: Optional[set[str]] = None
        self._last_det: List[Tuple[int,int,int,int,int,float]] = []

    def _open_cam(self):
        if _HAS_CAM_MGR:
            self._cam_mgr = CameraManager(cam_index=self.cam_index, width=self.size[0], height=self.size[1])
            self._cam_mgr._open()
            self._cam_mgr.acquire()
            return
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.size[0])
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

    def _ensure_model(self):
        if self._det_rt is None:
            self._det_rt = _load_ssd_interpreter(DET_PATH)

    def _name_for(self, cid: int) -> str:
        return self._labels[cid] if 0 <= cid < len(self._labels) else str(cid)

    def _is_allowed(self, name: str) -> bool:
        if not self._allowed or len(self._allowed) == 0:
            return True
        return name.lower() in self._allowed

    def _detect(self, frame):
        with self.det_lock:
            raw = _infer_ssd(frame, self._det_rt, 0.5)
        if not self._allowed:
            return raw
        out = []
        for (x,y,w,h,cid,sc) in raw:
            nm = self._name_for(cid)
            if self._is_allowed(nm):
                out.append((x,y,w,h,cid,sc))
        return out

    def gen(self, quality: int = 65, fps_limit: float = 30.0):
        DETECT_EVERY = 3
        with self.lock:
            self.stop_evt.clear()
            self.running = True
            self.fps = 0.0
        self._open_cam()
        self._ensure_model()
        frame_i = 0
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
                    self._last_det = self._detect(frame)
                frame_i += 1
                if self.draw_boxes:
                    for (x,y,w,h,cid,sc) in self._last_det:
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                        name = self._name_for(cid)
                        lbl = f"{name} {int(sc*100)}%"
                        cv2.putText(frame, lbl, (x, max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20,220,255), 2, cv2.LINE_AA)
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

_RT = _ObjRuntime()

def api_status(ctx: dict, **_) -> Dict[str, object]:
    allowed = None if _RT._allowed is None else sorted(list(_RT._allowed))
    return {"ok": True, "running": _RT.running, "fps": _RT.fps, "overlay": _RT.draw_boxes, "labels_total": len(_RT._labels), "allowed": allowed}

def api_labels(ctx: dict, **_) -> Dict[str, object]:
    return {"ok": True, "labels": _RT._labels}

def _parse_allow(val) -> set[str]:
    if val is None: return set()
    if isinstance(val, str):
        items = [t.strip().lower() for t in val.split(",")]
    elif isinstance(val, Iterable):
        items = [str(t).strip().lower() for t in val]
    else:
        items = []
    return set([t for t in items if t])

def api_allow(ctx: dict, allow=None, **_) -> Dict[str, object]:
    wanted = _parse_allow(allow)
    with _RT.lock:
        _RT._allowed = wanted if len(wanted) > 0 else None
    return {"ok": True, "allowed": None if _RT._allowed is None else sorted(list(_RT._allowed))}

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
