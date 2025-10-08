#!/usr/bin/env python3
from __future__ import annotations
import os, io, time, json, base64, tempfile, threading, subprocess, socket, datetime, queue, re, pathlib
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache
from flask import send_file
import requests
import numpy as np


try:
    from gtts import gTTS
    from gtts.lang import tts_langs
except Exception:
    gTTS = None
    def tts_langs(): return {}
try:
    from mutagen.mp3 import MP3
except Exception:
    MP3 = None
try:
    import sounddevice as sd
except Exception:
    sd = None
try:
    from vosk import Model, KaldiRecognizer
except Exception:
    Model = None
    KaldiRecognizer = None
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

LT_URL = os.environ.get("LT_URL") or "http://127.0.0.1:5005/translate"
USE_WHISPER = str(os.environ.get("USE_WHISPER", "")).lower() in ("1","true","yes","y")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")

_BUZZER_PIN = 18
_buzzer_kind = None
_buzzer_dev = None

def _ensure_buzzer():
    global _buzzer_kind, _buzzer_dev
    if _buzzer_kind:
        return
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(_BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
        _buzzer_kind = "rpi_gpio"
        _buzzer_dev = GPIO
        return
    except Exception:
        pass
    try:
        from gpiozero import OutputDevice
        _buzzer_dev = OutputDevice(_BUZZER_PIN, active_high=True, initial_value=False)
        _buzzer_kind = "gpiozero_out"
        return
    except Exception:
        _buzzer_kind = None
        _buzzer_dev = None

def _beep(ms: int, duty: float = 0.6):
    _ensure_buzzer()
    ms = max(1, min(3000, int(ms)))
    if _buzzer_kind == "rpi_gpio":
        GPIO = _buzzer_dev
        try:
            GPIO.output(_BUZZER_PIN, GPIO.HIGH)
            time.sleep(ms/1000.0)
        finally:
            try: GPIO.output(_BUZZER_PIN, GPIO.LOW)
            except Exception: pass
    elif _buzzer_kind == "gpiozero_out":
        try:
            _buzzer_dev.on()
            time.sleep(ms/1000.0)
        finally:
            try: _buzzer_dev.off()
            except Exception: pass

def _buzz_api(ms=None, duty=None, pattern=None, gap=120):
    duty = 0.6 if duty is None else float(duty)
    seq = None
    if isinstance(pattern, list):
        seq = [int(float(x)) for x in pattern]
    elif isinstance(pattern, str) and pattern.strip():
        s = pattern.strip()
        if s.startswith("["):
            try:
                seq = [int(float(x)) for x in json.loads(s)]
            except Exception:
                seq = None
        if seq is None:
            try:
                seq = [int(float(x)) for x in s.split(",") if x.strip()]
            except Exception:
                seq = None
    if seq:
        gap = max(0, int(float(gap)))
        for dur in seq:
            _beep(dur, duty)
            if gap:
                time.sleep(gap/1000.0)
        return {"ok": True, "pattern": seq, "gap_ms": gap, "duty": duty, "ts": int(time.time())}
    if ms is not None and str(ms).strip() != "":
        try:
            _beep(int(float(ms)), duty)
            return {"ok": True, "ms": int(float(ms)), "duty": duty, "ts": int(time.time())}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "Provide ms:int>0 or pattern:[int...]"}

_BAD_WORDS = {"fuck","shit","bitch","asshole","puta","gago","ulol","nigga","nigger"}

def _contains_bad(text: str) -> bool:
    t = (text or "").lower()
    return any(b in t for b in _BAD_WORDS)

def _censor(text: str) -> str:
    words = (text or "").split()
    out = []
    for w in words:
        wl = w.lower()
        if any(b in wl for b in _BAD_WORDS):
            out.append("beep")
        else:
            out.append(w)
    return " ".join(out)

def _tokens(text: str) -> list[str]:
    return (text or "").split()

def _weight(tok: str) -> int:
    core = re.sub(r"[^A-Za-z0-9]+", "", tok)
    return max(1, len(core))

def _estimate_ms_for_token(i: int, toks: list[str], total_duration_s: float) -> int:
    if total_duration_s <= 0 or not toks:
        return 150
    weights = [_weight(t) for t in toks]
    wsum = max(1, sum(weights))
    ms = int(total_duration_s * 1000.0 * (weights[i] / wsum))
    return max(100, min(ms, 1200))

_MPV = "mpv"
_MPV_BASE = ["--no-video","--really-quiet","--force-window=no","--idle=no","--term-playing-msg="]
_player_lock = threading.Lock()
_player_proc: Optional[subprocess.Popen] = None
_now: Dict[str, Any] = {"file": None, "lang": None, "paused": False, "ipc": None}

def _mpv_running() -> bool:
    return _player_proc is not None and _player_proc.poll() is None

def _mpv_ipc(cmd_list) -> bool:
    sock = _now.get("ipc")
    if not sock or not os.path.exists(sock):
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock)
        s.sendall((json.dumps({"command": cmd_list}) + "\n").encode("utf-8"))
        s.close()
        return True
    except Exception:
        return False

def _stop_player():
    global _player_proc
    if _mpv_running():
        try:
            _player_proc.terminate()
            _player_proc.wait(timeout=1.0)
        except Exception:
            try:
                _player_proc.kill()
            except Exception:
                pass
    _player_proc = None
    sock = _now.get("ipc")
    if sock and os.path.exists(sock):
        try: os.remove(sock)
        except OSError: pass
    _now.update({"file": None, "lang": None, "paused": False, "ipc": None})

def _wait_sock(sock_path: str, timeout_s=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if os.path.exists(sock_path): return True
        time.sleep(0.02)
    return os.path.exists(sock_path)

def _play_with_mpv(filepath: str, speed: float, volume: float, pitch: float, start_paused: bool = False):
    global _player_proc
    sock = f"/tmp/tts-mpv-{os.getpid()}-{int(time.time()*1000)}.sock"
    vol_pct = max(0, min(100, int(round(volume * 100))))
    args = [
        _MPV, *_MPV_BASE,
        "--audio-pitch-correction=yes",
        f"--speed={speed}",
        f"--volume={vol_pct}",
        f"--input-ipc-server={sock}",
    ]
    if start_paused:
        args.append("--pause=yes")
    if abs(pitch - 1.0) > 1e-3:
        af = f"lavfi=[asetrate=48000*{pitch},aresample=48000,atempo={1.0/pitch}]"
        args.append(f"--af={af}")
    args.append(filepath)
    _player_proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if _wait_sock(sock):
        _now['ipc'] = sock
    _player_proc.wait()
    _player_proc = None
    if os.path.exists(sock):
        try: os.remove(sock)
        except OSError: pass
    _now.update({"file": None, "lang": None, "paused": False, "ipc": None})

def _canon_lang(code: str) -> str:
    MODEL_ALIASES = {"fil":"tl", "tagalog":"tl", "ph":"tl", "en-us":"en", "en_gb":"en", "en-gb":"en"}
    c = (code or "").strip()
    if not c: return "en"
    c2 = c.split("-")[0].lower()
    return MODEL_ALIASES.get(c.lower(), MODEL_ALIASES.get(c2, c2))

def _mp3_duration_bytes(data: bytes) -> float:
    if MP3 is None:
        return 0.0
    with tempfile.NamedTemporaryFile(prefix="dur_", suffix=".mp3", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        return float(MP3(tmp).info.length)
    except Exception:
        return 0.0
    finally:
        try: os.remove(tmp)
        except OSError: pass

def _mp3_duration_file(path: Path) -> float:
    if MP3 is None:
        return 0.0
    try:
        return float(MP3(str(path)).info.length)
    except Exception:
        return 0.0

def _history_dir(CTX: dict) -> Path:
    root = Path(CTX["paths"]["resources_dir"])
    d = root / "history"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _f(val, lo, hi, default):
    try:
        return max(lo, min(hi, float(val)))
    except Exception:
        return default

def _buzz_if_new_bad(text: str, seen: set) -> None:
    for w in (text or "").split():
        wl = w.lower()
        if any(b in wl for b in _BAD_WORDS) and wl not in seen:
            _buzz_api(ms=150)
            seen.add(wl)

def _find_bad_indices(text: str) -> list[int]:
    bad_idx = []
    for i, w in enumerate((text or "").split()):
        wl = w.lower()
        if any(b in wl for b in _BAD_WORDS):
            bad_idx.append(i)
    return bad_idx

def _mpv_get_property(sock_path: str, prop: str):
    if not sock_path or not os.path.exists(sock_path):
        return None
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(0.2)
        s.connect(sock_path)
        payload = json.dumps({"command": ["get_property", prop]}) + "\n"
        s.sendall(payload.encode("utf-8"))
        data = s.recv(4096)
        s.close()
        if not data:
            return None
        return json.loads(data.decode("utf-8", errors="ignore"))
    except Exception:
        return None

def _run_beep_sync(sock_path: str, mp3_duration_s: float, toks: list[str], bad_indices: list[int], rate: float, lead_s: float = 0.12):
    if not toks or not bad_indices:
        return
    if rate <= 0:
        rate = 1.0
    duration = (mp3_duration_s or 0.0) / rate
    if duration <= 0:
        for _ in range(40):
            resp = _mpv_get_property(sock_path, "duration")
            if resp and resp.get("error") == "success" and isinstance(resp.get("data"), (int,float)):
                duration = float(resp["data"])
                break
            time.sleep(0.05)
        if duration <= 0:
            return
    def _w(tok: str) -> int:
        return max(1, len(re.sub(r"[^A-Za-z0-9]+","", tok)))
    weights = [_w(t) for t in toks]
    total_w = max(1, sum(weights))
    starts = []
    acc = 0.0
    for w in weights:
        starts.append((acc / total_w) * duration)
        acc += w
    fired = set()
    last_pos = -1.0
    while True:
        if not os.path.exists(sock_path):
            break
        resp = _mpv_get_property(sock_path, "time-pos")
        if not resp or resp.get("error") != "success":
            time.sleep(0.02)
            continue
        pos = resp.get("data")
        if pos is None:
            time.sleep(0.02)
            continue
        for idx in bad_indices:
            if idx in fired:
                continue
            t_start = starts[idx] - max(0.0, lead_s)
            if pos >= t_start and (last_pos < t_start or last_pos < 0):
                ms = _estimate_ms_for_token(idx, toks, duration)
                _buzz_api(ms=ms)
                fired.add(idx)
        last_pos = pos
        if len(fired) >= len(bad_indices):
            break
        if pos >= duration + 0.25:
            break
        time.sleep(0.02)

def _lt_norm(code: str) -> str:
    c = (code or "").strip().lower()
    if not c:
        return ""
    c = c.replace("_", "-")
    if c.startswith("en"):
        return "en"
    if c in ("fil", "fil-ph", "tl", "tl-ph", "tagalog"):
        return "tl"
    # map common variants to LT codes
    aliases = {
        "zh-hans": "zh", "zh-cn": "zh", "zh-hant": "zh",
        "pt-br": "pt", "pt-pt": "pt",
        "he": "he", "iw": "he",
    }
    base = c.split("-")[0]
    return aliases.get(c, aliases.get(base, base))

def _translate_text(text: str, src: str | None, dst: str | None) -> str:
    if not text or not LT_URL:
        return text
    s = _lt_norm(src or "")
    t = _lt_norm(dst or "")
    if not t or s == t:
        return text

    def _call_lt(q: str, source: str | None, target: str) -> str:
        payload = {"q": q, "source": source or "auto", "target": target, "format": "text"}
        try:
            r = requests.post(LT_URL, json=payload, timeout=12)
            r.raise_for_status()
            j = r.json()
            return j.get("translatedText") or j.get("translated_text") or q
        except Exception:
            return q

    if s and s != "en" and t != "en":
        mid = _call_lt(text, s, "en")
        if not mid or mid == text:
            mid = _call_lt(text, None, "en")
        out = _call_lt(mid, "en", t)
        return out

    return _call_lt(text, s or None, t)

def api_out_langs(CTX: dict, **params):
    base = LT_URL.rstrip("/").rsplit("/", 1)[0] + "/languages"
    try:
        r = requests.get(base, timeout=6)
        r.raise_for_status()
        items = r.json()
        out = []
        for it in items:
            code = (it.get("code") or "").lower()
            name = it.get("name") or code
            if code in ("tl","tagalog"): code, name = "fil", "Filipino"
            out.append({"code": code, "name": name})
        pin = ["fil","en"]
        for p in reversed(pin):
            i = next((k for k, x in enumerate(out) if x["code"] == p), None)
            if i is not None:
                out.insert(0, out.pop(i))
        return out
    except Exception:
        return [{"code":"fil","name":"Filipino"},{"code":"en","name":"English"}]

def api_langs(CTX: dict, **params):
    include_providers = str(params.get("providers", "")).lower() in ("1","true","yes")
    langs = tts_langs()
    items = [{"code": c, "name": n} for c, n in langs.items()]
    ph_locales = [
        {"code": "en-PH",  "name": "English (Philippines)",           "provider": "azure"},
        {"code": "fil-PH", "name": "Filipino (Philippines)",          "provider": "azure",
         "voices": ["fil-PH-AngeloNeural", "fil-PH-BlessicaNeural"]},
        {"code": "ceb-PH", "name": "Cebuano (Bisaya)",                 "provider": "espeak-ng"},
        {"code": "ilo-PH", "name": "Ilocano",                          "provider": "espeak-ng"},
        {"code": "pam-PH", "name": "Kapampangan",                      "provider": "espeak-ng"},
        {"code": "hil-PH", "name": "Hiligaynon",                       "provider": "espeak-ng"},
        {"code": "war-PH", "name": "Waray",                            "provider": "espeak-ng"},
    ]
    if not include_providers:
        ph_locales = [{"code": x["code"], "name": x["name"]} for x in ph_locales]
    have = {it["code"] for it in items}
    for x in ph_locales:
        if x["code"] not in have:
            items.append(x)
    items.sort(key=lambda x: x["name"].lower())
    PIN_TOP = ["en-PH","fil-PH","ceb-PH","ilo-PH","pam-PH","hil-PH","war-PH","en","tl"]
    for pref in reversed(PIN_TOP):
        i = next((k for k, it in enumerate(items) if it["code"] == pref), None)
        if i is None and pref in ("en","tl"):
            i = next((k for k, it in enumerate(items) if it["code"].startswith(pref + "-")), None)
        if i is not None:
            items.insert(0, items.pop(i))
    items.insert(0, {"code": "auto", "name": "Auto detect"})
    return items

def api_download_stream(CTX: dict, **params):
    fname = (params.get("file") or "").strip()
    if not fname or "/" in fname or "\\" in fname:
        return {"ok": False, "error": "Bad file"}
    p = _history_dir(CTX) / fname
    if not p.is_file():
        return {"ok": False, "error": "Not found"}
    return send_file(str(p), mimetype="audio/mpeg", as_attachment=True, download_name=fname)

def api_say_play(CTX: dict, **params):
    if gTTS is None:
        return {"ok": False, "error": "gTTS not installed"}
    raw_text = (params.get("text") or "").strip()
    if not raw_text:
        return {"ok": False, "error": "Missing text"}
    lang  = (params.get("lang")  or "en").strip()
    rate  = _f(params.get("rate",  "1.0"), 0.5, 2.0, 1.0)
    pitch = _f(params.get("pitch", "1.0"), 0.5, 2.0, 1.0)
    volume= _f(params.get("volume","1.0"), 0.0, 1.0, 1.0)
    safe_text = _censor(raw_text)
    try:
        if lang == "auto":
            try:
                from langdetect import detect as langdetect_detect
                lang = langdetect_detect(safe_text)
            except Exception:
                lang = "en"
        lang = _canon_lang(lang)
        tts = gTTS(text=safe_text, lang=lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        data = buf.getvalue()
        dur  = _mp3_duration_bytes(data)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = f"{ts}_{lang}.mp3"
        fpath = _history_dir(CTX) / fname
        with open(fpath, "wb") as f:
            f.write(data)
        toks = _tokens(raw_text)
        bad_indices = _find_bad_indices(raw_text)
        with _player_lock:
            _stop_player()
            _now.update({"file": fname, "lang": lang, "paused": False, "ipc": None})
            t_play = threading.Thread(target=_play_with_mpv, args=(str(fpath), rate, volume, pitch), daemon=True)
            t_play.start()
        def _arm_and_unpause():
            t0 = time.time()
            while time.time() - t0 < 3.0:
                sock = _now.get("ipc")
                if sock and os.path.exists(sock):
                    break
                time.sleep(0.01)
            sock = _now.get("ipc")
            if sock and os.path.exists(sock) and bad_indices:
                threading.Thread(target=_run_beep_sync, args=(sock, float(dur or 0.0), toks, bad_indices, float(rate or 1.0)), daemon=True).start()
            _mpv_ipc(["set_property","pause",False])
            _now["paused"] = False
        threading.Thread(target=_arm_and_unpause, daemon=True).start()
        def _start_sync_when_ready():
            t0 = time.time()
            while time.time() - t0 < 3.0:
                sock = _now.get("ipc")
                if sock and os.path.exists(sock):
                    break
                time.sleep(0.05)
            sock = _now.get("ipc")
            if sock and os.path.exists(sock) and bad_indices:
                threading.Thread(target=_run_beep_sync, args=(sock, float(dur or 0.0), toks, bad_indices), daemon=True).start()
        threading.Thread(target=_start_sync_when_ready, daemon=True).start()
        return {"ok": True, "status": "playing", "file": fname, "lang": lang, "duration": float(dur or 0.0)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def api_play_file(CTX: dict, **params):
    fname = (params.get("file") or "").strip()
    if not fname or "/" in fname or "\\" in fname:
        return {"ok": False, "error": "Bad file"}
    fpath = _history_dir(CTX) / fname
    if not fpath.is_file():
        return {"ok": False, "error": "Not found"}
    rate  = _f(params.get("rate",  "1.0"), 0.5, 2.0, 1.0)
    pitch = _f(params.get("pitch", "1.0"), 0.5, 2.0, 1.0)
    volume= _f(params.get("volume","1.0"), 0.0, 1.0, 1.0)
    dur = _mp3_duration_file(fpath)
    lang = fname.rsplit("_",1)[-1].replace(".mp3","") if "_" in fname else None
    raw_text = (params.get("text") or "").strip()
    toks = _tokens(raw_text) if raw_text else []
    bad_indices = _find_bad_indices(raw_text) if raw_text else []
    with _player_lock:
        _stop_player()
        _now.update({"file": fname, "lang": lang, "paused": True, "ipc": None})
        threading.Thread(target=_play_with_mpv, args=(str(fpath), rate, volume, pitch, True), daemon=True).start()
    def _arm_and_unpause():
        if not raw_text or not bad_indices:
            _mpv_ipc(["set_property","pause",False]); _now["paused"]=False; return
        t0 = time.time()
        while time.time() - t0 < 3.0:
            sock = _now.get("ipc")
            if sock and os.path.exists(sock):
                break
            time.sleep(0.01)
        sock = _now.get("ipc")
        if sock and os.path.exists(sock):
            threading.Thread(target=_run_beep_sync, args=(sock, float(dur or 0.0), toks, bad_indices, float(rate or 1.0)), daemon=True).start()
        _mpv_ipc(["set_property","pause",False]); _now["paused"]=False
    threading.Thread(target=_arm_and_unpause, daemon=True).start()
    def _start_sync_when_ready():
        if not toks or not bad_indices:
            return
        t0 = time.time()
        while time.time() - t0 < 3.0:
            sock = _now.get("ipc")
            if sock and os.path.exists(sock):
                break
            time.sleep(0.05)
        sock = _now.get("ipc")
        if sock and os.path.exists(sock):
            threading.Thread(target=_run_beep_sync, args=(sock, float(dur or 0.0), toks, bad_indices), daemon=True).start()
    threading.Thread(target=_start_sync_when_ready, daemon=True).start()
    return {"ok": True, "status": "playing", "file": fname, "lang": lang, "duration": float(dur or 0.0)}

def api_pause(CTX: dict, **params):
    with _player_lock:
        if not _mpv_running(): return {"ok": False, "error":"not playing"}
        ok = _mpv_ipc(["set_property","pause",True])
        if not ok: return {"ok": False, "error":"ipc failed"}
        _now["paused"] = True
    return {"ok": True, "status": "paused"}

def api_resume(CTX: dict, **params):
    with _player_lock:
        if not _mpv_running(): return {"ok": False, "error":"not playing"}
        ok = _mpv_ipc(["set_property","pause",False])
        if not ok: return {"ok": False, "error":"ipc failed"}
        _now["paused"] = False
    return {"ok": True, "status": "playing"}

def api_say_stop(CTX: dict, **params):
    with _player_lock:
        _stop_player()
    return {"ok": True, "status": "stopped"}

def api_status(CTX: dict, **params):
    return {"playing": _mpv_running(), "paused": bool(_now["paused"]), "file": _now["file"], "lang": _now["lang"]}

def api_history(CTX: dict, **params):
    rows = []
    for p in _history_dir(CTX).glob("*.mp3"):
        st = p.stat()
        rows.append({"file": p.name, "size": st.st_size, "mtime": int(st.st_mtime), "duration": _mp3_duration_file(p)})
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows

def api_download(CTX: dict, **params):
    fname = (params.get("file") or "").strip()
    if not fname or "/" in fname or "\\" in fname:
        return {"ok": False, "error": "Bad file"}
    p = _history_dir(CTX) / fname
    if not p.is_file():
        return {"ok": False, "error": "Not found"}
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return {"ok": True, "file": fname, "content_type": "audio/mpeg", "data_b64": b64}

def api_delete(CTX: dict, **params):
    fname = (params.get("file") or "").strip()
    if not fname or "/" in fname or "\\" in fname:
        return {"ok": False, "error": "Bad file"}
    p = _history_dir(CTX) / fname
    if not p.is_file():
        return {"ok": False, "error": "Not found"}
    with _player_lock:
        if _now["file"] == fname:
            _stop_player()
        try:
            p.unlink()
        except Exception as e:
            return {"ok": False, "error": str(e)}
        if _now["file"] == fname:
            _now.update({"file": None, "lang": None, "paused": False, "ipc": None})
    return {"ok": True}

_MODEL_DIRS = {
    "en": os.path.expanduser("~/vosk/vosk-model-small-en-us-0.15"),
    "tl": os.path.expanduser("~/vosk/vosk-model-tl-ph-generic-0.6"),
}
_DEFAULT_STT_LANG = "en"

@lru_cache(maxsize=4)
def _load_model(lang_code: str):
    if Model is None:
        raise RuntimeError("vosk not installed")
    key = _canon_lang(lang_code)
    path = _MODEL_DIRS.get(key)
    if not path or not os.path.isdir(path):
        raise FileNotFoundError(f"Vosk model for '{key}' not found at {path!r}")
    return Model(path)

def _make_recognizer(lang_code: str, samplerate: int = 16000):
    m = _load_model(lang_code)
    rec = KaldiRecognizer(m, samplerate)
    rec.SetWords(True)
    return rec

_recording = {"thread": None, "running": False, "lang": "auto", "text": "", "used_lang": None, "pcm": bytearray()}

def _score_text(s: str) -> int:
    return len([t for t in s.split() if t])

def _choose_lang_by_probe(buf: bytes, samplerate: int = 16000) -> str:
    best_lang, best_score = _DEFAULT_STT_LANG, -1
    for cand in ("tl", "en"):
        rec = _make_recognizer(cand, samplerate)
        rec.AcceptWaveform(buf)
        got = json.loads(rec.Result()).get("text","").strip()
        score = _score_text(got)
        if score > best_score:
            best_score, best_lang = score, cand
    return best_lang

def _wh_lang_norm(code: str) -> str | None:
    c = (code or "").strip().lower()
    if not c or c == "auto":
        return None
    c = c.replace("_","-")
    m = {
        "fil":"tl", "tagalog":"tl", "tl":"tl",
        "en":"en", "en-ph":"en", "en-us":"en", "en-gb":"en",
        "zh":"zh", "zh-cn":"zh", "zh-hans":"zh", "zh-hant":"zh", "zh-tw":"zh",
        "pt-br":"pt", "pt-pt":"pt",
        "he":"he", "iw":"he",
    }
    base = c.split("-")[0]
    return m.get(c, m.get(base, base))

_wh_model = None
def _get_whisper():
    global _wh_model
    if _wh_model is None:
        if WhisperModel is None:
            raise RuntimeError("whisper not installed")
        _wh_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    return _wh_model

def _record_loop_whisper(selected_lang: str):
    if sd is None:
        _recording.update({"running": False})
        return
    samplerate, blocksize = 16000, 4000
    q: "queue.Queue[bytes]" = queue.Queue()
    def cb(indata, frames, time_info, status):
        if _recording["running"]:
            q.put(bytes(indata))
    _recording.update({"pcm": bytearray(), "text": "", "used_lang": None, "running": True})
    with sd.RawInputStream(samplerate=samplerate, blocksize=blocksize, dtype='int16', channels=1, callback=cb):
        while _recording["running"]:
            try:
                chunk = q.get(timeout=0.5)
            except queue.Empty:
                continue
            _recording["pcm"].extend(chunk)

def _record_loop_vosk(selected_lang: str):
    if sd is None or KaldiRecognizer is None:
        _recording.update({"running": False})
        return
    samplerate, blocksize = 16000, 4000
    q: "queue.Queue[bytes]" = queue.Queue()
    probe = bytearray()
    have_rec = False
    lang_in_use = selected_lang
    rec = None
    seen_bad: set[str] = set()
    def callback(indata, frames, time_info, status):
        if _recording["running"]:
            q.put(bytes(indata))
    _recording.update({"text": "", "used_lang": None, "running": True})
    with sd.RawInputStream(samplerate=samplerate, blocksize=blocksize, dtype='int16', channels=1, callback=callback):
        while _recording["running"]:
            try:
                chunk = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not have_rec and lang_in_use.lower() == "auto":
                probe.extend(chunk)
                if len(probe) >= samplerate * 2 * 3:
                    lang_in_use = _choose_lang_by_probe(bytes(probe), samplerate)
                    rec = _make_recognizer(lang_in_use, samplerate)
                    have_rec = True
                continue
            if not have_rec:
                rec = _make_recognizer(lang_in_use, samplerate)
                have_rec = True
            if rec.AcceptWaveform(chunk):
                res = json.loads(rec.Result())
                txt = (res.get("text") or "").strip()
                if txt:
                    _buzz_if_new_bad(txt, seen_bad)
                    _recording["text"] += (" " + txt if _recording["text"] else txt)
            else:
                pres = json.loads(rec.PartialResult())
                ptxt = (pres.get("partial") or "").strip()
                if ptxt:
                    _buzz_if_new_bad(ptxt, seen_bad)
        if have_rec and rec is not None:
            final = json.loads(rec.FinalResult())
            ftxt = (final.get("text") or "").strip()
            if ftxt:
                _buzz_if_new_bad(ftxt, seen_bad)
                _recording["text"] += (" " + ftxt if _recording["text"] else ftxt)
    _recording["used_lang"] = (lang_in_use if lang_in_use != "auto" else _DEFAULT_STT_LANG)

def api_record_start(CTX: dict, **params):
    lang = (params.get("input_lang") or params.get("lang") or "auto").strip().lower()
    if _recording["running"]:
        return {"ok": False, "error": "already recording"}
    _recording["lang"] = lang or "auto"
    if not USE_WHISPER and lang.lower() != "auto":
        try: _ = _load_model(lang)
        except Exception as e:
            return {"ok": False, "error": f"STT model not available for '{lang}': {e}"}
    t = threading.Thread(
        target=_record_loop_whisper if USE_WHISPER else _record_loop_vosk,
        args=(lang,),
        daemon=True
    )
    _recording["thread"] = t
    _recording["running"] = True
    t.start()
    return {"ok": True, "status": "recording"}

def _pcm16_to_float32(pcm_bytes: bytes) -> "np.ndarray":
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    a = np.frombuffer(pcm_bytes, dtype=np.int16)
    return (a.astype(np.float32) / 32768.0)

def _whisper_transcribe(pcm_bytes: bytes, lang_pref: str | None) -> tuple[str, str]:
    audio = _pcm16_to_float32(pcm_bytes)
    if audio.size == 0:
        return "", (lang_pref or "auto") or ""
    model = _get_whisper()
    forced = _wh_lang_norm(lang_pref or "")
    def run(lang_code):
        segments, info = model.transcribe(
            audio,
            language=lang_code,
            task="transcribe",
            vad_filter=True,
            beam_size=1,
            without_timestamps=True
        )
        text = "".join(seg.text for seg in segments).strip()
        used_lang = (info.language or (lang_code or "auto") or "").lower()
        return text, used_lang
    text, used = run(forced)
    if not text and forced is not None:
        text, used = run(None)
    used = _lt_norm(used or "")
    return text, used or "auto"

def api_record_stop(CTX: dict, **params):
    if not _recording["running"]:
        pass
    _recording["running"] = False
    t = _recording.get("thread")
    if t: t.join(timeout=6.0)
    if USE_WHISPER:
        pcm = bytes(_recording.get("pcm") or b"")
        pref = _recording.get("lang") or "auto"
        text, used = _whisper_transcribe(pcm, pref)
        _recording["text"] = text
        _recording["used_lang"] = used or (pref if pref != "auto" else "en")
    else:
        if not _recording.get("used_lang"):
            _recording["used_lang"] = _DEFAULT_STT_LANG
    text = (_recording.get("text") or "").strip()
    used = _recording.get("used_lang") or "en"
    if _contains_bad(text):
        _buzz_api(ms=400)
    dst = (params.get("translate_to") or "").strip()
    translated = _translate_text(text, used, dst) if dst else text
    if translated is None:
        translated = ""
    return {"ok": True, "text": text, "translated": translated, "lang": used, "src": _lt_norm(used), "dst": _lt_norm(dst)}
