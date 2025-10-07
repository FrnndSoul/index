#!/usr/bin/env python3
from __future__ import annotations
import os, time, importlib.util, pathlib, sys
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

# ---------- Paths ----------
ROOT = pathlib.Path(__file__).resolve().parent
S_DIR = ROOT / "sensor-scripts"
R_DIR = ROOT / "resources"
R_DIR.mkdir(exist_ok=True)

def _p(*parts): return str(pathlib.Path(*parts))

# ---------- Shared Context ----------
CTX = {
    "paths": {
        "resources_dir": _p(R_DIR),
        "dht_db":        _p(R_DIR, "dht_readings.db"),
        "gv_db":         _p(R_DIR, "gv_data.db"),      # <- gas/vibration DB
    },
    "pins": {
        "dht": {"pin": int(os.environ.get("DHT_PIN", 4)),
                "kind": os.environ.get("DHT_KIND", "DHT11")},
        "buzzer": {"pin": int(os.environ.get("BUZZER_PIN", 18))},
        "ultra": {
            "A": {"trig": int(os.environ.get("ULTRA_A_TRIG", 23)),
                  "echo": int(os.environ.get("ULTRA_A_ECHO", 24))},
            "B": {"trig": int(os.environ.get("ULTRA_B_TRIG", 5)),
                  "echo": int(os.environ.get("ULTRA_B_ECHO", 6))}
        },
        "pir": {"pin": int(os.environ.get("PIR_PIN", 22))}
    },
    "state": { "ultra": {"A": None, "B": None} }
}

# ---------- Module aliases ----------
ALIASES = {
    "dht":        "temp-humidity.py",
    "buzzer":     "buzzer.py",
    "ultra":      "ultrasonic.py",
    "sonic":      "ultrasonic.py",
    "pir":        "pir-motion.py",
    "vibration":  "vibration.py",
    "gas":        "gas-sensor.py",
    "sound":      "sound.py",
    "rain":       "rain.py",
    "gps":        "gps.py",
    "acc":        "accelerometer.py",
    "tts":        "text-to-speech.py",
    "stt":        "stt.py",
    "cam":        "camera.py",
}

# ---------- Lazy module loader with cache ----------
_MOD_CACHE: dict[str, object] = {}

def _load(file: str):
    path = S_DIR / file
    if not path.exists():
        raise FileNotFoundError(path)
    key = str(path.resolve())
    if key in _MOD_CACHE:
        return _MOD_CACHE[key]
    name = f"sensor_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    _MOD_CACHE[key] = mod
    return mod

# ---------- App ----------
app = Flask(__name__)
CORS(app)

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "time": int(time.time()*1000), "ctx": CTX})

# ---------- Generic dispatcher: /api/<sensor>/<op> ----------
@app.route("/api/<sensor>/<op>", methods=["GET", "POST", "DELETE"])
def dispatch(sensor: str, op: str):
    fname = ALIASES.get(sensor, f"{sensor}.py")
    m = _load(fname)
    func = getattr(m, f"api_{op}", None)
    if func is None:
        return jsonify({"ok": False, "error": f"{fname} missing api_{op}"}), 404

    # Parse params for each method
    if request.method in ("GET", "DELETE"):
        params = request.args.to_dict()
    else:  # POST (and future JSON methods)
        params = request.get_json(silent=True) or {}

    try:
        return jsonify(func(CTX, **params))
    except TypeError:
        # Back-compat for handlers that expect a single dict
        return jsonify(func(CTX, params))

@app.get("/api/pir/status")
def pir_status():  return dispatch("pir", "status")

@app.get("/api/pir")
def pir_value():   return dispatch("pir", "pir")

@app.get("/api/pir/config")
def pir_cfg_get(): return dispatch("pir", "config")

@app.post("/api/pir/config")
def pir_cfg_set(): return dispatch("pir", "config_set")

@app.get("/api/pir/cam")
def pir_cam():
    m = _load(ALIASES["pir"])
    q = request.args.to_dict()
    resp = m.api_cam(CTX, **q)
    if isinstance(resp, Response):
        return resp
    return jsonify(resp)

@app.get("/api/ultra")
def ultra_a():
    return jsonify(_load(ALIASES["ultra"]).api_read(CTX, id="A"))

@app.get("/api/sonic")
def ultra_b():
    return jsonify(_load(ALIASES["sonic"]).api_read(CTX, id="B"))

@app.get("/api/dht")
def dht_latest():  return dispatch("dht", "latest")

@app.get("/api/dht/read")
def dht_read():    return dispatch("dht", "read")

@app.get("/api/dht/history")
def dht_hist():    return dispatch("dht", "history")

@app.route("/api/buzzer/beep", methods=["GET","POST"])
def buzz_beep():   return dispatch("buzzer", "beep")

@app.get("/api/gas")
def gas_read_short():
    return dispatch("gas", "read")

@app.get("/api/vibrate")
def vibrate_read_short():
    return dispatch("vibration", "read")

@app.get("/api/gas/latest")
def gas_latest_short():
    return dispatch("gas", "latest")

@app.get("/api/vibration/latest")
def vibration_latest_short():
    return dispatch("vibration", "latest")

@app.get("/api/gas/history")
def gas_history_short():
    return dispatch("gas", "history")

@app.get("/api/vibration/history")
def vibration_history_short():
    return dispatch("vibration", "history")
  
@app.get("/api/sound")
def sound_short(): return dispatch("sound", "read")

@app.get("/api/rain")
def rain_short():  return dispatch("rain", "read")

@app.get("/api/gps")
def gps_read_short():
    return dispatch("gps", "read")

@app.get("/api/acc")
def acc_read_short():
    return dispatch("acc", "read")
  
@app.get("/api/tts")
def tts_health():
    return dispatch("tts", "status")

@app.get("/api/tts/download")
def tts_download():
    m = _load(ALIASES["tts"])
    q = request.args.to_dict()
    resp = m.api_download_stream(CTX, **q)
    # If module returned a Flask Response, pass it through as-is.
    if isinstance(resp, Response):
        return resp
    # Otherwise, it returned an error dict — jsonify that.
    return jsonify(resp)

@app.delete("/api/tts/delete")
def tts_delete():
    m = _load(ALIASES["tts"])
    q = request.args.to_dict()
    return jsonify(m.api_delete(CTX, **q))
  
@app.get("/api/cam")
def cam_stream():
    m = _load(ALIASES["cam"])
    q = request.args.to_dict()
    resp = m.api_cam(CTX, **q)
    if isinstance(resp, Response):
        return resp
    return jsonify(resp)

@app.get("/api/cam/health")
def cam_health():
    return dispatch("cam", "health")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
