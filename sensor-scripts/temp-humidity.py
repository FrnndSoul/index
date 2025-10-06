"""
sensor-scripts/temp-humidity.py
Returns Act1-compatible keys:
  { "temperature_c": <float>, "humidity_percent": <float>, "ts": <seconds_int> }
DB still stores ms; we convert on output.
"""
from __future__ import annotations
import time, random, importlib.util, pathlib

def _load_storage(db_path:str):
    base = pathlib.Path(__file__).resolve().parents[1]   # .../index/
    storage_path = base / "resources" / "storage.py"
    spec = importlib.util.spec_from_file_location("storage", storage_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)   # type: ignore[attr-defined]
    store = m.Storage(db_path)
    store.ensure_schema()
    return store

def _sim():
    base_t, base_h = 29.0, 68.0
    return round(base_t + random.uniform(-0.5,0.5),1), round(base_h + random.uniform(-1.5,1.5),1)

def _read_dht(pin:int, kind:str, force_sim:bool=False):
    if force_sim:
        return _sim()
    try:
        import adafruit_dht, board
        cls = adafruit_dht.DHT22 if kind.upper()=="DHT22" else adafruit_dht.DHT11
        if pin != 4:
            raise RuntimeError("Template maps BCM4→board.D4 for CircuitPython path.")
        d = cls(board.D4, use_pulseio=False)
        for _ in range(5):
            try:
                t = float(d.temperature); h = float(d.humidity)
                if t is not None and h is not None:
                    return round(t,1), round(h,1)
            except Exception:
                time.sleep(0.8)
        return None, None
    except Exception:
        pass
    try:
        import Adafruit_DHT
        sensor = Adafruit_DHT.DHT22 if kind.upper()=="DHT22" else Adafruit_DHT.DHT11
        for _ in range(5):
            h, t = Adafruit_DHT.read_retry(sensor, pin, retries=2, delay_seconds=1)
            if t is not None and h is not None:
                return round(float(t),1), round(float(h),1)
        return None, None
    except Exception:
        return _sim()

def api_read(ctx, mode="live"):
    st   = _load_storage(ctx["db_path"])
    pin  = ctx["pins"]["dht"]["pin"]
    kind = ctx["pins"]["dht"]["kind"]
    if mode == "fallback":
        t, h = _read_dht(pin, kind, force_sim=True)
    else:
        t, h = _read_dht(pin, kind, force_sim=False)
        if t is None or h is None:
            t, h = _read_dht(pin, kind, force_sim=True)

    ts_ms = int(time.time()*1000)
    st.insert_dht(ts=ts_ms, temp_c=t, hum=h)
    return {
        "temperature_c": t,
        "humidity_percent": h,
        "ts": ts_ms // 1000  # seconds for Act1 JS
    }

def api_latest(ctx):
    st = _load_storage(ctx["db_path"])
    row = st.fetch_latest()
    if row is None:
        return {"temperature_c": None, "humidity_percent": None, "ts": int(time.time())}
    return {
        "temperature_c": row["temp_c"],
        "humidity_percent": row["hum"],
        "ts": int(row["ts"] // 1000)  # convert ms→s
    }

def api_history(ctx, limit=100, since=None):
    st = _load_storage(ctx["db_path"])
    rows = st.fetch_history(limit=limit, since=since)
    # Return a compact history compatible with your UI if needed later
    return {
        "rows": [
            {"temperature_c": r["temp_c"], "humidity_percent": r["hum"], "ts": int(r["ts"]//1000)}
            for r in rows
        ]
    }
