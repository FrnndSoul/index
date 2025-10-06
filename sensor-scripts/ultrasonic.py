# /home/pi/Documents/index/sensor-scripts/ultrasonic.py
from __future__ import annotations
import time, random, importlib.util, pathlib

SPEED_M_PER_S = 343.2  # ~20 °C

def _read_once(trig:int, echo:int, timeout_s:float=0.04):
    try:
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(trig, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(echo, GPIO.IN)
        time.sleep(0.0002)
        GPIO.output(trig, True); time.sleep(0.00001); GPIO.output(trig, False)

        t0 = time.time()
        while GPIO.input(echo) == 0:
            if time.time() - t0 > timeout_s: return None
        start = time.time()
        while GPIO.input(echo) == 1:
            if time.time() - start > timeout_s: return None
        end = time.time()
        dt = end - start
        return round((SPEED_M_PER_S * dt * 100.0) / 2.0, 1)
    except Exception:
        return None

def _sim(): return round(random.uniform(8.0, 200.0), 1)

def _load_oled_module():
    base = pathlib.Path(__file__).resolve().parent.parent
    p = base / "sensor-scripts" / "oled.py"
    if not p.exists(): return None
    spec = importlib.util.spec_from_file_location("oled_module", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def _oled_draw(ctx, a_val, b_val):
    mod = _load_oled_module()
    if not mod or not hasattr(mod, "api_frame"): return
    lines = [
        "ULTRA DISTANCES",
        f"A: {('--' if a_val is None else f'{a_val:.1f}')} cm",
        f"B: {('--' if b_val is None else f'{b_val:.1f}')} cm",
    ]
    try: mod.api_frame(ctx, lines=lines, clear="1")
    except Exception: pass

def api_read(ctx, id:str="A", trig:str|None=None, echo:str|None=None, attempts:int=3):
    id = (id or "A").upper()

    # shared state in CTX
    state = ctx.setdefault("state", {})
    ultra = state.setdefault("ultra", {"A": None, "B": None})

    # pins
    pins_cfg = (ctx.get("pins", {}).get("ultra", {})) if isinstance(ctx, dict) else {}
    defaults = pins_cfg.get(id, {})
    trig_pin = int(trig) if trig is not None else int(defaults.get("trig", 23 if id=="A" else 5))
    echo_pin = int(echo) if echo is not None else int(defaults.get("echo", 24 if id=="A" else 6))

    # read once (with a few retries), fallback to sim
    cm = None
    for _ in range(max(1, int(attempts))):
        cm = _read_once(trig_pin, echo_pin)
        if cm is not None: break
    simulated = cm is None
    if simulated: cm = _sim()

    # update state
    ultra[id] = cm

    # build deterministic frame using *this* value + the other side from state
    a_val = cm if id == "A" else ultra.get("A")
    b_val = cm if id == "B" else ultra.get("B")
    _oled_draw(ctx, a_val, b_val)

    resp = {"id": id, "distance_cm": cm, "ts": int(time.time())}
    if simulated: resp["simulated"] = True
    return resp
