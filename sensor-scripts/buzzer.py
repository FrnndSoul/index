# /home/pi/Documents/index/sensor-scripts/buzzer.py
from __future__ import annotations
import time, json
from gpiozero import PWMOutputDevice

_buzzer = None

def _dev(ctx):
    global _buzzer
    if _buzzer is None:
        pin = int(ctx.get("pins", {}).get("buzzer", {}).get("pin", 18))
        _buzzer = PWMOutputDevice(pin, frequency=2000, initial_value=0.0)
    return _buzzer

def _beep(ms: int, duty: float = 0.6):
    dev = _dev({"pins": {"buzzer": {}}})
    duty = max(0.0, min(1.0, float(duty)))
    ms = max(1, min(3000, int(ms)))
    dev.value = duty
    time.sleep(ms / 1000.0)
    dev.value = 0.0

def api_beep(ctx, ms=None, duty=None, pattern=None, gap=120):
    """
    Accepts:
      - ?ms=400[&duty=0.6]
      - ?pattern=200,200,600  (comma-separated)
      - JSON: {"pattern":[200,200,600]} or {"ms":400,"duty":0.7}
    """
    # normalize duty
    duty = 0.6 if duty is None else float(duty)

    # normalize pattern: list[int] if provided
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

    # single ms
    if ms is not None and str(ms).strip() != "":
        try:
            _beep(int(float(ms)), duty)
            return {"ok": True, "ms": int(float(ms)), "duty": duty, "ts": int(time.time())}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return {"ok": False, "error": "Provide ms:int>0 or pattern:[int...]"}
