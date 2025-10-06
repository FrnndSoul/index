"""
Rain sensor (YL-83 analog) on ADS1115.
Returns normalized rain intensity 0..1 from a single voltage sample (clamped by VREF).
No database storage; on-demand read only.

Query params (optional):
  mode=voltage  -> include raw voltage in the response
"""

from __future__ import annotations
import os, time
from typing import Optional, Tuple

# ===== Config =====
I2C_ADDR     = int(os.getenv("I2C_ADDR",  "0x48"), 16)
ADS_GAIN     = int(os.getenv("ADS_GAIN",  "1"))        # ±4.096 V
DATA_RATE    = int(os.getenv("ADS_DATA_RATE", "128"))  # slower OK for rain
RAIN_CH_IDX  = int(os.getenv("RAIN_CH", "2"))          # 0..3 => A0..A3
VREF         = float(os.getenv("VREF", "3.3"))

_ads = {"ads": None, "AnalogIn": None, "last_err": None}

def _ensure_ads():
    if _ads["ads"] is not None:
        return _ads["ads"], _ads["AnalogIn"]
    try:
        import board, busio
        import adafruit_ads1x15.ads1115 as ADS
        from adafruit_ads1x15.analog_in import AnalogIn
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c, address=I2C_ADDR)
        ads.gain = ADS_GAIN
        ads.data_rate = DATA_RATE
        _ads.update({"ads": ads, "AnalogIn": AnalogIn, "last_err": None})
        return ads, AnalogIn
    except Exception as e:
        _ads.update({"ads": None, "AnalogIn": None, "last_err": str(e)})
        return None, None

def _read_voltage() -> Optional[float]:
    ads, AnalogIn = _ensure_ads()
    if ads is None:
        return None
    try:
        import adafruit_ads1x15.ads1115 as ADS
        ch = [ADS.P0, ADS.P1, ADS.P2, ADS.P3][RAIN_CH_IDX]
        ain = AnalogIn(ads, ch)
        _ = ain.value  # warmup
        return float(ain.voltage)
    except Exception as e:
        _ads["last_err"] = str(e)
        return None

def _normalize_0_1(volts: float) -> float:
    x = volts / max(0.001, VREF)
    if x < 0.0: x = 0.0
    if x > 1.0: x = 1.0
    return float(x)

def api_read(CTX: dict, **params):
    ts = int(time.time() * 1000)
    v = _read_voltage()
    if v is None:
        return {"ok": False, "error": "sensor read failed", "detail": _ads.get("last_err")}
    intensity = _normalize_0_1(v)
    resp = {"ok": True, "intensity": round(intensity, 4), "ts": ts}
    if params.get("mode") == "voltage":
        resp.update({"volts": round(v, 4), "vref": VREF})
    return resp
