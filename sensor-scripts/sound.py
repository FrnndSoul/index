"""
Sound sensor (KY-037 analog) on ADS1115.
Returns normalized loudness 0..1 computed from a short burst peak-to-peak envelope.
No database storage; on-demand read only.

Query params (optional):
  mode=voltage  -> also include instantaneous min/max volts of the burst

Based on your working approach: short burst sampling, peak-to-peak, normalize by a reference. 
"""

from __future__ import annotations
import os, time
from typing import Optional, Tuple

# ===== Config (tune if needed) =====
I2C_ADDR      = int(os.getenv("I2C_ADDR",  "0x48"), 16)
ADS_GAIN      = int(os.getenv("ADS_GAIN",  "1"))        # ±4.096 V
DATA_RATE     = int(os.getenv("ADS_DATA_RATE", "475"))  # 8..860 SPS
SOUND_CH_IDX  = int(os.getenv("SOUND_CH", "1"))         # 0..3 => A0..A3
VREF          = float(os.getenv("VREF", "3.3"))         # normalize target
BURST_SAMPLES = int(os.getenv("SOUND_BURST_N", "120"))  # ~250 ms @ ~475 SPS

# Lazy-inited ADS objects
_ads = {"ads": None, "AnalogIn": None, "last_err": None}

def _ensure_ads() -> Tuple[Optional[object], Optional[object]]:
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

def _burst_envelope_pp() -> Optional[Tuple[float, float, float]]:
    """Return (vpp, vmin, vmax) over a short burst; None on failure."""
    ads, AnalogIn = _ensure_ads()
    if ads is None:
        return None
    try:
        import adafruit_ads1x15.ads1115 as ADS
        ch = [ADS.P0, ADS.P1, ADS.P2, ADS.P3][SOUND_CH_IDX]
        ain = AnalogIn(ads, ch)
        # warmup read
        _ = ain.value
        vmin, vmax = 10.0, -10.0
        n = BURST_SAMPLES
        for _ in range(n):
            v = ain.voltage
            if v < vmin: vmin = v
            if v > vmax: vmax = v
        vpp = max(0.0, vmax - vmin)
        return vpp, vmin, vmax
    except Exception as e:
        _ads["last_err"] = str(e)
        return None

def _normalize_0_1(volts_pp: float) -> float:
    # Reference envelope for “loud”. Adjust 0.6 V as needed for your sensor gain.
    ref_pp = float(os.getenv("SOUND_REF_PPV", "0.6"))
    if ref_pp <= 0: ref_pp = 0.6
    x = volts_pp / ref_pp
    if x < 0.0: x = 0.0
    if x > 1.0: x = 1.0
    return float(x)

def api_read(CTX: dict, **params):
    ts = int(time.time() * 1000)
    burst = _burst_envelope_pp()
    if burst is None:
        return {"ok": False, "error": "sensor read failed", "detail": _ads.get("last_err")}
    vpp, vmin, vmax = burst
    now_01 = _normalize_0_1(vpp)

    resp = {"ok": True, "now": round(now_01, 4), "vpp": round(vpp, 4), "ts": ts}
    if params.get("mode") == "voltage":
        resp.update({"vmin": round(vmin, 4), "vmax": round(vmax, 4), "vref": VREF})
    return resp
