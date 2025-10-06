"""
NEO-6M GPS reader (UART) – live read, no database.
Optional OLED update when query param display=1 is provided.

Endpoints (via generic dispatcher in index.py):
  GET /api/gps/read?read_s=1.0&debug=1&display=1
"""

from __future__ import annotations
import time, re, importlib.util, pathlib
from typing import Optional, Dict, Any

try:
    import serial  # pyserial
except Exception:
    serial = None  # type: ignore

# ---------- Fixed config (no env) ----------
GPS_PORT       = "/dev/serial0"   # try /dev/ttyAMA0 if needed
GPS_BAUD       = 9600
GPS_TIMEOUT_S  = 0.2
READ_WINDOW_S  = 1.0               # default read duration if not overridden by query

# Fallbacks if the main port fails to open
_FALLBACK_PORTS = [
    GPS_PORT, "/dev/ttyAMA0", "/dev/ttyS0", "/dev/ttyUSB0", "/dev/ttyACM0"
]

# ---------- Helpers ----------
def _open_serial() -> Optional["serial.Serial"]:
    if serial is None:
        return None
    for p in _FALLBACK_PORTS:
        try:
            if not p:
                continue
            ser = serial.Serial(p, GPS_BAUD, timeout=GPS_TIMEOUT_S)
            return ser
        except Exception:
            continue
    return None

def _dm_to_deg(dm: str, hemi: str) -> Optional[float]:
    if not dm or not hemi:
        return None
    try:
        if "." not in dm:
            return None
        head, tail = dm.split(".", 1)
        mins_frac = float(f"{head[-2:]}.{tail}")
        deg = int(head[:-2]) if head[:-2] else 0
        val = deg + mins_frac / 60.0
        if hemi in ("S", "W"):
            val = -val
        return val
    except Exception:
        return None

_knots_to_kmh = 1.852

def _parse_rmc(fields: list[str]) -> dict:
    # $GxRMC,hhmmss.sss,A,llll.ll,a,yyyyy.yy,a,x.x,x.x,ddmmyy,...
    d: Dict[str, Any] = {"has": False}
    try:
        status = fields[2] if len(fields) > 2 else ""
        if status != "A":
            return d
        lat = _dm_to_deg(fields[3], fields[4]) if len(fields) > 4 else None
        lon = _dm_to_deg(fields[5], fields[6]) if len(fields) > 6 else None
        spd_kn = float(fields[7]) if len(fields) > 7 and fields[7] else 0.0
        crs_deg = float(fields[8]) if len(fields) > 8 and fields[8] else None
        d.update({"has": True, "lat": lat, "lon": lon,
                  "speed_kmh": spd_kn * _knots_to_kmh, "course_deg": crs_deg})
        return d
    except Exception:
        return d

def _parse_gga(fields: list[str]) -> dict:
    # $GxGGA,hhmmss.sss,llll.ll,a,yyyyy.yy,a,fix,nsat,hdop,alt,M,...
    d: Dict[str, Any] = {"has": False}
    try:
        fix_q = int(fields[6]) if len(fields) > 6 and fields[6] else 0
        if fix_q <= 0:
            return d
        sats = int(float(fields[7])) if len(fields) > 7 and fields[7] else None
        hdop = float(fields[8]) if len(fields) > 8 and fields[8] else None
        alt  = float(fields[9]) if len(fields) > 9 and fields[9] else None
        lat = _dm_to_deg(fields[2], fields[3]) if len(fields) > 3 else None
        lon = _dm_to_deg(fields[4], fields[5]) if len(fields) > 5 else None
        d.update({"has": True, "lat": lat, "lon": lon,
                  "sats": sats, "hdop": hdop, "alt_m": alt})
        return d
    except Exception:
        return d

_nmea_rx = re.compile(r'^\$(GP|GN)(RMC|GGA),')

def _read_nmea_window(seconds: float) -> dict:
    ser = _open_serial()
    if ser is None:
        return {"err": "serial open failed"}

    deadline = time.time() + max(0.1, float(seconds))
    last_rmc_raw = last_gga_raw = None
    rmc = {}
    gga = {}
    try:
        while time.time() < deadline:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line or not _nmea_rx.match(line):
                continue
            parts = line.split("*")[0].split(",")
            typ = parts[0][3:]
            if typ == "RMC":
                last_rmc_raw = line
                rmc = _parse_rmc(parts)
            elif typ == "GGA":
                last_gga_raw = line
                gga = _parse_gga(parts)
    finally:
        try: ser.close()
        except Exception: pass

    return {"rmc": rmc, "gga": gga, "raw": {"rmc": last_rmc_raw, "gga": last_gga_raw}}

def _merge_fix(rmc: dict, gga: dict) -> dict:
    lat = rmc.get("lat") if rmc.get("lat") is not None else gga.get("lat")
    lon = rmc.get("lon") if rmc.get("lon") is not None else gga.get("lon")
    return {
        "fix": bool(rmc.get("has") or gga.get("has")),
        "lat": lat,
        "lon": lon,
        "speed_kmh": rmc.get("speed_kmh"),
        "course_deg": rmc.get("course_deg"),
        "sats": gga.get("sats"),
        "hdop": gga.get("hdop"),
        "alt_m": gga.get("alt_m"),
    }

# ---------- OLED hook ----------
def _oled_api_frame(ctx: dict, lines: list[str]) -> str:
    """Call sensor-scripts/oled.py: api_frame(ctx, lines=...)."""
    try:
        here = pathlib.Path(__file__).resolve()
        oled_path = here.parent / "oled.py"
        spec = importlib.util.spec_from_file_location("oled_mod", str(oled_path))
        if not spec or not spec.loader:
            return "unavailable"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        res = mod.api_frame(ctx, lines=lines, clear="1")
        return "ok" if res and res.get("ok") else "error"
    except Exception:
        return "unavailable"

# ---------- API ----------
def api_read(CTX: dict, **params):
    """
    Read GPS for a short window and report latest fix data.
    Query:
      read_s: float seconds (default READ_WINDOW_S)
      debug:  "1" to include last raw NMEA lines
      display:"1" to also print to OLED (3 lines: Lon/Lat/Alt)
    """
    try:
        read_s = float(params.get("read_s", READ_WINDOW_S))
    except Exception:
        read_s = READ_WINDOW_S

    ts = int(time.time() * 1000)

    if serial is None:
        return {"ok": False, "error": "pyserial not installed", "ts": ts}

    result = _read_nmea_window(read_s)
    if "err" in result:
        return {"ok": False, "error": result["err"], "ts": ts}

    merged = _merge_fix(result.get("rmc", {}), result.get("gga", {}))
    resp: Dict[str, Any] = {"ok": True, "ts": ts, **merged}

    lng = merged.get("lon")
    lat = merged.get("lat")
    alt = merged.get("alt_m")
    line1 = f"Lon: {lng:.6f}" if isinstance(lng, (int, float)) else "Lon: --"
    line2 = f"Lat: {lat:.6f}" if isinstance(lat, (int, float)) else "Lat: --"
    line3 = f"Alt: {alt:.1f} m" if isinstance(alt, (int, float)) else "Alt: --"
    resp["oled"] = _oled_api_frame(CTX, [line1, line2, line3])

    if params.get("debug") == "1":
        resp["raw"] = result.get("raw")

    return resp
