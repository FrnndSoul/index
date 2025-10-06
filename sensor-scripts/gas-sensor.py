"""
Gas/Smoke (MQ-2 via ADS1115) with hysteresis + email alerts.

Env (optional):
  USE_ADS1115=1
  ADS_ADDR_HEX=0x48           # 0x48..0x4B
  GAS_CHANNEL=0               # 0..3 => A0..A3
  VREF_VOLTS=3.3

  # Gas thresholds + hysteresis (ppm)
  GAS_WARN_ON=300  GAS_WARN_OFF=250
  GAS_ALRT_ON=400  GAS_ALRT_OFF=350

  # Email cooldowns (seconds)
  EMAIL_COOLDOWN_S_GAS=300
"""

from __future__ import annotations
import os, time, sqlite3, typing as t
import pathlib, importlib.util

# ----- ADS1115 config -----
USE_ADS1115  = os.getenv("USE_ADS1115", "1") in ("1","true","True")
ADS_ADDR_HEX = int(os.getenv("ADS_ADDR_HEX", "0x48"), 16)
GAS_CHANNEL  = int(os.getenv("GAS_CHANNEL", "0"))
VREF_VOLTS   = float(os.getenv("VREF_VOLTS", "3.3"))

# ----- thresholds / hysteresis -----
GAS_WARN_ON  = float(os.getenv("GAS_WARN_ON",  "300"))
GAS_WARN_OFF = float(os.getenv("GAS_WARN_OFF", "250"))
GAS_ALRT_ON  = float(os.getenv("GAS_ALRT_ON",  "400"))
GAS_ALRT_OFF = float(os.getenv("GAS_ALRT_OFF", "350"))

EMAIL_COOLDOWN_S_GAS = int(os.getenv("EMAIL_COOLDOWN_S_GAS", "300"))

# ----- email helper (uses send_email from resources/email_notifier.py) -----
def _import_email_notifier():
    here = pathlib.Path(__file__).resolve()
    email_path = here.parent.parent / "resources" / "email_notifier.py"
    spec = importlib.util.spec_from_file_location("email_notifier", str(email_path))
    if not spec or not spec.loader:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

_email = _import_email_notifier()

# ----- ADS1115 access -----
_ads_ctx = {"ads": None, "AnalogIn": None, "addr": None, "last_err": None}

def _get_ads(addr=None):
    if _ads_ctx["ads"] is not None and (addr is None or addr == _ads_ctx["addr"]):
        return _ads_ctx["ads"], _ads_ctx["AnalogIn"]
    if not USE_ADS1115:
        return None, None
    try:
        import board, busio
        import adafruit_ads1x15.ads1115 as ADS
        from adafruit_ads1x15.analog_in import AnalogIn
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c, address=addr if addr is not None else ADS_ADDR_HEX)
        _ads_ctx.update({"ads": ads, "AnalogIn": AnalogIn, "addr": addr or ADS_ADDR_HEX, "last_err": None})
        return ads, AnalogIn
    except Exception as e:
        _ads_ctx.update({"ads": None, "AnalogIn": None, "addr": addr or ADS_ADDR_HEX, "last_err": str(e)})
        return None, None

def _read_voltage(ch: int | None = None) -> float | None:
    ads, AnalogIn = _get_ads(None)
    if ads is None:
        return None
    try:
        import adafruit_ads1x15.ads1115 as ADS
        idx = GAS_CHANNEL if ch is None else int(ch)
        chan = AnalogIn(ads, [ADS.P0, ADS.P1, ADS.P2, ADS.P3][idx])
        return float(chan.voltage)
    except Exception as e:
        _ads_ctx["last_err"] = str(e)
        return None

def _voltage_to_ppm_placeholder(voltage_v: float) -> float:
    ratio = max(0.0, min(1.0, voltage_v / max(0.001, VREF_VOLTS)))
    return float(min(5000.0, ratio * 1000.0))

def _read_hardware() -> float | None:
    v = _read_voltage()
    if v is None:
        return None
    return _voltage_to_ppm_placeholder(v)

def read_value() -> float | None:
    if os.getenv("SIMULATION", "0") == "1":
        return 0.0
    return _read_hardware()

# ----- DB helpers -----
def _db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def _ensure_schema(db_path: str) -> None:
    with _db(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sensor TEXT NOT NULL,
                value REAL NOT NULL,
                ts INTEGER NOT NULL
            );
        """)
        conn.execute("""CREATE INDEX IF NOT EXISTS idx_readings_sensor_ts
                        ON readings(sensor, ts DESC);""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS email_log(
                key TEXT PRIMARY KEY,
                last_sent_ts INTEGER NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_state(
                sensor TEXT PRIMARY KEY,
                level  TEXT NOT NULL,      -- 'normal' | 'warn' | 'alert'
                ts     INTEGER NOT NULL
            );
        """)

def _insert(db_path: str, sensor: str, value: float, ts_ms: int) -> None:
    with _db(db_path) as conn:
        conn.execute("INSERT INTO readings(sensor, value, ts) VALUES(?, ?, ?);",
                     (sensor, float(value), int(ts_ms)))

def _latest(db_path: str, sensor: str) -> dict[str, t.Any] | None:
    with _db(db_path) as conn:
        row = conn.execute(
            "SELECT value, ts FROM readings WHERE sensor=? ORDER BY ts DESC LIMIT 1;",
            (sensor,)
        ).fetchone()
        if not row: return None
        v, ts = row
        return {"ok": True, "sensor": sensor, "value": float(v), "ts": int(ts)}

def _history(db_path: str, sensor: str, limit: int) -> list[dict[str, t.Any]]:
    with _db(db_path) as conn:
        cur = conn.execute(
            "SELECT value, ts FROM readings WHERE sensor=? ORDER BY ts DESC LIMIT ?;",
            (sensor, int(limit))
        )
        return [{"sensor": sensor, "value": float(v), "ts": int(ts)} for (v, ts) in cur.fetchall()]

# ----- alert state + cooldown -----
def _get_state(db_path: str, sensor: str) -> tuple[str,int] | None:
    with _db(db_path) as conn:
        row = conn.execute("SELECT level, ts FROM alert_state WHERE sensor=?;", (sensor,)).fetchone()
        return (row[0], int(row[1])) if row else None

def _set_state(db_path: str, sensor: str, level: str, ts_ms: int) -> None:
    with _db(db_path) as conn:
        conn.execute(
            "INSERT INTO alert_state(sensor, level, ts) VALUES(?,?,?) "
            "ON CONFLICT(sensor) DO UPDATE SET level=excluded.level, ts=excluded.ts;",
            (sensor, level, int(ts_ms))
        )

def _cooldown_ok(db_path: str, key: str, cooldown_s: int) -> bool:
    now_ms = int(time.time() * 1000)
    with _db(db_path) as conn:
        row = conn.execute("SELECT last_sent_ts FROM email_log WHERE key=?;", (key,)).fetchone()
        if row and (now_ms - int(row[0])) < cooldown_s * 1000:
            return False
        conn.execute(
            "INSERT INTO email_log(key,last_sent_ts) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET last_sent_ts=excluded.last_sent_ts;",
            (key, now_ms)
        )
    return True

def _email(subject: str, html: str) -> None:
    if _email and hasattr(_email, "send_email"):
        _email.send_email(subject, html)

def _apply_hysteresis_and_notify(db_path: str, ppm: float, ts_ms: int) -> None:
    # current state
    cur = _get_state(db_path, "gas")
    cur_level = cur[0] if cur else "normal"

    next_level = cur_level
    if cur_level == "normal":
        if ppm >= GAS_ALRT_ON:
            next_level = "alert"
        elif ppm >= GAS_WARN_ON:
            next_level = "warn"
    elif cur_level == "warn":
        if ppm >= GAS_ALRT_ON:
            next_level = "alert"
        elif ppm <= GAS_WARN_OFF:
            next_level = "normal"
    elif cur_level == "alert":
        if ppm <= GAS_ALRT_OFF:
            # drop to warn if still above warn-on; else normal
            next_level = "warn" if ppm >= GAS_WARN_ON else "normal"

    if next_level != cur_level:
        _set_state(db_path, "gas", next_level, ts_ms)
        key = f"gas_{next_level}"
        if _cooldown_ok(db_path, key, EMAIL_COOLDOWN_S_GAS):
            subj = f"[GAS] State: {next_level.upper()}  (ppm={ppm:.0f})"
            body = f"""
            <h3>Gas state changed</h3>
            <p><b>New state:</b> {next_level}</p>
            <p><b>Reading:</b> {ppm:.0f} ppm</p>
            <p><b>Time:</b> {ts_ms} (epoch ms)</p>
            <p>Thresholds: WARN_ON={GAS_WARN_ON}, WARN_OFF={GAS_WARN_OFF},
            ALERT_ON={GAS_ALRT_ON}, ALERT_OFF={GAS_ALRT_OFF}</p>
            """
            _email(subj, body)

# ----- API ops -----
def api_read(CTX: dict, **params):
    db_path = CTX["paths"]["gv_db"]
    _ensure_schema(db_path)

    val = read_value()
    if val is None:
        return {"ok": False, "error": "sensor read failed", "detail": _ads_ctx.get("last_err")}

    ts = int(time.time() * 1000)
    _insert(db_path, "gas", float(val), ts)
    _apply_hysteresis_and_notify(db_path, float(val), ts)
    return {"ok": True, "sensor": "gas", "value": float(val), "ts": ts}

def api_latest(CTX: dict, **params):
    db_path = CTX["paths"]["gv_db"]
    _ensure_schema(db_path)
    row = _latest(db_path, "gas")
    return row or {"ok": False, "error": "no data"}

def api_history(CTX: dict, **params):
    db_path = CTX["paths"]["gv_db"]
    _ensure_schema(db_path)
    limit = int(params.get("limit", 100))
    return {"ok": True, "items": _history(db_path, "gas", limit)}
