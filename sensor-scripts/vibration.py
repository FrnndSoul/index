"""
Vibration (SW-420 digital) with email alert + cooldown.

Env (optional):
  VIB_DO_PIN=25
  VIB_DEBOUNCE_MS=30
  VIB_ALERT_THRESHOLD=1.0     # 0.0/1.0 since digital
  EMAIL_COOLDOWN_S_VIB=120
"""

from __future__ import annotations
import os, time, sqlite3, pathlib, importlib.util, typing as t

VIB_DO_PIN    = int(os.getenv("VIB_DO_PIN", "25"))
VIB_DEBOUNCE_MS = int(os.getenv("VIB_DEBOUNCE_MS", "30"))
VIB_ALERT_THRESHOLD = float(os.getenv("VIB_ALERT_THRESHOLD", "1.0"))
EMAIL_COOLDOWN_S_VIB = int(os.getenv("EMAIL_COOLDOWN_S_VIB", "120"))

# ----- email bridge -----
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

# ----- GPIO read with short debounce vote -----
_GPIO_READY = False
def _gpio_ensure():
    global _GPIO_READY
    if _GPIO_READY:
        return
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(VIB_DO_PIN, GPIO.IN)
    _GPIO_READY = True

def _read_hardware() -> float | None:
    try:
        _gpio_ensure()
        import RPi.GPIO as GPIO
        end = time.time() + (VIB_DEBOUNCE_MS / 1000.0)
        highs = lows = 0
        while time.time() < end:
            v = GPIO.input(VIB_DO_PIN)
            if v: highs += 1
            else: lows  += 1
            time.sleep(0.002)
        return 1.0 if highs >= lows else 0.0
    except Exception:
        return None

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

# ----- cooldown + email -----
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

def _maybe_email(db_path: str, value: float, ts_ms: int) -> None:
    if value < VIB_ALERT_THRESHOLD:
        return
    if _cooldown_ok(db_path, "vibration_alert", EMAIL_COOLDOWN_S_VIB):
        if _email and hasattr(_email, "send_email"):
            subj = f"[VIBRATION] Detected (value={value:.1f})"
            body = f"<p>Vibration triggered at {ts_ms} (epoch ms). Value={value:.1f}</p>"
            _email.send_email(subj, body)

# ----- API ops -----
def api_read(CTX: dict, **params):
    db_path = CTX["paths"]["gv_db"]
    _ensure_schema(db_path)
    val = read_value()
    if val is None:
        return {"ok": False, "error": "sensor read failed"}
    ts = int(time.time() * 1000)
    _insert(db_path, "vibration", float(val), ts)
    _maybe_email(db_path, float(val), ts)
    return {"ok": True, "sensor": "vibration", "value": float(val), "ts": ts}

def api_latest(CTX: dict, **params):
    db_path = CTX["paths"]["gv_db"]
    _ensure_schema(db_path)
    row = _latest(db_path, "vibration")
    return row or {"ok": False, "error": "no data"}

def api_history(CTX: dict, **params):
    db_path = CTX["paths"]["gv_db"]
    _ensure_schema(db_path)
    limit = int(params.get("limit", 100))
    return {"ok": True, "items": _history(db_path, "vibration", limit)}
