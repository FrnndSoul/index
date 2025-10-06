"""
resources/storage.py — SQLite helpers for DHT readings
"""
from __future__ import annotations
import sqlite3
from typing import Optional, List, Dict

SCHEMA = """
CREATE TABLE IF NOT EXISTS dht_readings(
  id     INTEGER PRIMARY KEY AUTOINCREMENT,
  ts     INTEGER NOT NULL,
  temp_c REAL,
  hum    REAL
);
CREATE INDEX IF NOT EXISTS idx_dht_ts ON dht_readings(ts DESC);
"""

class Storage:
    def __init__(self, path: str):
        self.path = path

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10, isolation_level=None)

    def ensure_schema(self):
        with self._conn() as con:
            con.executescript(SCHEMA)

    def insert_dht(self, ts:int, temp_c:float, hum:float) -> int:
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO dht_readings(ts,temp_c,hum) VALUES(?,?,?)",
                (ts, temp_c, hum)
            )
            return cur.lastrowid

    def fetch_latest(self) -> Optional[Dict]:
        with self._conn() as con:
            cur = con.execute("SELECT ts,temp_c,hum FROM dht_readings ORDER BY ts DESC LIMIT 1")
            row = cur.fetchone()
            if not row: return None
            ts, temp_c, hum = row
            return {"ts": ts, "temp_c": temp_c, "hum": hum}

    def fetch_history(self, limit:int=100, since:Optional[int]=None) -> List[Dict]:
        with self._conn() as con:
            if since is not None:
                cur = con.execute(
                    "SELECT ts,temp_c,hum FROM dht_readings WHERE ts>=? ORDER BY ts DESC LIMIT ?",
                    (since, limit)
                )
            else:
                cur = con.execute(
                    "SELECT ts,temp_c,hum FROM dht_readings ORDER BY ts DESC LIMIT ?",
                    (limit,)
                )
            return [{"ts": ts, "temp_c": temp_c, "hum": hum} for (ts, temp_c, hum) in cur.fetchall()]
