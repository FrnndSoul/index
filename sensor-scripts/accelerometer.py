#!/usr/bin/env python3
"""
MPU-6050 accelerometer (I2C) — on-demand read, no database.

Endpoint (via index.py generic dispatcher):
  GET /api/acc/read?recal=1&win_ms=200

Returns:
  {
    "ok": True,
    "ax": <g>, "ay": <g>, "az": <g>,   # filtered + deadbanded
    "amag": <g>,                        # vector magnitude
    "alarm": true|false,                # hysteresis + dwell
    "ts": <epoch_ms>
  }
"""

from __future__ import annotations
import os, time, math
from typing import Optional

# =========================
# Config (env-tunable)
# =========================
I2C_BUS_NUM   = int(os.getenv("I2C_BUS", "1"))
MPU_ADDR      = int(os.getenv("MPU_ADDR", "0x68"), 16)   # 0x68 (AD0=GND), 0x69 (AD0=VCC)
SAMPLE_HZ     = float(os.getenv("SAMPLE_HZ", "50"))      # internal read loop target Hz

# Alarm thresholds (in g), with hysteresis and dwell
ACC_MAG_ON        = float(os.getenv("ACC_MAG_ON",  "1.0"))
ACC_MAG_OFF       = float(os.getenv("ACC_MAG_OFF", "0.8"))
ACC_ALARM_DWELL_MS= int(os.getenv("ACC_ALARM_DWELL_MS", "150"))

# Filtering / tuning
ACC_LPF_ALPHA   = float(os.getenv("ACC_LPF_ALPHA", "0.2"))    # EMA alpha 0..1
ACC_DEADBAND_G  = float(os.getenv("ACC_DEADBAND_G", "0.03"))  # zero-out small jitters
MPU_FS_SEL      = int(os.getenv("MPU_FS_SEL", "0"))           # 0=±2g,1=±4g,2=±8g,3=±16g

# Optional GPIO indicator (LED/Buzzer on BCM18 by default)
GPIO_PIN        = int(os.getenv("GPIO_PIN", "18"))

# =========================
# GPIO (optional)
# =========================
_HAVE_GPIO = False
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
    _HAVE_GPIO = True
except Exception:
    _HAVE_GPIO = False

def _gpio_set(on: bool):
    if _HAVE_GPIO:
        GPIO.output(GPIO_PIN, GPIO.HIGH if on else GPIO.LOW)

# =========================
# I2C / MPU registers
# =========================
from smbus2 import SMBus

PWR_MGMT_1   = 0x6B
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
WHO_AM_I     = 0x75

# =========================
# Module state (persists across calls)
# =========================
_offs = {"ax": 0.0, "ay": 0.0, "az": 0.0}
_filt = {"ax": 0.0, "ay": 0.0, "az": 0.0}
_calibrated = False

_alarm_on = False
_dwell_start_ms: Optional[float] = None

# =========================
# Helpers
# =========================
def _accel_sens_g_per_lsb() -> float:
    return {0:16384.0, 1:8192.0, 2:4096.0, 3:2048.0}[MPU_FS_SEL & 0x03]

def _i2c_read_i16(bus: SMBus, addr: int, reg_h: int) -> int:
    hi = bus.read_byte_data(addr, reg_h)
    lo = bus.read_byte_data(addr, reg_h+1)
    val = (hi << 8) | lo
    if val & 0x8000:
        val = -((65535 - val) + 1)
    return val

def _mpu_init(bus: SMBus) -> bool:
    # Wake + basic config
    bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0x01)  # PLL clock
    time.sleep(0.05)
    bus.write_byte_data(MPU_ADDR, CONFIG, 0x06)      # DLPF ~5Hz
    bus.write_byte_data(MPU_ADDR, GYRO_CONFIG, 0x00)
    afs = (MPU_FS_SEL & 0x03) << 3
    bus.write_byte_data(MPU_ADDR, ACCEL_CONFIG, afs)
    time.sleep(0.05)
    who = bus.read_byte_data(MPU_ADDR, WHO_AM_I)
    return (who & 0x7E) == 0x68

def _calibrate(bus: SMBus, secs: float = 1.0) -> None:
    global _offs, _calibrated
    n = max(20, int(SAMPLE_HZ * secs))
    sens = _accel_sens_g_per_lsb()
    ax=ay=az=0.0
    for _ in range(n):
        ax += _i2c_read_i16(bus, MPU_ADDR, ACCEL_XOUT_H)   / sens
        ay += _i2c_read_i16(bus, MPU_ADDR, ACCEL_XOUT_H+2) / sens
        az += _i2c_read_i16(bus, MPU_ADDR, ACCEL_XOUT_H+4) / sens
        time.sleep(1.0 / SAMPLE_HZ)
    _offs = {"ax": ax/n, "ay": ay/n, "az": az/n}
    _calibrated = True

def _ema_filter(ax: float, ay: float, az: float) -> tuple[float,float,float]:
    # EMA: y += alpha*(x - y)
    a = max(0.01, min(0.99, ACC_LPF_ALPHA))
    _filt["ax"] += a*(ax - _filt["ax"])
    _filt["ay"] += a*(ay - _filt["ay"])
    _filt["az"] += a*(az - _filt["az"])
    return _filt["ax"], _filt["ay"], _filt["az"]

def _deadband(x: float) -> float:
    return 0.0 if abs(x) < ACC_DEADBAND_G else x

def _apply_alarm(amag: float, now_ms: float) -> bool:
    """Hysteresis with dwell; updates module-level state and GPIO."""
    global _alarm_on, _dwell_start_ms
    if not _alarm_on:
        if amag >= ACC_MAG_ON:
            if _dwell_start_ms is None:
                _dwell_start_ms = now_ms
            elif (now_ms - _dwell_start_ms) >= ACC_ALARM_DWELL_MS:
                _alarm_on = True
                _gpio_set(True)
        else:
            _dwell_start_ms = None
    else:
        if amag < ACC_MAG_OFF:
            _alarm_on = False
            _gpio_set(False)
    return _alarm_on

# =========================
# Single-shot read over a short window
# =========================
def _read_window(win_ms: int = 200) -> tuple[float,float,float]:
    """
    Read accelerometer over a short window, apply EMA, return filtered (ax, ay, az) in g.
    """
    sens = _accel_sens_g_per_lsb()
    dt = 1.0 / max(5.0, min(200.0, SAMPLE_HZ))
    t_end = time.time() + (max(10, win_ms) / 1000.0)

    with SMBus(I2C_BUS_NUM) as bus:
        if not _mpu_init(bus):
            raise RuntimeError("MPU-6050 not detected")
        global _calibrated
        if not _calibrated:
            _calibrate(bus, secs=1.0)
        # warmup read
        _ = _i2c_read_i16(bus, MPU_ADDR, ACCEL_XOUT_H)
        # sampling loop
        while time.time() < t_end:
            ax = _i2c_read_i16(bus, MPU_ADDR, ACCEL_XOUT_H)   / sens - _offs["ax"]
            ay = _i2c_read_i16(bus, MPU_ADDR, ACCEL_XOUT_H+2) / sens - _offs["ay"]
            az = _i2c_read_i16(bus, MPU_ADDR, ACCEL_XOUT_H+4) / sens - _offs["az"]
            fx, fy, fz = _ema_filter(ax, ay, az)
            time.sleep(dt)

    # deadband after filter
    return _deadband(_filt["ax"]), _deadband(_filt["ay"]), _deadband(_filt["az"])

# =========================
# API op for index.py
# =========================
def api_read(CTX: dict, **params):
    """
    On-demand read; optional:
      - recal=1  -> force re-calibration of offsets (1s)
      - win_ms   -> sampling window in ms (default 200)
    """
    ts = int(time.time() * 1000)
    try:
        win_ms = int(params.get("win_ms", 200))
    except Exception:
        win_ms = 200

    try:
        with SMBus(I2C_BUS_NUM) as bus:
            if not _mpu_init(bus):
                return {"ok": False, "error": "mpu not detected", "ts": ts}
            if str(params.get("recal", "0")) == "1":
                _calibrate(bus, secs=1.0)
    except Exception as e:
        return {"ok": False, "error": str(e), "ts": ts}

    try:
        ax, ay, az = _read_window(win_ms=win_ms)
        amag = math.sqrt(ax*ax + ay*ay + az*az)
        alarm = _apply_alarm(amag, float(ts))
        return {
            "ok": True,
            "ax": round(ax, 4),
            "ay": round(ay, 4),
            "az": round(az, 4),
            "amag": round(amag, 4),
            "alarm": bool(alarm),
            "ts": ts
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "ts": ts}
