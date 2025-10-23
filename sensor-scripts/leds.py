"""
sensor-scripts/leds.py
Targets:
  Single LEDs: red(20), green(16), blue(12)  [digital]
  RGB LED: R(26), G(19), B(6)                 [PWM]
Actions:
  on | off | blink | color(#RRGGBB or named color for rgb)
"""

from __future__ import annotations
import os, threading, time, atexit, re
from typing import Dict, Any, Optional

# ----- GPIO import / flags -----
try:
    import RPi.GPIO as GPIO
    _HAVE_GPIO = True
except Exception:
    _HAVE_GPIO = False

# ----- Pin map (env overrides allowed) -----
PIN_SINGLE = {
    "red":   int(os.environ.get("RED_LED",   "20")),
    "green": int(os.environ.get("GREEN_LED", "16")),
    "blue":  int(os.environ.get("BLUE_LED",  "12")),
}
PIN_RGB = {
    "r": int(os.environ.get("RGB_R", "26")),
    "g": int(os.environ.get("RGB_G", "19")),
    "b": int(os.environ.get("RGB_B", "6")),
}
RGB_COMMON_ANODE = os.environ.get("RGB_COMMON_ANODE", "0").lower() in ("1", "true", "yes")
_PWM_FREQ = 1000  # Hz

_state = {
    "single": {"red": "off", "green": "off", "blue": "off"},
    "rgb": {"mode": "off", "hex": "#000000"},
    "blink_threads": {},   # name -> thread
    "blink_flags": {},     # name -> threading.Event
    "inited": False,
    "cleaned": False,
}

_pwm = {"r": None, "g": None, "b": None}  # type: ignore

# -------- Named color support --------
COLOR_MAP = {
    "white":"#ffffff","warmwhite":"#fff4e5","warm white":"#fff4e5","coolwhite":"#f2ffff","cool white":"#f2ffff",
    "red":"#ff0000","green":"#00ff00","blue":"#0000ff","yellow":"#ffff00","orange":"#ffa500","pink":"#ff69b4",
    "hotpink":"#ff69b4","hot pink":"#ff69b4","magenta":"#ff00ff","purple":"#800080","violet":"#8a2be2",
    "indigo":"#4b0082","cyan":"#00ffff","teal":"#008080","lime":"#00ff00","amber":"#ffbf00","gold":"#ffd700",
    "rose":"#ff007f","lavender":"#e6e6fa","sky":"#87ceeb","skyblue":"#87ceeb","sky blue":"#87ceeb",
    "aqua":"#00ffff","mint":"#98ff98","turquoise":"#40e0d0","peach":"#ffcba4"
}
_HEX6_RE = re.compile(r'^#?[0-9a-fA-F]{6}$')
_HEX3_RE = re.compile(r'^#?([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])$')

def _normalize_color(value: str | None) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    s = value.strip().lower()
    if _HEX6_RE.match(s):
        return '#' + s.lstrip('#')
    m3 = _HEX3_RE.match(s)
    if m3:
        r, g, b = m3.groups()
        return '#' + (r*2 + g*2 + b*2)
    key = re.sub(r'\s+', ' ', s).strip()
    if key in COLOR_MAP:
        return COLOR_MAP[key].lower()
    return None

# -------------------------------------

def _gpio_setup_once():
    if _state["inited"]:
        return
    if not _HAVE_GPIO:
        _state["inited"] = True
        return

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    # Single LEDs as outputs
    for pin in PIN_SINGLE.values():
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    # RGB PWM channels
    for pin in PIN_RGB.values():
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    _pwm_r = GPIO.PWM(PIN_RGB["r"], _PWM_FREQ); _pwm_r.start(0)
    _pwm_g = GPIO.PWM(PIN_RGB["g"], _PWM_FREQ); _pwm_g.start(0)
    _pwm_b = GPIO.PWM(PIN_RGB["b"], _PWM_FREQ); _pwm_b.start(0)
    _pwm.update({"r": _pwm_r, "g": _pwm_g, "b": _pwm_b})

    _state["inited"] = True

def _cleanup():
    if _state["cleaned"]:
        return
    # stop blinking threads
    for name, ev in list(_state["blink_flags"].items()):
        try: ev.set()
        except: pass
    for name, th in list(_state["blink_threads"].items()):
        try:
            if th.is_alive():
                th.join(timeout=0.2)
        except: pass
    _state["blink_flags"].clear()
    _state["blink_threads"].clear()

    # turn everything off
    try: _rgb_off()
    except: pass
    for k in _state["single"].keys():
        try: _write_single(k, False)
        except: pass

    # stop PWM and cleanup GPIO
    if _HAVE_GPIO:
        try:
            if _pwm["r"]: _pwm["r"].stop()
            if _pwm["g"]: _pwm["g"].stop()
            if _pwm["b"]: _pwm["b"].stop()
        except: pass
        try: GPIO.cleanup()
        except: pass
    _state["cleaned"] = True

atexit.register(_cleanup)

def _write_single(name: str, on: bool):
    _gpio_setup_once()
    _state["single"][name] = "on" if on else "off"
    if not _HAVE_GPIO:
        return
    GPIO.output(PIN_SINGLE[name], GPIO.HIGH if on else GPIO.LOW)

def _set_rgb_hex(hexcolor: str):
    _gpio_setup_once()
    hexcolor = (hexcolor or "").strip().lstrip('#')
    if len(hexcolor) != 6:
        hexcolor = "ffffff" if _state["rgb"]["mode"] == "on" else "000000"
    r = int(hexcolor[0:2], 16)
    g = int(hexcolor[2:4], 16)
    b = int(hexcolor[4:6], 16)

    _state["rgb"]["mode"] = "color" if (r or g or b) else "off"
    _state["rgb"]["hex"]  = "#" + hexcolor.lower()

    if not _HAVE_GPIO:
        return

    def duty(x):
        pct = max(0, min(100, (x/255.0)*100.0))
        return 100.0 - pct if RGB_COMMON_ANODE else pct

    if _pwm["r"]: _pwm["r"].ChangeDutyCycle(duty(r))
    if _pwm["g"]: _pwm["g"].ChangeDutyCycle(duty(g))
    if _pwm["b"]: _pwm["b"].ChangeDutyCycle(duty(b))

def _rgb_on():
    _set_rgb_hex("ffffff")

def _rgb_off():
    _set_rgb_hex("000000")

def _blink_worker(name: str, is_rgb: bool):
    flag = _state["blink_flags"][name]
    on = False
    while not flag.is_set():
        on = not on
        if is_rgb:
            _rgb_on() if on else _rgb_off()
        else:
            _write_single(name, on)
        time.sleep(0.5)  # 1 Hz
    # ensure off when stopping
    if is_rgb:
        _rgb_off()
        _state["rgb"]["mode"] = "off"; _state["rgb"]["hex"] = "#000000"
    else:
        _write_single(name, False)
        _state["single"][name] = "off"

def _stop_blink(name: str):
    ev = _state["blink_flags"].get(name)
    th = _state["blink_threads"].get(name)
    if ev:
        ev.set()
    if th and th.is_alive():
        try: th.join(timeout=0.2)
        except: pass
    _state["blink_flags"].pop(name, None)
    _state["blink_threads"].pop(name, None)

def _start_blink(name: str, is_rgb: bool):
    _stop_blink(name)
    ev = threading.Event()
    th = threading.Thread(target=_blink_worker, args=(name, is_rgb), daemon=True)
    _state["blink_flags"][name] = ev
    _state["blink_threads"][name] = th
    th.start()

def _snapshot() -> Dict[str, Any]:
    rgb_mode = _state["rgb"]["mode"]
    leds = {
        "red":   _state["single"]["red"],
        "green": _state["single"]["green"],
        "blue":  _state["single"]["blue"],
        "rgb":   ( _state["rgb"]["hex"] if rgb_mode == "color"
                   else ("on" if rgb_mode == "on" else ("blink" if "rgb" in _state["blink_threads"] else "off")) )
    }
    return {"leds": leds}

# -------- API --------

def api_status(ctx: Dict[str, Any]) -> Dict[str, Any]:
    _gpio_setup_once()
    return _snapshot()

def api_set(ctx: Dict[str, Any], action: str, target: str, value: Optional[str] = None) -> Dict[str, Any]:
    """
    POST body: { action, target, value? }
      action: "on" | "off" | "blink" | "color"(rgb only)
      target: "red" | "green" | "blue" | "rgb"
      value:  "#RRGGBB" or named color when action=="color" and target=="rgb"
    """
    _gpio_setup_once()
    target = (target or "").lower()
    action = (action or "").lower()

    if target not in ("red", "green", "blue", "rgb"):
        return {"ok": False, "error": "invalid target"}

    # stop any blinking on that target
    _stop_blink(target if target != "rgb" else "rgb")

    if target == "rgb":
        if action == "on":
            _rgb_on();        _state["rgb"]["mode"] = "on"
        elif action == "off":
            _rgb_off();       _state["rgb"]["mode"] = "off"
        elif action == "blink":
            _start_blink("rgb", True); _state["rgb"]["mode"] = "blink"
        elif action == "color":
            hexv = _normalize_color(value)
            if not hexv:
                return {"ok": False, "error": "color needs #RRGGBB, #RGB, or a known name"}
            _set_rgb_hex(hexv)
        else:
            return {"ok": False, "error": "invalid action"}
    else:
        if action == "on":
            _write_single(target, True)
        elif action == "off":
            _write_single(target, False)
        elif action == "blink":
            _start_blink(target, False)
            _state["single"][target] = "blink"
        else:
            return {"ok": False, "error": "invalid action for single LED"}

    return {"ok": True, **_snapshot()}
