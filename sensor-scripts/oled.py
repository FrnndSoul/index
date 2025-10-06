# sensor-scripts/oled.py
from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont
import time

_device = None
_font = None

def _get_device():
    global _device, _font
    if _device is not None:
        return _device
    try:
        from luma.core.interface.serial import i2c
        from luma.oled.device import sh1106
        serial = i2c(port=1, address=0x3C)
        return sh1106(serial)
    except Exception:
        _device = None
        return None

def api_show(ctx, text:str="", line:int=0, clear:str="1"):
    # legacy single-line draw (kept for compatibility)
    dev = _get_device()
    if dev is None:
        return {"ok": False, "error": "oled not available"}
    W, H = dev.width, dev.height
    img = Image.new('1', (W, H), 0 if (clear not in ("0","false","False")) else 0)
    draw = ImageDraw.Draw(img)
    y = max(0, int(line)) * 10
    draw.text((0, y), (text or "")[:20], fill=255, font=_font)
    dev.display(img)
    return {"ok": True, "ts": int(time.time())}

def api_frame(ctx, lines:list[str]|None=None, clear:str="1"):
    """Draw a full frame (multiple lines) in one update."""
    dev = _get_device()
    if dev is None:
        return {"ok": False, "error": "oled not available"}
    lines = lines or []
    W, H = dev.width, dev.height
    img = Image.new('1', (W, H), 0 if (clear not in ("0","false","False")) else 0)
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        draw.text((0, i*10), (str(text) if text is not None else "")[:21], fill=255, font=_font)
    dev.display(img)
    return {"ok": True, "ts": int(time.time())}
