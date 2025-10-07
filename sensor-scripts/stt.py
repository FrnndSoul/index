#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
TTS_FILE = HERE / "text-to-speech.py"

spec = importlib.util.spec_from_file_location("sensor_tts", TTS_FILE)
if not spec or not spec.loader:
    raise ImportError(f"Cannot import {TTS_FILE}")
tts = importlib.util.module_from_spec(spec)
sys.modules["sensor_tts"] = tts
spec.loader.exec_module(tts)  # type: ignore[attr-defined]

def api_langs(CTX: dict, **params):
    langs = [{"code": "tl", "name": "Filipino (tl)"},
                 {"code": "en", "name": "English (en)"}]

    # unique & sorted
    seen = set(); uniq=[]
    for it in langs:
        k = it["code"]
        if k not in seen:
            seen.add(k); uniq.append(it)
    uniq.sort(key=lambda x: x["name"].lower())

    # put common ones on top and add Auto detect
    pin_top = ["tl","en"]
    for pref in reversed(pin_top):
        i = next((idx for idx, it in enumerate(uniq) if it["code"] == pref), None)
        if i is not None:
            uniq.insert(0, uniq.pop(i))
    uniq.insert(0, {"code": "auto", "name": "Auto detect"})
    return uniq
