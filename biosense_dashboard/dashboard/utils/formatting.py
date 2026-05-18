"""
utils/formatting.py — Display formatting helpers for biomedical values.
"""

from typing import Optional


def fmt_bpm(bpm: Optional[float]) -> str:
    if bpm is None: return "---"
    return f"{bpm:.0f}"

def fmt_spo2(spo2: Optional[float]) -> str:
    if spo2 is None: return "---"
    return f"{spo2:.1f}%"

def fmt_pi(pi: Optional[float]) -> str:
    if pi is None: return "---"
    return f"{pi:.2f}%"

def fmt_temp(t: float) -> str:
    return f"{t:.1f} °C"

def fmt_accel(v: float) -> str:
    return f"{v:+.4f} g"

def fmt_gyro(v: float) -> str:
    return f"{v:+.2f} °/s"

def sqi_colour(label: str) -> str:
    return {
        "EXCELLENT": "#00e676",
        "GOOD"     : "#69f0ae",
        "FAIR"     : "#ffeb3b",
        "POOR"     : "#ff9800",
        "NO SIGNAL": "#f44336",
    }.get(label, "#607d8b")

def motion_colour(state: str) -> str:
    return {
        "STABLE"        : "#00e676",
        "LOW MOTION"    : "#69f0ae",
        "MEDIUM MOTION" : "#ffeb3b",
        "HIGH MOTION"   : "#ff9800",
        "INVALID SIGNAL": "#f44336",
    }.get(state, "#607d8b")

def quality_colour(label: str) -> str:
    return {
        "EXCELLENT": "#00e676",
        "GOOD"     : "#69f0ae",
        "FAIR"     : "#ffeb3b",
        "POOR"     : "#ff9800",
        "INVALID"  : "#f44336",
    }.get(label, "#607d8b")

def bpm_colour(bpm: Optional[float]) -> str:
    if bpm is None: return "#607d8b"
    if 50 <= bpm <= 100: return "#00e676"
    if 40 <= bpm <= 120: return "#ffeb3b"
    return "#f44336"

def spo2_colour(spo2: Optional[float]) -> str:
    if spo2 is None: return "#607d8b"
    if spo2 >= 95: return "#00e676"
    if spo2 >= 90: return "#ffeb3b"
    return "#f44336"
