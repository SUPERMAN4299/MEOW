#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        MAX30102 RESEARCH-GRADE OPTICAL BIOSENSING PROTOTYPE                ║
║        Real-Time PPG · SpO2 · Heart Rate · Perfusion Index                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Platform   : Raspberry Pi (any model with hardware I2C)                   ║
║  Sensor     : Maxim MAX30102 — Pulse Oximeter & Heart-Rate Sensor          ║
║  Protocol   : I2C @ 0x57 via smbus2 (no proprietary sensor libs)          ║
║  Python     : 3.9+                                                         ║
║  Version    : 2.0.0  —  Research Prototype                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DISCLAIMER : Non-clinical, research-grade, experimental analysis only.    ║
║               Not approved for medical diagnosis or treatment decisions.    ║
║               All outputs are AI-assisted interpretations for research.     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DEPENDENCIES                                                               ║
║      pip install smbus2 scipy numpy                                        ║
║                                                                             ║
║  HARDWARE WIRING                                                            ║
║      VIN  → 3.3 V   (Pin 1)                                               ║
║      GND  → GND     (Pin 6)                                               ║
║      SDA  → GPIO2   (Pin 3)                                               ║
║      SCL  → GPIO3   (Pin 5)                                               ║
║                                                                             ║
║  ENABLE I2C                                                                 ║
║      sudo raspi-config → Interface Options → I2C → Enable → Reboot        ║
╚══════════════════════════════════════════════════════════════════════════════╝

References:
  [1] Webster J.G. — Design of Pulse Oximeters (IOP, 1997)
  [2] Maxim Integrated — MAX30102 Datasheet Rev 3 (2018)
  [3] Mendelson Y. — Pulse Oximetry (IEEE EMBS, 2002)
  [4] Allen J. — Photoplethysmography and its application (Physiol. Meas., 2007)
"""

# ─────────────────────────────────────────────────────────────────────────────
#  STDLIB
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

# ─────────────────────────────────────────────────────────────────────────────
#  NUMPY  (required)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    print("[FATAL] numpy not found.  pip install numpy")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  SCIPY  (strongly recommended — graceful degradation without it)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from scipy.signal import butter, filtfilt, sosfiltfilt, butter as _b
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  SMBUS2  (required on Raspberry Pi)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import smbus2
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  §1  SYSTEM CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class CFG:
    """
    Central configuration namespace.
    All tunable parameters live here — change once, applies everywhere.
    """

    # ── I2C / Hardware ────────────────────────────────────────────────────
    I2C_BUS         : int   = 1        # RPi bus (1 on all modern Pi models)
    SENSOR_ADDR     : int   = 0x57     # MAX30102 factory-fixed address

    # ── Acquisition ───────────────────────────────────────────────────────
    SAMPLE_RATE     : int   = 100      # Hz  (50/100/200/400/800/1000)
    LED_CURRENT     : int   = 0x3C     # ~12 mA  (0x00=0 mA … 0xFF≈51 mA)
    ADC_RANGE       : int   = 0x20     # 4096 nA full-scale sensitivity
    PULSE_WIDTH     : int   = 0x03     # 411 µs → 18-bit effective resolution

    # ── Ring-buffer / window sizes ────────────────────────────────────────
    RING_SIZE       : int   = 500      # Deep ring buffer (5 s @ 100 Hz)
    CALC_WINDOW     : int   = 200      # Samples per analysis frame (2 s)
    STEP_SIZE       : int   = 50       # Overlap stride (75 % overlap)
    DISPLAY_AVG_N   : int   = 6        # Readings averaged for display

    # ── Bandpass filter (Butterworth, zero-phase) ─────────────────────────
    BPF_LOW_HZ      : float = 0.5      # High-pass corner (removes baseline)
    BPF_HIGH_HZ     : float = 4.0      # Low-pass  corner (removes HF noise)
    BPF_ORDER       : int   = 4        # Filter order (higher → sharper)

    # ── Baseline / DC removal ─────────────────────────────────────────────
    BASELINE_TAU    : float = 0.995    # IIR forgetting factor (≈10 s TC)

    # ── Peak detection ────────────────────────────────────────────────────
    PEAK_MIN_BPM    : int   = 30       # Physiological lower limit
    PEAK_MAX_BPM    : int   = 220      # Physiological upper limit
    PEAK_PROM_FRAC  : float = 0.30     # Prominence ≥ 30 % of signal range
    PEAK_WLEN_S     : float = 0.8      # Prominence window width (seconds)

    # ── Finger detection ─────────────────────────────────────────────────
    FINGER_DC_MIN   : int   = 50_000   # Minimum IR DC for tissue contact
    FINGER_AC_MIN   : int   = 500      # Minimum IR AC for pulsatile signal

    # ── Signal quality thresholds ─────────────────────────────────────────
    SQI_AC_DC_LOW   : float = 0.002    # AC/DC too low  → no signal
    SQI_AC_DC_HIGH  : float = 0.30     # AC/DC too high → motion / clipping
    SQI_CV_MAX      : float = 0.20     # Peak-interval CV ceiling (stability)

    # ── Perfusion index ───────────────────────────────────────────────────
    PI_SMOOTH_N     : int   = 4        # Readings to average for PI display
    PI_LOW_THRESH   : float = 0.2      # PI below this → low perfusion warn

    # ── Display ───────────────────────────────────────────────────────────
    REFRESH_HZ      : float = 1.0      # Terminal refresh rate (Hz)
    TERM_WIDTH      : int   = 72       # Console line width


# ══════════════════════════════════════════════════════════════════════════════
#  §2  MAX30102 REGISTER MAP  (Maxim Datasheet Rev 3, §8.4)
# ══════════════════════════════════════════════════════════════════════════════

class REG:
    INT_STATUS1 = 0x00;  INT_STATUS2 = 0x01
    INT_ENABLE1 = 0x02;  INT_ENABLE2 = 0x03
    FIFO_WR_PTR = 0x04;  OVF_COUNTER = 0x05;  FIFO_RD_PTR = 0x06
    FIFO_DATA   = 0x07
    FIFO_CONFIG = 0x08
    MODE_CONFIG = 0x09
    SPO2_CONFIG = 0x0A
    LED1_PA     = 0x0C   # RED  LED pulse amplitude
    LED2_PA     = 0x0D   # IR   LED pulse amplitude
    PILOT_PA    = 0x10
    TEMP_INT    = 0x1F;  TEMP_FRAC   = 0x20;  TEMP_CONFIG = 0x21
    REV_ID      = 0xFE;  PART_ID     = 0xFF   # 0x15 = MAX30102

class MODE:
    HR    = 0x02   # IR only
    SPO2  = 0x03   # RED + IR
    RESET = 0x40   # Soft reset
    SHDN  = 0x80   # Shutdown

SAMPLE_RATE_BITS = {50:0x00, 100:0x01, 200:0x02,
                    400:0x03, 800:0x04, 1000:0x05}


# ══════════════════════════════════════════════════════════════════════════════
#  §3  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VitalFrame:
    """Snapshot of all computed biomedical parameters for one analysis frame."""
    timestamp     : float = 0.0
    bpm           : Optional[float] = None
    spo2          : Optional[float] = None
    pi            : Optional[float] = None   # Perfusion Index (%)
    sqi           : float = 0.0              # Signal Quality Index 0-100
    sqi_label     : str   = "NO SIGNAL"
    finger_on     : bool  = False
    dc_ir         : float = 0.0
    ac_ir         : float = 0.0
    noise_est     : float = 0.0
    peak_cv       : Optional[float] = None   # Coefficient of variation of RR
    alerts        : List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
#  §4  LOW-LEVEL I2C DRIVER
# ══════════════════════════════════════════════════════════════════════════════

def _reg_write(bus, reg: int, val: int) -> None:
    bus.write_byte_data(CFG.SENSOR_ADDR, reg, val & 0xFF)

def _reg_read(bus, reg: int) -> int:
    return bus.read_byte_data(CFG.SENSOR_ADDR, reg)

def _reg_block(bus, reg: int, n: int) -> bytes:
    return bytes(bus.read_i2c_block_data(CFG.SENSOR_ADDR, reg, n))


def sensor_init(bus) -> bool:
    """
    Initialise MAX30102 for dual-channel (RED + IR) SpO2 acquisition.

    Sequence:
      1. Part-ID verification  →  confirms genuine MAX30102
      2. Soft reset            →  clean power-on state
      3. FIFO config           →  4× HW average, roll-over enabled
      4. SpO2 config           →  ADC range, sample rate, LED pulse width
      5. LED current           →  RED and IR drive current
      6. Operating mode        →  SpO2 (both LEDs active)
      7. Clear FIFO pointers   →  start fresh

    Returns True if sensor is detected and configured correctly.
    """
    try:
        pid = _reg_read(bus, REG.PART_ID)
    except OSError as exc:
        _err(f"I2C error on bus {CFG.I2C_BUS} @ 0x{CFG.SENSOR_ADDR:02X}: {exc}")
        _err("Check wiring, I2C enable (raspi-config), and power supply.")
        return False

    if pid != 0x15:
        _err(f"Part ID mismatch: got 0x{pid:02X}, expected 0x15 (MAX30102)")
        return False

    # Soft reset — wait for bit to self-clear (max 1 ms per datasheet)
    _reg_write(bus, REG.MODE_CONFIG, MODE.RESET)
    for _ in range(20):
        time.sleep(0.005)
        if not (_reg_read(bus, REG.MODE_CONFIG) & MODE.RESET):
            break

    # FIFO: SMP_AVE=4 (b010 << 5), FIFO_ROLLOVER_EN=1, FIFO_A_FULL=0xF
    _reg_write(bus, REG.FIFO_CONFIG, 0x5F)

    # SPO2_CONFIG: [6:5] ADC range | [4:2] sample rate | [1:0] pulse width
    sr_bits = SAMPLE_RATE_BITS.get(CFG.SAMPLE_RATE, 0x01)
    _reg_write(bus, REG.SPO2_CONFIG,
               CFG.ADC_RANGE | (sr_bits << 2) | CFG.PULSE_WIDTH)

    # LED current (both channels equal — adjust CFG.LED_CURRENT)
    _reg_write(bus, REG.LED1_PA, CFG.LED_CURRENT)   # RED
    _reg_write(bus, REG.LED2_PA, CFG.LED_CURRENT)   # IR

    # SpO2 mode: RED + IR both active
    _reg_write(bus, REG.MODE_CONFIG, MODE.SPO2)

    # Clear FIFO
    _reg_write(bus, REG.FIFO_WR_PTR, 0)
    _reg_write(bus, REG.OVF_COUNTER, 0)
    _reg_write(bus, REG.FIFO_RD_PTR, 0)

    return True


def read_fifo_sample(bus) -> Tuple[int, int]:
    """
    Decode one 6-byte FIFO record into (RED, IR) 18-bit ADC values.

    FIFO byte layout (SpO2 mode):
        [0..2] RED  — bits 17:0 packed into 3 bytes (MSB first)
        [3..5] IR   — bits 17:0 packed into 3 bytes (MSB first)
    """
    try:
        raw = _reg_block(bus, REG.FIFO_DATA, 6)
        red = ((raw[0] << 16) | (raw[1] << 8) | raw[2]) & 0x3FFFF
        ir  = ((raw[3] << 16) | (raw[4] << 8) | raw[5]) & 0x3FFFF
        return red, ir
    except OSError:
        return 0, 0


# ══════════════════════════════════════════════════════════════════════════════
#  §5  ACQUISITION THREAD / RING BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class RingBuffer:
    """
    Fixed-size circular buffer for streaming PPG samples.
    Thread-safe append; numpy-slice for analysis windows.
    """
    def __init__(self, maxlen: int):
        self._buf  = deque(maxlen=maxlen)

    def append(self, val: float) -> None:
        self._buf.append(val)

    def as_array(self) -> np.ndarray:
        return np.array(self._buf, dtype=np.float64)

    def __len__(self) -> int:
        return len(self._buf)


def collect_window(bus,
                   ir_ring  : RingBuffer,
                   red_ring : RingBuffer,
                   n_samples: int) -> bool:
    """
    Append `n_samples` new (RED, IR) pairs into the ring buffers.
    Uses precise sleep timing to honour SAMPLE_RATE.

    Returns True if all samples were read without error.
    """
    dt      = 1.0 / CFG.SAMPLE_RATE
    t_next  = time.perf_counter() + dt
    ok      = True

    for _ in range(n_samples):
        red, ir = read_fifo_sample(bus)
        ir_ring.append(float(ir))
        red_ring.append(float(red))
        now = time.perf_counter()
        sleep_t = t_next - now
        if sleep_t > 0:
            time.sleep(sleep_t)
        t_next += dt
        if ir == 0 and red == 0:
            ok = False

    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  §6  SIGNAL PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

# ── 6.1  IIR Baseline Tracker ─────────────────────────────────────────────

class BaselineTracker:
    """
    Single-pole IIR (exponential moving average) baseline estimator.

    Models the slow-moving DC component of the PPG waveform. By
    subtracting this estimate we perform adaptive baseline removal
    without the group-delay artefacts of a finite-impulse filter.

        y[n] = τ·y[n-1] + (1-τ)·x[n]

    τ (BASELINE_TAU) controls the time constant:
        τ = 0.995 → TC ≈ 1/( (1-τ)·fs ) ≈ 10 s @ 100 Hz
    """
    def __init__(self):
        self._state : Optional[float] = None

    def reset(self) -> None:
        self._state = None

    def process(self, signal: np.ndarray) -> np.ndarray:
        τ = CFG.BASELINE_TAU
        out = np.empty_like(signal)
        s   = self._state if self._state is not None else signal[0]
        for i, x in enumerate(signal):
            s       = τ * s + (1 - τ) * x
            out[i]  = x - s           # AC-coupled output
        self._state = s
        return out


# ── 6.2  Butterworth Bandpass ──────────────────────────────────────────────

def _design_bpf(fs: float):
    """
    Design a zero-phase Butterworth bandpass filter.
    Returns (b, a) IIR coefficients for use with filtfilt().
    Cached after first call to avoid repeated design overhead.
    """
    nyq = 0.5 * fs
    low = CFG.BPF_LOW_HZ  / nyq
    hi  = CFG.BPF_HIGH_HZ / nyq
    return butter(CFG.BPF_ORDER, [low, hi], btype="band")


_bpf_cache = {}

def bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply zero-phase Butterworth BPF.  Requires scipy.
    filtfilt() achieves zero phase distortion by forward-backward pass.
    Falls back to moving average if scipy unavailable or signal too short.
    """
    min_len = 3 * (2 * CFG.BPF_ORDER) + 1
    if SCIPY_AVAILABLE and len(signal) >= min_len:
        key = (fs, CFG.BPF_ORDER, CFG.BPF_LOW_HZ, CFG.BPF_HIGH_HZ)
        if key not in _bpf_cache:
            _bpf_cache[key] = _design_bpf(fs)
        b, a = _bpf_cache[key]
        try:
            return filtfilt(b, a, signal)
        except Exception:
            pass
    # Fallback: weighted moving average (Gaussian-like)
    w = int(fs / CFG.BPF_HIGH_HZ)
    w = max(3, w | 1)                        # ensure odd width ≥ 3
    kernel = np.hamming(w);  kernel /= kernel.sum()
    return np.convolve(signal, kernel, mode="same")


# ── 6.3  Waveform Normalisation ───────────────────────────────────────────

def normalize_waveform(signal: np.ndarray) -> np.ndarray:
    """
    Min-max normalise signal to [-1, +1].
    Needed for threshold-independent peak detection.
    Returns zero array if signal is flat (no oscillation).
    """
    rng = signal.max() - signal.min()
    if rng < 1e-9:
        return np.zeros_like(signal)
    return 2.0 * (signal - signal.min()) / rng - 1.0


# ── 6.4  Full Filter Pipeline ─────────────────────────────────────────────

def process_ppg(raw: np.ndarray,
                baseline_tracker: BaselineTracker,
                fs: float) -> np.ndarray:
    """
    Complete PPG signal conditioning pipeline:

      1. IIR baseline removal  — removes slow drift & DC offset
      2. Butterworth BPF       — retains 0.5–4.0 Hz (40–240 BPM band)
      3. Waveform normalisation — scale-independent amplitude

    Output is a clean, normalised PPG waveform ready for peak detection.
    """
    ac   = baseline_tracker.process(raw)    # step 1: adaptive DC removal
    filt = bandpass_filter(ac, fs)          # step 2: bandpass
    norm = normalize_waveform(filt)         # step 3: normalise
    return norm


# ── 6.5  Noise Estimation ────────────────────────────────────────────────

def estimate_noise(raw: np.ndarray) -> float:
    """
    Estimate high-frequency noise power using first-difference variance.

    The first difference dx[n] = x[n] - x[n-1] amplifies high-frequency
    content. Its standard deviation approximates the noise floor.
    Normalised by signal range to give a dimensionless noise ratio.
    """
    if len(raw) < 4:
        return 1.0
    dx  = np.diff(raw)
    rng = raw.max() - raw.min()
    if rng < 1e-6:
        return 1.0
    return float(np.std(dx) / rng)


# ══════════════════════════════════════════════════════════════════════════════
#  §7  PEAK DETECTION  (validated, multi-criterion)
# ══════════════════════════════════════════════════════════════════════════════

def detect_peaks(normed: np.ndarray, fs: float) -> np.ndarray:
    """
    Robust systolic peak detector for PPG signals.

    Strategy:
      • Use scipy.signal.find_peaks with physiological constraints when
        scipy is available (recommended path).
      • Fall back to a threshold-crossing detector otherwise.

    Constraints applied:
      - Minimum inter-peak distance  : fs·60/PEAK_MAX_BPM  samples
      - Maximum inter-peak distance  : fs·60/PEAK_MIN_BPM  samples
      - Minimum prominence           : PEAK_PROM_FRAC × signal range
      - Prominence window            : PEAK_WLEN_S seconds

    Returns array of peak sample indices.
    """
    min_dist  = int(fs * 60 / CFG.PEAK_MAX_BPM)
    max_dist  = int(fs * 60 / CFG.PEAK_MIN_BPM)
    prom_min  = CFG.PEAK_PROM_FRAC * (normed.max() - normed.min())
    wlen      = int(CFG.PEAK_WLEN_S * fs)

    if SCIPY_AVAILABLE:
        peaks, props = find_peaks(
            normed,
            distance   = max(1, min_dist),
            prominence = max(0.01, prom_min),
            wlen       = max(3, wlen),
        )
        return peaks
    else:
        # Fallback: simple threshold-crossing with refractor period
        threshold = np.mean(normed) + 0.1 * np.std(normed)
        peaks = []
        for i in range(1, len(normed) - 1):
            if (normed[i] > threshold
                    and normed[i] > normed[i-1]
                    and normed[i] > normed[i+1]):
                if not peaks or (i - peaks[-1]) >= min_dist:
                    peaks.append(i)
        return np.array(peaks, dtype=int)


def validate_peaks(peaks: np.ndarray, fs: float) -> np.ndarray:
    """
    Remove physiologically implausible peaks by filtering intervals.

    Any inter-peak interval that implies a heart rate outside
    [PEAK_MIN_BPM, PEAK_MAX_BPM] is flagged. Runs of consistent
    intervals are kept; isolated outliers are discarded.
    """
    if len(peaks) < 2:
        return peaks

    intervals = np.diff(peaks) / fs          # seconds
    bpm_each  = 60.0 / intervals

    valid_mask = (bpm_each >= CFG.PEAK_MIN_BPM) & (bpm_each <= CFG.PEAK_MAX_BPM)

    # Retain peak if both surrounding intervals are valid
    keep = np.ones(len(peaks), dtype=bool)
    for i in range(len(valid_mask)):
        if not valid_mask[i]:
            keep[i]     = False
            keep[i + 1] = False

    return peaks[keep]


# ══════════════════════════════════════════════════════════════════════════════
#  §8  VITAL SIGN COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

# ── 8.1  Heart Rate ───────────────────────────────────────────────────────

def compute_bpm(peaks: np.ndarray, fs: float) -> Optional[float]:
    """
    Compute heart rate from validated peak locations.

    Uses the median inter-peak interval (robust to outlier beats).
    Median preferred over mean to suppress ectopic beats.

    Returns BPM float or None if fewer than 2 valid peaks.
    """
    if len(peaks) < 2:
        return None
    intervals_s = np.diff(peaks) / fs
    median_s    = float(np.median(intervals_s))
    if median_s <= 0:
        return None
    bpm = 60.0 / median_s
    return round(bpm, 1) if CFG.PEAK_MIN_BPM <= bpm <= CFG.PEAK_MAX_BPM else None


# ── 8.2  SpO2  (Ratio-of-Ratios) ─────────────────────────────────────────

# Empirical SpO2 calibration table (R → SpO2 %).
# Derived from the simplified Beer-Lambert model for haemoglobin.
# Real pulse oximeters use factory-calibrated lookup tables.
# Reference: Webster (1997), Eq 10.11.
_SPO2_R_TABLE = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
_SPO2_V_TABLE = np.array([100., 99., 98., 97., 96., 95., 93., 91., 88., 85., 80.])


def _r_to_spo2(R: float) -> float:
    """Interpolate SpO2 from ratio R using empirical calibration table."""
    R = max(_SPO2_R_TABLE[0], min(_SPO2_R_TABLE[-1], R))
    return float(np.interp(R, _SPO2_R_TABLE, _SPO2_V_TABLE))


def compute_spo2(ir_raw  : np.ndarray,
                 red_raw : np.ndarray,
                 peaks   : np.ndarray) -> Optional[float]:
    """
    Estimate SpO2 via the Ratio-of-Ratios (RoR) method.

    For each detected pulse cycle:
        AC_ch = peak-to-trough amplitude  (pulsatile component)
        DC_ch = local mean within cycle   (tissue/venous baseline)

        R = (AC_red / DC_red) / (AC_ir / DC_ir)
        SpO2 = interp(R, calibration_table)

    Per-cycle R values are median-aggregated to suppress noisy beats.

    Reference: [1] §10, [3]
    """
    if len(peaks) < 2 or len(ir_raw) < 10 or len(red_raw) < 10:
        return None

    r_vals = []
    for i in range(len(peaks) - 1):
        seg_ir  = ir_raw [peaks[i]:peaks[i+1]]
        seg_red = red_raw[peaks[i]:peaks[i+1]]
        if len(seg_ir) < 3:
            continue
        dc_ir  = float(np.mean(seg_ir))
        dc_red = float(np.mean(seg_red))
        if dc_ir < 1 or dc_red < 1:
            continue
        ac_ir  = float(seg_ir.max()  - seg_ir.min())
        ac_red = float(seg_red.max() - seg_red.min())
        if ac_ir < 1:
            continue
        R = (ac_red / dc_red) / (ac_ir / dc_ir)
        if 0.3 <= R <= 1.8:              # physiologically plausible range
            r_vals.append(R)

    if not r_vals:
        return None

    R_median = float(np.median(r_vals))
    spo2     = _r_to_spo2(R_median)
    return round(max(70.0, min(100.0, spo2)), 1)


# ── 8.3  Perfusion Index ─────────────────────────────────────────────────

def compute_pi(ir_raw: np.ndarray) -> Optional[float]:
    """
    Perfusion Index (PI) — measures peripheral perfusion strength.

    Formula (Mendelson 2002):
        PI (%) = (AC_ir / DC_ir) × 100

    Where:
        AC_ir = RMS amplitude of the pulsatile (AC) IR signal
        DC_ir = Mean (DC) of raw IR channel

    Clinical context (non-diagnostic reference values):
        PI < 0.2 %  : very low perfusion (cold extremities, vasoconstriction)
        PI 0.2–2 %  : normal resting peripheral perfusion
        PI > 2 %    : high perfusion (warm, vasodilated)

    Note: This is experimental / research-grade — not a clinical measurement.
    """
    if len(ir_raw) < 10:
        return None
    dc = float(np.mean(ir_raw))
    if dc < 1:
        return None
    # RMS of AC component (after mean removal)
    ac_rms = float(np.sqrt(np.mean((ir_raw - dc) ** 2)))
    pi = (ac_rms / dc) * 100.0
    return round(max(0.0, min(25.0, pi)), 3)


# ── 8.4  Peak Interval Variability ───────────────────────────────────────

def peak_interval_cv(peaks: np.ndarray, fs: float) -> Optional[float]:
    """
    Coefficient of Variation (CV) of RR intervals.

    CV = std(RR) / mean(RR)

    Low CV (< 0.05) → metronomic, stable rhythm.
    High CV (> 0.20) → irregular rhythm or motion artefact.
    Used as a waveform consistency metric in SQI computation.
    """
    if len(peaks) < 3:
        return None
    intervals = np.diff(peaks) / fs
    m = float(np.mean(intervals))
    if m < 1e-6:
        return None
    return float(np.std(intervals) / m)


# ══════════════════════════════════════════════════════════════════════════════
#  §9  SIGNAL QUALITY INDEX  (multi-dimensional)
# ══════════════════════════════════════════════════════════════════════════════

def compute_sqi(ir_raw   : np.ndarray,
                normed   : np.ndarray,
                peaks    : np.ndarray,
                fs       : float,
                noise_est: float) -> Tuple[float, str]:
    """
    Multi-dimensional Signal Quality Index (SQI), score 0–100.

    Component scores (each 0–100, then weighted average):

      W1 (30 %) — AC/DC ratio (perfusion adequacy)
        Maps the IR AC/DC ratio to a quality score.
        Too low → no signal; too high → saturation or motion.

      W2 (30 %) — Noise floor (HF noise estimation)
        Low noise ratio → high score.
        Above 0.3 → signal unusable.

      W3 (25 %) — Peak spacing consistency (rhythm stability)
        CV of RR intervals → stable = good, irregular = poor.
        Absent peaks → 0.

      W4 (15 %) — Peak count plausibility
        At least 2 peaks required; more peaks over window → higher confidence.

    SQI Label thresholds:
        90–100 : EXCELLENT
        70–89  : GOOD
        45–69  : FAIR
        10–44  : POOR
        0–9    : NO SIGNAL
    """
    dc = float(np.mean(ir_raw)) if len(ir_raw) else 1.0
    ac = float(np.std(ir_raw))  if len(ir_raw) else 0.0

    # — Component 1: AC/DC ratio quality ─────────────────────────────────
    ac_dc = ac / dc if dc > 0 else 0.0
    if   ac_dc < CFG.SQI_AC_DC_LOW:  w1 = 0.0
    elif ac_dc > CFG.SQI_AC_DC_HIGH: w1 = max(0.0, 100.0 - (ac_dc - CFG.SQI_AC_DC_HIGH) * 500)
    else:
        # Optimal AC/DC range: 0.01–0.05 for normal PPG
        optimal = 0.03
        w1 = 100.0 * math.exp(-((ac_dc - optimal) ** 2) / (2 * 0.02 ** 2))

    # — Component 2: Noise floor score ────────────────────────────────────
    w2 = max(0.0, 100.0 * (1.0 - noise_est / 0.30))

    # — Component 3: RR interval consistency ──────────────────────────────
    cv = peak_interval_cv(peaks, fs)
    if cv is None:
        w3 = 0.0
    else:
        w3 = max(0.0, 100.0 * (1.0 - cv / CFG.SQI_CV_MAX))

    # — Component 4: Peak count plausibility ──────────────────────────────
    expected_peaks = (len(ir_raw) / fs) * (80 / 60)   # assume 80 BPM nominal
    peak_ratio = len(peaks) / max(1.0, expected_peaks)
    w4 = max(0.0, min(100.0, 100.0 * min(peak_ratio, 1.0 / max(peak_ratio, 0.01)) ))

    # — Weighted aggregate ─────────────────────────────────────────────────
    sqi = 0.30 * w1 + 0.30 * w2 + 0.25 * w3 + 0.15 * w4

    # — Label ──────────────────────────────────────────────────────────────
    if   sqi >= 90: label = "EXCELLENT"
    elif sqi >= 70: label = "GOOD"
    elif sqi >= 45: label = "FAIR"
    elif sqi >= 10: label = "POOR"
    else:           label = "NO SIGNAL"

    return round(sqi, 1), label


# ══════════════════════════════════════════════════════════════════════════════
#  §10  FINGER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_finger(ir_raw: np.ndarray) -> bool:
    """
    Two-criterion finger contact detector:

      1. DC criterion: mean IR value must exceed tissue contact threshold.
         Without a finger the sensor reads near-zero (ambient light blocked).

      2. AC criterion: pulsatile amplitude must be detectable.
         A finger at rest still shows cardiac pulsations; a hard surface does not.

    Both criteria must pass simultaneously.
    """
    if len(ir_raw) == 0:
        return False
    dc = float(np.mean(ir_raw))
    ac = float(ir_raw.max() - ir_raw.min())
    return dc > CFG.FINGER_DC_MIN and ac > CFG.FINGER_AC_MIN


# ══════════════════════════════════════════════════════════════════════════════
#  §11  AI-ASSISTED SIGNAL INTERPRETATION  (experimental, non-clinical)
# ══════════════════════════════════════════════════════════════════════════════

def interpret_signal(frame: VitalFrame) -> List[str]:
    """
    Lightweight rule-based signal interpretation engine.

    Generates contextual alerts based on the computed vital frame.
    All outputs are labelled as experimental / non-clinical.

    Alert categories:
      • Contact quality     — finger placement issues
      • Perfusion adequacy  — PI-based peripheral circulation commentary
      • Rhythm consistency  — RR interval variability warnings
      • Signal integrity    — noise and waveform artefacts
      • Physiological range — values outside expected normal windows

    IMPORTANT: These are AI-assisted optical signal observations.
               They do NOT constitute medical diagnosis or advice.
    """
    alerts = []

    if not frame.finger_on:
        alerts.append("CONTACT  | No finger detected — place finger flat on sensor")
        return alerts

    if frame.sqi_label in ("NO SIGNAL", "POOR"):
        alerts.append("SIGNAL   | Poor optical coupling — press finger more firmly")
        return alerts

    # Perfusion index interpretation
    if frame.pi is not None:
        if frame.pi < CFG.PI_LOW_THRESH:
            alerts.append(f"PERFUSION| Low PI={frame.pi:.2f}% — possible vasoconstriction or cold finger")
        elif frame.pi > 8.0:
            alerts.append(f"PERFUSION| High PI={frame.pi:.2f}% — strong peripheral perfusion")

    # Rhythm consistency
    if frame.peak_cv is not None:
        if frame.peak_cv > 0.20:
            alerts.append(f"RHYTHM   | High beat variability (CV={frame.peak_cv:.2f}) — motion or arrhythmia?")
        elif frame.peak_cv < 0.03:
            alerts.append(f"RHYTHM   | Very regular pulse (CV={frame.peak_cv:.3f}) — excellent waveform stability")

    # BPM range commentary
    if frame.bpm is not None:
        if frame.bpm > 100:
            alerts.append(f"RATE     | Elevated rate ({frame.bpm} BPM) — tachycardia range (research-grade obs.)")
        elif frame.bpm < 50:
            alerts.append(f"RATE     | Low rate ({frame.bpm} BPM) — bradycardia range (research-grade obs.)")

    # SpO2 range commentary
    if frame.spo2 is not None:
        if frame.spo2 < 94:
            alerts.append(f"SPO2     | Low optical SpO2={frame.spo2}% — experimental signal only, verify clinically")

    # Noise floor warning
    if frame.noise_est > 0.15:
        alerts.append(f"NOISE    | Elevated signal noise ({frame.noise_est:.3f}) — minimise motion")

    if not alerts:
        alerts.append("STATUS   | Waveform within normal research parameters")

    return alerts


# ══════════════════════════════════════════════════════════════════════════════
#  §12  ANALYSIS ENGINE  (single frame computation)
# ══════════════════════════════════════════════════════════════════════════════

class AnalysisEngine:
    """
    Stateful analysis engine that processes one overlapping window per call.

    Maintains:
      - BaselineTracker for adaptive DC removal (persistent IIR state)
      - Rolling history deques for temporal smoothing of vital estimates
    """

    def __init__(self):
        self._baseline_ir  = BaselineTracker()
        self._baseline_red = BaselineTracker()
        self._bpm_hist     = deque(maxlen=CFG.DISPLAY_AVG_N)
        self._spo2_hist    = deque(maxlen=CFG.DISPLAY_AVG_N)
        self._pi_hist      = deque(maxlen=CFG.PI_SMOOTH_N)
        self._fs           = float(CFG.SAMPLE_RATE)

    def reset(self) -> None:
        """Reset all state — call when finger is removed."""
        self._baseline_ir.reset()
        self._baseline_red.reset()
        self._bpm_hist.clear()
        self._spo2_hist.clear()
        self._pi_hist.clear()

    def analyse(self,
                ir_ring  : RingBuffer,
                red_ring : RingBuffer) -> VitalFrame:
        """
        Run the full analysis pipeline on the current ring-buffer contents.

        Returns a populated VitalFrame.
        """
        frame = VitalFrame(timestamp=time.time())

        # ── Extract analysis window ───────────────────────────────────────
        ir_raw  = ir_ring.as_array()[-CFG.CALC_WINDOW:]
        red_raw = red_ring.as_array()[-CFG.CALC_WINDOW:]

        if len(ir_raw) < 30:
            frame.sqi_label = "NO SIGNAL"
            return frame

        # ── Finger detection ──────────────────────────────────────────────
        frame.finger_on = detect_finger(ir_raw)
        if not frame.finger_on:
            self.reset()
            frame.sqi_label = "NO SIGNAL"
            frame.alerts    = interpret_signal(frame)
            return frame

        # ── DC / AC characterisation ──────────────────────────────────────
        frame.dc_ir = float(np.mean(ir_raw))
        frame.ac_ir = float(np.std(ir_raw))

        # ── Noise estimation (on raw signal) ─────────────────────────────
        frame.noise_est = estimate_noise(ir_raw)

        # ── PPG conditioning ──────────────────────────────────────────────
        normed_ir = process_ppg(ir_raw.copy(), self._baseline_ir, self._fs)

        # ── Peak detection & validation ───────────────────────────────────
        peaks     = detect_peaks(normed_ir, self._fs)
        peaks_val = validate_peaks(peaks, self._fs)

        # ── Signal Quality Index ──────────────────────────────────────────
        frame.sqi, frame.sqi_label = compute_sqi(
            ir_raw, normed_ir, peaks_val, self._fs, frame.noise_est
        )

        # ── Peak interval CV ──────────────────────────────────────────────
        frame.peak_cv = peak_interval_cv(peaks_val, self._fs)

        # ── Vital signs (only when signal is acceptable) ──────────────────
        if frame.sqi >= 10 and len(peaks_val) >= 2:
            bpm  = compute_bpm(peaks_val, self._fs)
            spo2 = compute_spo2(ir_raw, red_raw, peaks_val)
            pi   = compute_pi(ir_raw)

            if bpm  is not None: self._bpm_hist.append(bpm)
            if spo2 is not None: self._spo2_hist.append(spo2)
            if pi   is not None: self._pi_hist.append(pi)

        # ── Temporally smoothed display values ────────────────────────────
        if self._bpm_hist:
            frame.bpm  = round(float(np.median(list(self._bpm_hist))),  1)
        if self._spo2_hist:
            frame.spo2 = round(float(np.median(list(self._spo2_hist))), 1)
        if self._pi_hist:
            frame.pi   = round(float(np.mean(list(self._pi_hist))),     3)

        # ── AI-assisted interpretation ────────────────────────────────────
        frame.alerts = interpret_signal(frame)

        return frame


# ══════════════════════════════════════════════════════════════════════════════
#  §13  TERMINAL DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

# ANSI escape codes (degrade gracefully on non-ANSI terminals)
class ANSI:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GREY   = "\033[90m"
    BG_BLK = "\033[40m"

    @staticmethod
    def supported() -> bool:
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_USE_ANSI = ANSI.supported()

def _c(code: str, text: str) -> str:
    return f"{code}{text}{ANSI.RESET}" if _USE_ANSI else text

def _hr(char: str = "─") -> str:
    return char * CFG.TERM_WIDTH

def _fmt_val(val, fmt: str, unit: str, na: str = "  ---  ") -> str:
    return f"{val:{fmt}}{unit}" if val is not None else na

# SQI label → colour mapping
_SQI_COLOUR = {
    "EXCELLENT": ANSI.GREEN,
    "GOOD":      ANSI.GREEN,
    "FAIR":      ANSI.YELLOW,
    "POOR":      ANSI.RED,
    "NO SIGNAL": ANSI.RED,
}

_SQI_BLOCK = {
    "EXCELLENT": "████████",
    "GOOD":      "██████░░",
    "FAIR":      "████░░░░",
    "POOR":      "██░░░░░░",
    "NO SIGNAL": "░░░░░░░░",
}


def print_header() -> None:
    """Print static session header to terminal."""
    W = CFG.TERM_WIDTH
    print()
    print(_c(ANSI.CYAN + ANSI.BOLD, "═" * W))
    print(_c(ANSI.CYAN + ANSI.BOLD,
             "  MAX30102 OPTICAL BIOSENSING PROTOTYPE".center(W)))
    print(_c(ANSI.GREY,
             "  Research-Grade · Non-Clinical · AI-Assisted".center(W)))
    print(_c(ANSI.CYAN + ANSI.BOLD, "═" * W))
    print(_c(ANSI.GREY, f"  I2C Bus : {CFG.I2C_BUS}  |  "
             f"Addr : 0x{CFG.SENSOR_ADDR:02X}  |  "
             f"SR   : {CFG.SAMPLE_RATE} Hz  |  "
             f"Filter : {'BPF+BL' if SCIPY_AVAILABLE else 'MA'}"))
    print(_c(ANSI.GREY, f"  Window  : {CFG.CALC_WINDOW} smp  |  "
             f"Step : {CFG.STEP_SIZE} smp  |  "
             f"LED  : 0x{CFG.LED_CURRENT:02X}  |  "
             f"ADC  : 18-bit"))
    print(_c(ANSI.CYAN + ANSI.BOLD, "─" * W))
    print(_c(ANSI.GREY,
             "  Place finger firmly and steadily on sensor.  Ctrl+C to exit."))
    print(_c(ANSI.CYAN + ANSI.BOLD, "═" * W))
    print()


_frame_count = 0

def print_frame(frame: VitalFrame) -> None:
    """
    Render one analysis frame to the terminal.

    Layout:
      ┌──────────────────────────────────────────────────────────────────────┐
      │ [HH:MM:SS]  Frame N   FINGER: ●/○    SQI: 87.4  [GOOD] ████████    │
      ├──────────────────────────────────────────────────────────────────────┤
      │  ❤  BPM  :   72.4        🩸 SpO2 :  98.1 %      PI  :  1.234 %    │
      │  DC  IR  : 180432        AC  IR :  2341.0        Noise:  0.021      │
      ├──────────────────────────────────────────────────────────────────────┤
      │  [ALERT / INTERPRETATION LINES]                                      │
      └──────────────────────────────────────────────────────────────────────┘
    """
    global _frame_count
    _frame_count += 1

    ts          = time.strftime("%H:%M:%S")
    W           = CFG.TERM_WIDTH
    sqi_col     = _SQI_COLOUR.get(frame.sqi_label, ANSI.GREY)
    sqi_bar     = _SQI_BLOCK.get(frame.sqi_label,  "░░░░░░░░")
    finger_sym  = (_c(ANSI.GREEN, "● ON ") if frame.finger_on
                   else _c(ANSI.RED,    "○ OFF"))

    bpm_s  = _fmt_val(frame.bpm,  "6.1f", " BPM")
    spo2_s = _fmt_val(frame.spo2, "5.1f", " %  ")
    pi_s   = _fmt_val(frame.pi,   "6.3f", " %  ")

    dc_s    = f"{frame.dc_ir:>9.0f}"
    ac_s    = f"{frame.ac_ir:>9.1f}"
    noise_s = f"{frame.noise_est:6.3f}"

    print(_c(ANSI.GREY, _hr("─")))

    # — Status bar ─────────────────────────────────────────────────────────
    sqi_str = f"SQI: {frame.sqi:5.1f}  "
    sqi_lbl = _c(sqi_col + ANSI.BOLD, f"[{frame.sqi_label:<9}]")
    sqi_bar_c = _c(sqi_col, sqi_bar)
    print(f"  {_c(ANSI.WHITE, ts)}  #{_frame_count:<5}"
          f"  FINGER: {finger_sym}"
          f"    {sqi_str}{sqi_lbl} {sqi_bar_c}")

    # — Vital signs row ─────────────────────────────────────────────────────
    print(f"  {_c(ANSI.RED,   '❤  BPM ')} : "
          f"{_c(ANSI.WHITE + ANSI.BOLD, bpm_s)}"
          f"    {_c(ANSI.CYAN, '🩸 SpO2')} : "
          f"{_c(ANSI.WHITE + ANSI.BOLD, spo2_s)}"
          f"    {_c(ANSI.YELLOW, 'PI')}"
          f" : {_c(ANSI.WHITE + ANSI.BOLD, pi_s)}")

    # — Signal metrics row ──────────────────────────────────────────────────
    cv_s = (f"{frame.peak_cv:.3f}" if frame.peak_cv is not None else " --- ")
    print(f"  {_c(ANSI.GREY, 'DC  IR')} : {_c(ANSI.DIM, dc_s)}"
          f"    {_c(ANSI.GREY, 'AC  IR')} : {_c(ANSI.DIM, ac_s)}"
          f"    {_c(ANSI.GREY, 'Noise')} : {_c(ANSI.DIM, noise_s)}"
          f"    {_c(ANSI.GREY, 'RR-CV')} : {_c(ANSI.DIM, cv_s)}")

    # — AI Interpretation ───────────────────────────────────────────────────
    if frame.alerts:
        print(_c(ANSI.GREY, "  " + "· " * (W // 2 - 1)))
        for alert in frame.alerts[:3]:        # cap at 3 lines
            colour = (ANSI.YELLOW if any(k in alert for k in ("WARN","LOW","HIGH","Elev"))
                      else ANSI.GREY)
            print(f"  {_c(colour, '▸ ' + alert)}")


def print_footer(session_start: float) -> None:
    """Print session summary at exit."""
    elapsed = time.time() - session_start
    print()
    print(_c(ANSI.CYAN + ANSI.BOLD, "═" * CFG.TERM_WIDTH))
    print(_c(ANSI.GREY,
             f"  Session ended  |  Uptime: {elapsed:.0f} s  |  "
             f"Frames: {_frame_count}  |  MAX30102 powered down."))
    print(_c(ANSI.CYAN + ANSI.BOLD, "═" * CFG.TERM_WIDTH))
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  §14  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)

def _warn(msg: str) -> None:
    print(f"[WARN ] {msg}")

def _info(msg: str) -> None:
    print(f"[INFO ] {msg}")


def check_dependencies() -> bool:
    """Verify all required packages are importable before starting."""
    ok = True
    if not SMBUS_AVAILABLE:
        _err("smbus2 not found.  pip install smbus2")
        ok = False
    if not SCIPY_AVAILABLE:
        _warn("scipy not found — reduced filter quality.  pip install scipy")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  §15  MAIN  —  Acquisition / Analysis Loop
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Entry point — orchestrates acquisition, analysis, and display.

    Loop architecture (overlapping window design):
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Cold start: fill ring buffer with CALC_WINDOW samples              │
    │                                                                      │
    │  Then repeat:                                                        │
    │    1. Collect STEP_SIZE new samples → append to ring buffers        │
    │    2. Run AnalysisEngine on latest CALC_WINDOW samples              │
    │    3. Render terminal frame                                          │
    │                                                                      │
    │  This gives 75% window overlap → smoother, lower-latency output.   │
    └──────────────────────────────────────────────────────────────────────┘
    """
    if not check_dependencies():
        sys.exit(1)

    session_start = time.time()
    print_header()

    # ── Open I2C bus ──────────────────────────────────────────────────────
    try:
        bus = smbus2.SMBus(CFG.I2C_BUS)
    except Exception as exc:
        _err(f"Cannot open I2C bus {CFG.I2C_BUS}: {exc}")
        _err("Enable I2C: sudo raspi-config → Interface Options → I2C → Enable")
        sys.exit(1)

    # ── Initialise sensor ─────────────────────────────────────────────────
    _info(f"Connecting to MAX30102 @ 0x{CFG.SENSOR_ADDR:02X} on I2C-{CFG.I2C_BUS} …")
    if not sensor_init(bus):
        bus.close()
        sys.exit(1)
    _info(f"Sensor OK — Part ID=0x15  SR={CFG.SAMPLE_RATE} Hz  18-bit ADC")
    _info("Warming up acquisition buffer …")

    # ── Ring buffers ──────────────────────────────────────────────────────
    ir_ring  = RingBuffer(CFG.RING_SIZE)
    red_ring = RingBuffer(CFG.RING_SIZE)

    # ── Analysis engine ───────────────────────────────────────────────────
    engine = AnalysisEngine()

    # ── Refresh timing ────────────────────────────────────────────────────
    refresh_dt   = 1.0 / CFG.REFRESH_HZ
    last_display = 0.0
    consecutive_errors = 0
    MAX_ERRORS = 10

    # ── Cold start: fill initial analysis window ──────────────────────────
    collect_window(bus, ir_ring, red_ring, CFG.CALC_WINDOW)
    _info("Ready. Monitoring started.\n")

    try:
        while True:
            # 1. Acquire STEP_SIZE new samples (incremental overlap)
            ok = collect_window(bus, ir_ring, red_ring, CFG.STEP_SIZE)

            if not ok:
                consecutive_errors += 1
                if consecutive_errors >= MAX_ERRORS:
                    _err("Persistent I2C read failures — check sensor connection.")
                    break
                time.sleep(0.5)
                continue
            consecutive_errors = 0

            # 2. Run analysis engine
            frame = engine.analyse(ir_ring, red_ring)

            # 3. Display at configured refresh rate
            now = time.time()
            if now - last_display >= refresh_dt:
                print_frame(frame)
                last_display = now

    except KeyboardInterrupt:
        pass   # clean exit

    finally:
        print_footer(session_start)
        try:
            _reg_write(bus, REG.MODE_CONFIG, MODE.SHDN)
            bus.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()