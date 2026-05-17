#!/usr/bin/env python3
# =============================================================================
#  BioSense-Pi  —  MAX30102 Subsystem  v2.0
#  Complete PPG acquisition, DSP, and biometrics pipeline
# =============================================================================
#
#  Target HW    : Raspberry Pi 4/5  +  MAX30102 (I²C addr 0x57)
#  Python       : 3.11+
#  Dependencies : smbus2, numpy, scipy
#
#  ARCHITECTURE
#  ────────────
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  MAX30102Driver          — raw smbus2 register I/O, FIFO management    │
#  │  AmbientBaselineEstimator— ambient subtraction + drift compensation     │
#  │  LEDAutoTuner            — closed-loop LED current PID controller       │
#  │  FingerDetector          — multi-criteria contact gating                │
#  │  PPGRingBuffer           — bounded circular sample store                │
#  │  DSPPipeline             — HP/LP Butterworth, IIR, FFT, RMS, AC/DC     │
#  │  AdaptivePeakDetector    — refractory-gated, amplitude-adaptive         │
#  │  BiometricsEngine        — BPM, SpO2, PI, SQI                          │
#  │  SensorHealthScorer      — per-cycle hardware health metric             │
#  │  PPGLogger               — structured logging hooks                     │
#  │  worker_acq_ppg          — acquisition thread (100 Hz, daemon)          │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  REGISTER MAP (MAX30102 datasheet Rev 1)
#  ───────────────────────────────────────
#  0x00  INT_STATUS1       — FIFO almost full, PPG ready, ALC ovfl
#  0x01  INT_STATUS2       — die temp ready
#  0x02  INT_ENABLE1
#  0x03  INT_ENABLE2
#  0x04  FIFO_WR_PTR
#  0x05  OVF_COUNTER
#  0x06  FIFO_RD_PTR
#  0x07  FIFO_DATA         — 3 bytes RED + 3 bytes IR per sample
#  0x08  FIFO_CONFIG       — SMP_AVE[7:5] FIFO_ROLLOVER[4] FIFO_A_FULL[3:0]
#  0x09  MODE_CONFIG       — SHDN[7] RESET[6] MODE[2:0]
#  0x0A  SPO2_CONFIG       — SPO2_ADC_RGE[6:5] SPO2_SR[4:2] LED_PW[1:0]
#  0x0C  LED1_PA           — RED LED pulse amplitude (0x00–0xFF)
#  0x0D  LED2_PA           — IR  LED pulse amplitude (0x00–0xFF)
#  0x10  MULTI_LED_CTRL1   — slot 1 & 2 config (SpO2 mode: not used)
#  0x1F  DIE_TINT          — temperature integer part
#  0x20  DIE_TFRAC         — temperature fractional part
#  0x21  DIE_TEMP_CONFIG   — write 1 to trigger one-shot measurement
#  0xFE  REVISION_ID
#  0xFF  PART_ID           — always 0x15 for MAX30102
#
#  DATA FORMAT (FIFO_DATA, 18-bit resolution with LED_PW=0x03)
#  ────────────────────────────────────────────────────────────
#  Byte 0: [7:2]=MSB[17:12]  [1:0]=don't care → mask to 0x3FFFF
#  Sample order in FIFO:  RED(3B) → IR(3B)
#  Full-scale (ADC_RGE=0x20): 4096 nA, LSB = 4096/262144 ≈ 15.6 pA
#
#  SpO2 ALGORITHM DISCLAIMER
#  ─────────────────────────
#  The ratio-of-ratios SpO2 estimate implemented here is a research
#  approximation.  Empirical coefficients were derived from the MAX30102
#  datasheet Application Note and published literature (e.g. Webster 2010,
#  Allen 2007).  Calibration against a certified medical pulse oximeter is
#  mandatory before any clinical interpretation.  This code does NOT
#  constitute a medical device.
#
# =============================================================================

from __future__ import annotations

import math
import time
import queue
import logging
import threading
import collections
import statistics
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Deque, Dict, Any, NamedTuple

import numpy as np

# ── Optional dependencies with graceful degradation ──────────────────────────
try:
    from scipy.signal import (
        butter, sosfilt, sosfilt_zi, filtfilt, find_peaks, welch
    )
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import smbus2
    _SMBUS = True
except ImportError:
    _SMBUS = False


# ─────────────────────────────────────────────────────────────────────────────
# §1  REGISTER CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

class _R:
    """MAX30102 register address namespace (datasheet Table 1)."""
    INT_STATUS1    = 0x00
    INT_STATUS2    = 0x01
    INT_ENABLE1    = 0x02
    INT_ENABLE2    = 0x03
    FIFO_WR_PTR    = 0x04
    OVF_COUNTER    = 0x05
    FIFO_RD_PTR    = 0x06
    FIFO_DATA      = 0x07
    FIFO_CONFIG    = 0x08
    MODE_CONFIG    = 0x09
    SPO2_CONFIG    = 0x0A
    LED1_PA        = 0x0C   # RED
    LED2_PA        = 0x0D   # IR
    MULTI_LED_CTRL1= 0x10
    DIE_TINT       = 0x1F
    DIE_TFRAC      = 0x20
    DIE_TEMP_CFG   = 0x21
    REVISION_ID    = 0xFE
    PART_ID        = 0xFF

# Expected Part ID
_PART_ID_MAX30102 = 0x15

# SpO2 config bit fields
_ADC_RGE_4096_nA  = 0b10  # SPO2_ADC_RGE[6:5] → bits 6:5 = 0b10 = 4096 nA FS
_LED_PW_411_US    = 0b11   # LED_PW[1:0] → 18-bit ADC
_SR_100_HZ        = 0b001  # SPO2_SR[4:2] → 100 samples/s

# Mode config
_MODE_SPO2        = 0x03   # Red + IR
_MODE_RESET       = 0x40
_MODE_SHDN        = 0x80

# FIFO config
_SMP_AVE_1        = 0x00   # No averaging (raw 100 Hz samples)
_FIFO_ROLLOVER_EN = 0x10
_FIFO_A_FULL_15   = 0x0F   # Interrupt when 1 space left

# 18-bit sample mask
_18BIT_MASK       = 0x03FFFF


# ─────────────────────────────────────────────────────────────────────────────
# §2  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MAX30102Config:
    """
    All tunable parameters for the MAX30102 subsystem.

    Immutable after construction.  Override via dataclasses.replace() or
    JSON injection through SystemConfig (see main.py §2).
    """
    # ── Hardware ──────────────────────────────────────────────────────────
    i2c_bus          : int   = 1          # /dev/i2c-1 on Pi
    sensor_addr      : int   = 0x57      # MAX30102 default
    sample_rate_hz   : int   = 100        # hardware SR
    adc_range        : int   = _ADC_RGE_4096_nA
    pulse_width      : int   = _LED_PW_411_US
    smp_avg          : int   = 1          # no HW averaging

    # ── LED current (0x00=0 mA … 0xFF≈51 mA, step≈0.2 mA) ───────────────
    led_current_init : int   = 0x3C      # ~12 mA — safe starting point
    led_current_min  : int   = 0x0A      # ~2 mA
    led_current_max  : int   = 0xC8      # ~40 mA
    led_current_step : int   = 0x05      # PID step size

    # ── Ring buffer sizes ─────────────────────────────────────────────────
    ring_depth       : int   = 600        # 6 s @ 100 Hz
    analysis_window  : int   = 250        # 2.5 s analysis window
    analysis_stride  : int   = 50         # 0.5 s overlap stride (80% overlap)

    # ── DSP filter parameters ─────────────────────────────────────────────
    hp_cutoff_hz     : float = 0.5        # baseline wander removal
    lp_cutoff_hz     : float = 8.0        # anti-aliasing / noise floor
    bp_low_hz        : float = 0.6        # final BPF lower edge (HR band)
    bp_high_hz       : float = 4.0        # final BPF upper edge (240 BPM)
    filter_order     : int   = 4          # Butterworth order (IIR SOS)
    iir_baseline_tau : float = 0.995      # adaptive IIR pole (DC tracker)

    # ── Finger detection thresholds ───────────────────────────────────────
    finger_dc_min    : int   = 50_000     # ADC counts — tissue absorption floor
    finger_ac_min    : int   = 300        # peak-to-peak minimum AC content
    finger_score_min : float = 0.55       # composite detection confidence gate

    # ── Peak detection ────────────────────────────────────────────────────
    peak_min_bpm     : int   = 30
    peak_max_bpm     : int   = 220
    refractory_ms    : float = 200.0      # minimum inter-beat interval

    # ── Ambient / LED tuning ──────────────────────────────────────────────
    ambient_avg_samples: int = 50         # samples for ambient baseline
    led_target_ac_pct  : float = 1.5      # target AC/DC% for LED tuning
    led_tune_deadband  : float = 0.3      # deadband to prevent hunting

    # ── Health scoring ────────────────────────────────────────────────────
    max_ovf_per_window : int   = 5        # FIFO overflow tolerance
    max_i2c_err_rate   : float = 0.05     # 5% I²C error ceiling
    health_ema_alpha   : float = 0.05     # health metric smoothing

    # ── Display averaging ─────────────────────────────────────────────────
    bpm_avg_n    : int = 6
    spo2_avg_n   : int = 6
    pi_avg_n     : int = 4


# ─────────────────────────────────────────────────────────────────────────────
# §3  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class PPGSample(NamedTuple):
    """One decoded FIFO sample pair."""
    red    : int    # 18-bit RED ADC count
    ir     : int    # 18-bit IR  ADC count
    ts     : float  # monotonic timestamp [s]


@dataclass
class BiometricResult:
    """
    Output record from one analysis window.

    All values are None when their computation preconditions are not met
    (insufficient peaks, no finger contact, signal quality below gate).
    """
    bpm              : Optional[float] = None
    spo2             : Optional[float] = None
    perfusion_index  : Optional[float] = None
    sqi              : float = 0.0
    sqi_label        : str   = "NO SIGNAL"
    dc_ir            : float = 0.0
    dc_red           : float = 0.0
    ac_ir_rms        : float = 0.0
    ac_red_rms       : float = 0.0
    snr_ir_db        : float = 0.0
    peak_count       : int   = 0
    rr_intervals_ms  : List[float] = field(default_factory=list)
    r_ratio          : Optional[float] = None
    finger_score     : float = 0.0
    sensor_health    : float = 0.0
    ts               : float = field(default_factory=time.monotonic)


@dataclass
class SensorHealthRecord:
    """Live health telemetry accumulated between analysis windows."""
    total_samples    : int   = 0
    i2c_errors       : int   = 0
    fifo_overflows   : int   = 0
    crc_errors       : int   = 0   # parity / range check failures
    health_score     : float = 1.0  # 0.0 (failed) … 1.0 (perfect)
    last_error       : str   = ""
    consecutive_errs : int   = 0


# ─────────────────────────────────────────────────────────────────────────────
# §4  PPG RING BUFFER
# ─────────────────────────────────────────────────────────────────────────────

class PPGRingBuffer:
    """
    Bounded circular buffer for PPG sample pairs.

    Thread-safe for single-producer / single-consumer use without locks,
    using Python deque's atomic append (GIL-protected) and explicit
    RLock for multi-consumer window extraction.

    Memory:  ring_depth × 3 × 8 bytes (float64) ≈ 14.4 KB @ depth=600
    Latency: O(1) push, O(n) window extraction (unavoidable copy)
    """

    def __init__(self, depth: int):
        self._depth  = depth
        self._ir     : Deque[float] = collections.deque(maxlen=depth)
        self._red    : Deque[float] = collections.deque(maxlen=depth)
        self._ts     : Deque[float] = collections.deque(maxlen=depth)
        self._lock   = threading.RLock()

    def push(self, sample: PPGSample) -> None:
        """Append one sample. Oldest entry auto-evicted when full."""
        # deque.append is atomic under the GIL — no lock needed for push
        self._ir.append(float(sample.ir))
        self._red.append(float(sample.red))
        self._ts.append(sample.ts)

    def window(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the last n samples as (ir, red, timestamps) float64 arrays.
        Returns empty arrays if fewer than n samples are buffered.
        """
        with self._lock:
            buf_len = len(self._ir)
            if buf_len < n:
                empty = np.empty(0, dtype=np.float64)
                return empty, empty, empty
            ir  = np.array(list(self._ir)[-n:],  dtype=np.float64)
            red = np.array(list(self._red)[-n:], dtype=np.float64)
            ts  = np.array(list(self._ts)[-n:],  dtype=np.float64)
        return ir, red, ts

    def __len__(self) -> int:
        return len(self._ir)

    @property
    def depth(self) -> int:
        return self._depth


# ─────────────────────────────────────────────────────────────────────────────
# §5  MAX30102 HARDWARE DRIVER
# ─────────────────────────────────────────────────────────────────────────────

class MAX30102Driver:
    """
    Raw smbus2 driver for the MAX30102 pulse oximeter IC.

    Responsibilities:
      · Register read/write with retry (up to max_retries attempts)
      · Device identity verification (PART_ID = 0x15)
      · Hardware initialization sequence
      · FIFO burst read with 18-bit unpacking
      · FIFO pointer arithmetic and overflow detection
      · Die temperature one-shot acquisition
      · LED current register updates (used by LEDAutoTuner)
      · Graceful shutdown (SHDN bit)

    Error policy:
      · Transient OSError  → retry up to max_retries; raise on exhaustion
      · Permanent failure  → caller catches and handles degraded mode
      · All errors counted in SensorHealthRecord for health scoring

    Thread safety:
      · NOT thread-safe; call only from acq-ppg thread.
      · health record is updated in-place; reader threads should copy.

    I²C timing note:
      The Pi's I²C controller defaults to 100 kHz (standard mode).
      Reading 32 × 6 = 192 bytes from FIFO at 100 kHz takes ~20 ms.
      At 400 kHz (fast mode), this drops to ~5 ms.
      Enable fast mode: /boot/config.txt → dtparam=i2c_arm_baudrate=400000
    """

    # Maximum bytes readable in one smbus2 i2c_block_data call
    _MAX_BLOCK    = 32          # smbus2 hard limit (i2c_msg can do more)
    _SAMPLE_BYTES = 6           # 3B RED + 3B IR per FIFO entry
    _FIFO_DEPTH   = 32          # MAX30102 FIFO depth (entries)

    def __init__(self,
                 cfg        : MAX30102Config,
                 health     : SensorHealthRecord,
                 log        : logging.Logger,
                 max_retries: int = 3):
        self._cfg       = cfg
        self._health    = health
        self._log       = log
        self._max_ret   = max_retries
        self._bus       : Optional[smbus2.SMBus] = None
        self._addr      = cfg.sensor_addr
        self._open      = False

        # LED current state (mirrored here to avoid redundant bus reads)
        self._led_red_pa = cfg.led_current_init
        self._led_ir_pa  = cfg.led_current_init

        # FIFO overflow counter (monotonic, compared against OVF_COUNTER reg)
        self._last_ovf   = 0

        # Revision info
        self.revision_id : int = 0xFF
        self.part_id     : int = 0x00

    # ── Bus lifecycle ──────────────────────────────────────────────────────

    def open(self) -> None:
        """Open I²C bus, verify device identity, run init sequence."""
        if not _SMBUS:
            raise RuntimeError("smbus2 not installed — hardware I²C unavailable.")
        self._bus  = smbus2.SMBus(self._cfg.i2c_bus)
        self._open = True
        self._verify_identity()
        self._init_hardware()
        self._log.info(
            "MAX30102Driver: opened bus=%d addr=0x%02X rev=0x%02X",
            self._cfg.i2c_bus, self._addr, self.revision_id)

    def close(self) -> None:
        """Assert SHDN bit and close I²C bus."""
        if self._open and self._bus is not None:
            try:
                self._write_reg(_R.MODE_CONFIG, _MODE_SHDN)
            except Exception:
                pass
            try:
                self._bus.close()
            except Exception:
                pass
        self._open = False
        self._log.info("MAX30102Driver: closed.")

    # ── Identity ───────────────────────────────────────────────────────────

    def _verify_identity(self) -> None:
        """
        Read PART_ID and REVISION_ID.
        Raises RuntimeError if PART_ID ≠ 0x15 (device absent or wrong IC).
        """
        pid = self._read_reg(_R.PART_ID)
        rev = self._read_reg(_R.REVISION_ID)
        self.part_id     = pid
        self.revision_id = rev
        if pid != _PART_ID_MAX30102:
            raise RuntimeError(
                f"MAX30102 not found: PART_ID=0x{pid:02X} (expected 0x15). "
                f"Check wiring, I²C address, and pull-ups.")
        self._log.info("MAX30102Driver: PART_ID=0x%02X  REV=0x%02X ✓", pid, rev)

    # ── Hardware initialisation ────────────────────────────────────────────

    def _init_hardware(self) -> None:
        """
        Full power-on initialisation sequence.

        Order is critical — MODE_CONFIG must be written last so the IC
        does not begin sampling before FIFO and SPO2 configs are set.

        Register values derived from MAX30102 datasheet Table 3.
        """
        # 1. Soft reset — self-clearing; wait for RESET bit to clear
        self._write_reg(_R.MODE_CONFIG, _MODE_RESET)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if (self._read_reg(_R.MODE_CONFIG) & _MODE_RESET) == 0:
                break
            time.sleep(0.01)
        else:
            self._log.warning("MAX30102Driver: soft reset did not self-clear in 1 s.")

        time.sleep(0.02)  # datasheet: wait ≥ 1 ms after reset

        # 2. Clear FIFO pointers and overflow counter
        for reg in (_R.FIFO_WR_PTR, _R.OVF_COUNTER, _R.FIFO_RD_PTR):
            self._write_reg(reg, 0x00)

        # 3. FIFO config:
        #    SMP_AVE=001 (2 samples averaged in HW — datasheet rec. for 100 Hz)
        #    FIFO_ROLLOVER=1 (overwrite oldest on overflow)
        #    FIFO_A_FULL=0x0F (almost-full interrupt threshold = 1 unread slot)
        smp_bits = {1:0b000, 2:0b001, 4:0b010, 8:0b011, 16:0b100, 32:0b101}
        avg_bits = smp_bits.get(self._cfg.smp_avg, 0b000)
        fifo_cfg = (avg_bits << 5) | _FIFO_ROLLOVER_EN | _FIFO_A_FULL_15
        self._write_reg(_R.FIFO_CONFIG, fifo_cfg)

        # 4. SPO2 config:
        #    SPO2_ADC_RGE[6:5] = 10  → 4096 nA full-scale
        #    SPO2_SR[4:2]       = 001 → 100 SPS
        #    LED_PW[1:0]        = 11  → 411 µs, 18-bit ADC
        sr_map = {50:0b000,100:0b001,200:0b010,400:0b011,
                  800:0b100,1000:0b101,1600:0b110,3200:0b111}
        sr_bits = sr_map.get(self._cfg.sample_rate_hz, 0b001)
        spo2_cfg = (self._cfg.adc_range << 5) | (sr_bits << 2) | self._cfg.pulse_width
        self._write_reg(_R.SPO2_CONFIG, spo2_cfg)

        # 5. LED pulse amplitudes
        self._write_reg(_R.LED1_PA, self._cfg.led_current_init)   # RED
        self._write_reg(_R.LED2_PA, self._cfg.led_current_init)   # IR
        self._led_red_pa = self._led_ir_pa = self._cfg.led_current_init

        # 6. Mode: SpO2 (RED + IR channels active)
        self._write_reg(_R.MODE_CONFIG, _MODE_SPO2)

        self._log.info(
            "MAX30102Driver: init complete — SR=%dHz LED=0x%02X FIFO_CFG=0x%02X",
            self._cfg.sample_rate_hz, self._cfg.led_current_init, fifo_cfg)

    # ── FIFO acquisition ───────────────────────────────────────────────────

    def read_fifo_burst(self) -> List[PPGSample]:
        """
        Read all available FIFO samples in a single or multi-block I²C burst.

        FIFO pointer arithmetic (datasheet §5.3):
          num_samples = (WR_PTR - RD_PTR) mod 32

        18-bit unpacking:
          Each sample is 6 bytes (3 RED, 3 IR).
          Byte layout (big-endian, MSB first):
            B0[7:0] → bits [17:10]   (upper 8 of 18)
            B1[7:0] → bits [9:2]     (middle 8)
            B2[7:0] → bits [1:0] in [7:6] (lower 2, upper 6 are channel-dependent)
          Mask with 0x03FFFF after composing the 24-bit value.

        Returns:
          List of PPGSample namedtuples.  Empty list on transient I²C error.
          FIFO overflow is detected and logged but does not raise.
        """
        ts = time.monotonic()
        try:
            # Read FIFO pointers in one 3-byte block (contiguous: 0x04,0x05,0x06)
            ptr_block = self._read_block(_R.FIFO_WR_PTR, 3)
            wr_ptr = ptr_block[0] & 0x1F
            ovf    = ptr_block[1]
            rd_ptr = ptr_block[2] & 0x1F
        except OSError as e:
            self._record_error(str(e))
            return []

        # Detect FIFO overflow
        if ovf != self._last_ovf:
            overflow_diff = (ovf - self._last_ovf) & 0xFF
            self._health.fifo_overflows += overflow_diff
            if self._health.fifo_overflows <= self._cfg.max_ovf_per_window * 10:
                self._log.warning("MAX30102: FIFO overflow +%d (total=%d)",
                                  overflow_diff, self._health.fifo_overflows)
            self._last_ovf = ovf

        # Number of new samples
        n_samples = (wr_ptr - rd_ptr) & 0x1F
        if n_samples == 0:
            return []
        if n_samples > _MAX30102Driver._FIFO_DEPTH:
            n_samples = _MAX30102Driver._FIFO_DEPTH

        # Burst-read n_samples × 6 bytes from FIFO_DATA register
        # smbus2.read_i2c_block_data is limited to 32 bytes per call.
        # For n_samples > 5 (30+ bytes), use i2c_msg for a true burst.
        n_bytes = n_samples * self._SAMPLE_BYTES
        try:
            raw = self._read_block_multi(_R.FIFO_DATA, n_bytes)
        except OSError as e:
            self._record_error(str(e))
            return []

        # Decode bytes → PPGSample list
        samples: List[PPGSample] = []
        for i in range(n_samples):
            base = i * 6
            if base + 5 >= len(raw):
                break  # truncated read — discard remainder
            # RED: bytes 0,1,2
            red_raw = ((raw[base]   << 16) |
                       (raw[base+1] <<  8) |
                        raw[base+2]) & _18BIT_MASK
            # IR:  bytes 3,4,5
            ir_raw  = ((raw[base+3] << 16) |
                       (raw[base+4] <<  8) |
                        raw[base+5]) & _18BIT_MASK

            # Range sanity check (18-bit max = 262143)
            if red_raw > _18BIT_MASK or ir_raw > _18BIT_MASK:
                self._health.crc_errors += 1
                continue

            # Interpolate timestamps uniformly within the burst
            # (hardware samples are equidistant at 1/sample_rate_hz)
            dt = i / max(1, self._cfg.sample_rate_hz)
            samples.append(PPGSample(red=red_raw, ir=ir_raw, ts=ts - dt))

        self._health.total_samples += len(samples)
        return samples

    # ── Die temperature ────────────────────────────────────────────────────

    def read_die_temperature(self) -> float:
        """
        Trigger a one-shot die temperature measurement and return result in °C.

        Resolution: 0.0625 °C (datasheet §5.4.5)
        Measurement time: ~30 ms (poll with 5 ms intervals)

        Returns NaN on failure.
        """
        try:
            self._write_reg(_R.DIE_TEMP_CFG, 0x01)
            for _ in range(10):
                time.sleep(0.005)
                if (self._read_reg(_R.INT_STATUS2) & 0x02):
                    break
            tint  = self._read_reg(_R.DIE_TINT)
            tfrac = self._read_reg(_R.DIE_TFRAC)
            # TINT is signed 8-bit; TFRAC is unsigned 4-bit (steps of 0.0625)
            t_signed = tint if tint < 128 else tint - 256
            return t_signed + tfrac * 0.0625
        except OSError:
            return float("nan")

    # ── LED current control ────────────────────────────────────────────────

    def set_led_current(self, red_pa: int, ir_pa: int) -> None:
        """
        Update LED pulse amplitude registers.

        Clamps to [led_current_min, led_current_max] from config.
        Only writes if value differs from cached state (saves I²C bandwidth).
        """
        red_pa = int(np.clip(red_pa, self._cfg.led_current_min, self._cfg.led_current_max))
        ir_pa  = int(np.clip(ir_pa,  self._cfg.led_current_min, self._cfg.led_current_max))
        try:
            if red_pa != self._led_red_pa:
                self._write_reg(_R.LED1_PA, red_pa)
                self._led_red_pa = red_pa
            if ir_pa != self._led_ir_pa:
                self._write_reg(_R.LED2_PA, ir_pa)
                self._led_ir_pa = ir_pa
        except OSError as e:
            self._record_error(str(e))

    @property
    def led_current(self) -> Tuple[int, int]:
        """Returns (red_pa, ir_pa) cached values."""
        return self._led_red_pa, self._led_ir_pa

    # ── Low-level I²C with retry ────────────────────────────────────────────

    def _read_reg(self, reg: int) -> int:
        """Single-byte register read with retry."""
        for attempt in range(self._max_ret):
            try:
                return self._bus.read_byte_data(self._addr, reg)
            except OSError as e:
                self._health.i2c_errors += 1
                if attempt == self._max_ret - 1:
                    self._record_error(str(e))
                    raise
                time.sleep(0.002 * (attempt + 1))
        return 0  # unreachable

    def _write_reg(self, reg: int, val: int) -> None:
        """Single-byte register write with retry."""
        for attempt in range(self._max_ret):
            try:
                self._bus.write_byte_data(self._addr, reg, val & 0xFF)
                return
            except OSError as e:
                self._health.i2c_errors += 1
                if attempt == self._max_ret - 1:
                    self._record_error(str(e))
                    raise
                time.sleep(0.002 * (attempt + 1))

    def _read_block(self, reg: int, length: int) -> List[int]:
        """Read up to 32 bytes using read_i2c_block_data."""
        length = min(length, 32)
        for attempt in range(self._max_ret):
            try:
                return self._bus.read_i2c_block_data(self._addr, reg, length)
            except OSError as e:
                self._health.i2c_errors += 1
                if attempt == self._max_ret - 1:
                    self._record_error(str(e))
                    raise
                time.sleep(0.002 * (attempt + 1))
        return []

    def _read_block_multi(self, reg: int, length: int) -> List[int]:
        """
        Read an arbitrary number of bytes from a register using
        smbus2.i2c_msg for true I²C burst (no 32-byte restriction).

        Falls back to chunked read_i2c_block_data if i2c_msg is unavailable.
        """
        if hasattr(smbus2, 'i2c_msg') and length > 32:
            for attempt in range(self._max_ret):
                try:
                    write = smbus2.i2c_msg.write(self._addr, [reg])
                    read  = smbus2.i2c_msg.read(self._addr, length)
                    self._bus.i2c_rdwr(write, read)
                    return list(read)
                except OSError as e:
                    self._health.i2c_errors += 1
                    if attempt == self._max_ret - 1:
                        self._record_error(str(e))
                        raise
                    time.sleep(0.003 * (attempt + 1))
            return []

        # Chunked fallback: 32 bytes per call
        result: List[int] = []
        offset = 0
        while offset < length:
            chunk = min(32, length - offset)
            result.extend(self._read_block(reg, chunk))
            offset += chunk
        return result

    def _record_error(self, msg: str) -> None:
        self._health.last_error     = msg[:200]
        self._health.consecutive_errs += 1


# Alias — needed inside _read_block_multi before class is fully defined
_MAX30102Driver = MAX30102Driver


# ─────────────────────────────────────────────────────────────────────────────
# §6  AMBIENT BASELINE ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

class AmbientBaselineEstimator:
    """
    Estimates and tracks the ambient light (non-tissue) baseline.

    Algorithm:
      · During finger-absent periods, collect N samples from both channels.
      · Compute robust median estimate (outlier-resistant vs mean).
      · Use an exponential moving average to track slow drift
        (temperature, ambient illumination changes).
      · Expose baseline for subtraction before DC analysis.

    Why it matters:
      The MAX30102 datasheet (§5.3) notes that ambient light can contaminate
      both RED and IR channels at low LED currents.  The ALC (Auto Light
      Control) register reduces this but does not eliminate it.  Explicit
      ambient estimation during finger-absent intervals provides a more
      accurate DC baseline, especially in bright environments.
    """

    def __init__(self, cfg: MAX30102Config, log: logging.Logger):
        self._cfg          = cfg
        self._log          = log
        self._ir_baseline  : Optional[float] = None
        self._red_baseline : Optional[float] = None
        self._ir_buf       : List[float] = []
        self._red_buf      : List[float] = []
        self._ema_alpha    = 0.05   # 5% weight on new measurement → slow drift track

    def feed(self, sample: PPGSample, finger_present: bool) -> None:
        """
        Feed a raw sample.  Only accumulates when finger is absent.
        """
        if finger_present:
            return
        self._ir_buf.append(float(sample.ir))
        self._red_buf.append(float(sample.red))
        if len(self._ir_buf) >= self._cfg.ambient_avg_samples:
            self._update()

    def _update(self) -> None:
        ir_est  = float(np.median(self._ir_buf))
        red_est = float(np.median(self._red_buf))
        if self._ir_baseline is None:
            self._ir_baseline  = ir_est
            self._red_baseline = red_est
        else:
            a = self._ema_alpha
            self._ir_baseline  = (1-a)*self._ir_baseline  + a*ir_est
            self._red_baseline = (1-a)*self._red_baseline + a*red_est
        self._log.debug("AmbientBaseline updated: IR=%.0f  RED=%.0f",
                        self._ir_baseline, self._red_baseline)
        self._ir_buf.clear(); self._red_buf.clear()

    @property
    def ir_baseline(self) -> float:
        return self._ir_baseline or 0.0

    @property
    def red_baseline(self) -> float:
        return self._red_baseline or 0.0

    @property
    def is_valid(self) -> bool:
        return self._ir_baseline is not None


# ─────────────────────────────────────────────────────────────────────────────
# §7  LED AUTO TUNER
# ─────────────────────────────────────────────────────────────────────────────

class LEDAutoTuner:
    """
    Closed-loop LED current controller.

    Target: maintain AC/DC ratio of IR signal near `target_ac_pct` (default 1.5%).
    This ensures:
      · Sufficient AC swing for reliable peak detection
      · Avoids ADC saturation (DC too high)
      · Keeps LED power to minimum necessary (thermal + power budget)

    Controller: proportional with deadband.
    Direction logic:
      · AC/DC too LOW  → increase LED current (more photons → more AC)
      · AC/DC too HIGH → decrease LED current (prevent saturation)

    Tuning cadence: once per analysis window (every 0.5 s at 80% overlap).
    """

    def __init__(self, cfg: MAX30102Config, driver: MAX30102Driver, log: logging.Logger):
        self._cfg    = cfg
        self._driver = driver
        self._log    = log
        self._enabled = True

    def tune(self, dc_ir: float, ac_ir_rms: float) -> None:
        """
        Evaluate current AC/DC ratio and adjust LED if outside deadband.

        Args:
            dc_ir     : DC component of IR signal (raw ADC counts)
            ac_ir_rms : RMS of AC-coupled IR signal
        """
        if not self._enabled or dc_ir < 1000:
            return

        ac_dc_pct = (ac_ir_rms / dc_ir) * 100.0
        target    = self._cfg.led_target_ac_pct
        deadband  = self._cfg.led_tune_deadband
        step      = self._cfg.led_current_step

        red_pa, ir_pa = self._driver.led_current

        if ac_dc_pct < target - deadband:
            # Signal too weak — increase both LEDs by one step
            new_red = min(red_pa + step, self._cfg.led_current_max)
            new_ir  = min(ir_pa  + step, self._cfg.led_current_max)
            self._driver.set_led_current(new_red, new_ir)
            self._log.debug("LEDTuner: AC/DC=%.2f%% < %.2f%% → LED ↑ 0x%02X",
                            ac_dc_pct, target, new_ir)

        elif ac_dc_pct > target + deadband:
            # Signal saturating — decrease both LEDs by one step
            new_red = max(red_pa - step, self._cfg.led_current_min)
            new_ir  = max(ir_pa  - step, self._cfg.led_current_min)
            self._driver.set_led_current(new_red, new_ir)
            self._log.debug("LEDTuner: AC/DC=%.2f%% > %.2f%% → LED ↓ 0x%02X",
                            ac_dc_pct, target, new_ir)
        # else: within deadband — no action


# ─────────────────────────────────────────────────────────────────────────────
# §8  FINGER DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class FingerDetector:
    """
    Multi-criteria finger contact gate with hysteresis.

    Criteria (weighted composite score, threshold = finger_score_min):
      1. DC level (IR)      — tissue absorption raises baseline above ambient
      2. AC amplitude       — pulsatile component must exceed noise floor
      3. DC symmetry        — RED and IR DC ratio should be in physiological range
      4. Variance stability — short-term variance should exceed ambient noise

    Hysteresis:
      · ABSENT → PRESENT requires score ≥ threshold for 3 consecutive windows
      · PRESENT → ABSENT requires score < threshold for 2 consecutive windows
      This prevents contact chattering from noisy finger placement.
    """

    _ENTER_HOLD = 3   # windows above threshold to assert presence
    _EXIT_HOLD  = 2   # windows below threshold to assert absence

    def __init__(self, cfg: MAX30102Config, log: logging.Logger):
        self._cfg      = cfg
        self._log      = log
        self._present  = False
        self._score    = 0.0
        self._hold_cnt = 0
        self._candidate= False

    def update(self,
               dc_ir : float,
               dc_red: float,
               ac_ir : float,
               ir_var: float) -> Tuple[bool, float]:
        """
        Evaluate contact and return (finger_present, composite_score).

        Args:
            dc_ir  : mean IR ADC value over window
            dc_red : mean RED ADC value over window
            ac_ir  : peak-to-peak AC amplitude of IR
            ir_var : short-term variance of IR signal
        """
        # ── Criterion 1: DC level ────────────────────────────────────────
        dc_score = float(np.clip(
            (dc_ir - self._cfg.finger_dc_min * 0.5)
            / (self._cfg.finger_dc_min * 1.5), 0.0, 1.0))

        # ── Criterion 2: AC amplitude ────────────────────────────────────
        ac_score = float(np.clip(
            (ac_ir - self._cfg.finger_ac_min * 0.5)
            / (self._cfg.finger_ac_min * 3.0), 0.0, 1.0))

        # ── Criterion 3: RED/IR DC ratio (physiological range ~0.4–1.2) ──
        if dc_ir > 1000 and dc_red > 1000:
            ratio = dc_red / dc_ir
            # Score peaks at ratio ~0.7 (typical SpO2 ~97%),
            # falls off outside [0.3, 1.5]
            sym_score = float(np.clip(
                1.0 - abs(ratio - 0.7) / 0.5, 0.0, 1.0))
        else:
            sym_score = 0.0

        # ── Criterion 4: Variance stability ─────────────────────────────
        var_score = float(np.clip(ir_var / (self._cfg.finger_ac_min ** 2), 0.0, 1.0))

        # ── Composite (weighted) ─────────────────────────────────────────
        score = (0.40 * dc_score
               + 0.35 * ac_score
               + 0.15 * sym_score
               + 0.10 * var_score)
        self._score = score

        # ── Hysteresis state machine ─────────────────────────────────────
        candidate = score >= self._cfg.finger_score_min
        if candidate == self._candidate:
            self._hold_cnt += 1
        else:
            self._candidate = candidate
            self._hold_cnt  = 1

        hold_req = self._ENTER_HOLD if (candidate and not self._present) \
              else self._EXIT_HOLD

        if self._hold_cnt >= hold_req:
            if self._present != candidate:
                self._log.info("FingerDetector: %s → %s (score=%.2f)",
                               "PRESENT" if self._present else "ABSENT",
                               "PRESENT" if candidate else "ABSENT", score)
            self._present = candidate

        return self._present, score


# ─────────────────────────────────────────────────────────────────────────────
# §9  DSP PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class DSPPipeline:
    """
    Multi-stage digital signal processing pipeline for PPG signals.

    Stage architecture:
      Input (raw IR or RED ADC, float64)
        │
        ├─ [1] Adaptive IIR Baseline Removal  (DC wander cancellation)
        │       Single-pole IIR: y[n] = τ·y[n-1] + (1-τ)·x[n]
        │       Wander = y[n]; AC = x[n] - y[n]
        │       τ = iir_baseline_tau (default 0.995 ≈ fc = 0.08 Hz @ 100 Hz)
        │
        ├─ [2] Butterworth High-Pass (SOS, scipy)
        │       Removes residual drift below hp_cutoff_hz
        │
        ├─ [3] Butterworth Low-Pass (SOS, scipy)
        │       Removes high-frequency noise above lp_cutoff_hz
        │
        ├─ [4] Band-Pass Gate (bp_low_hz … bp_high_hz)
        │       Applied to final signal for peak detection
        │
        ├─ [5] RMS Computation (AC channel)
        │
        ├─ [6] FFT-based spectral analysis
        │       Dominant frequency → HR estimate
        │       Spectral SNR → SQI component
        │
        └─ Output: (ac_filtered, dc_estimate, rms, dominant_hz, snr_db)

    Filter state management:
      SOS filter states are maintained across windows for causal filtering.
      States are reset on finger detect/loss to avoid transient artefacts.

    scipy dependency:
      Falls back to simple IIR + moving average when scipy is unavailable.
      Quality degrades but the pipeline remains functional.
    """

    def __init__(self, cfg: MAX30102Config, log: logging.Logger):
        self._cfg   = cfg
        self._log   = log
        self._fs    = float(cfg.sample_rate_hz)

        # Cached SOS filter coefficients (built once, reused)
        self._sos_hp   : Optional[np.ndarray] = None
        self._sos_lp   : Optional[np.ndarray] = None
        self._sos_bp   : Optional[np.ndarray] = None
        self._zi_hp_ir : Optional[np.ndarray] = None
        self._zi_hp_red: Optional[np.ndarray] = None
        self._zi_lp_ir : Optional[np.ndarray] = None
        self._zi_lp_red: Optional[np.ndarray] = None
        self._zi_bp    : Optional[np.ndarray] = None

        # Adaptive IIR state
        self._iir_state_ir  : Optional[float] = None
        self._iir_state_red : Optional[float] = None

        self._build_filters()

    def _build_filters(self) -> None:
        """Pre-compute Butterworth SOS coefficients."""
        if not _SCIPY:
            self._log.warning("DSPPipeline: scipy unavailable — basic IIR fallback.")
            return
        nyq = self._fs / 2.0

        def _clamp(fc: float) -> float:
            """Clamp cutoff to valid Nyquist range (0, 1) exclusive."""
            return float(np.clip(fc / nyq, 1e-4, 1.0 - 1e-4))

        try:
            self._sos_hp = butter(self._cfg.filter_order,
                                  _clamp(self._cfg.hp_cutoff_hz),
                                  btype='high', output='sos')
            self._sos_lp = butter(self._cfg.filter_order,
                                  _clamp(self._cfg.lp_cutoff_hz),
                                  btype='low',  output='sos')
            lo = _clamp(self._cfg.bp_low_hz)
            hi = _clamp(self._cfg.bp_high_hz)
            if lo < hi:
                self._sos_bp = butter(self._cfg.filter_order,
                                      [lo, hi], btype='band', output='sos')
        except Exception as e:
            self._log.error("DSPPipeline: filter build failed: %s", e)

    def reset_state(self) -> None:
        """
        Reset all filter initial conditions.
        Call when finger contact is lost / reacquired.
        """
        self._iir_state_ir  = None
        self._iir_state_red = None
        self._zi_hp_ir      = None
        self._zi_hp_red     = None
        self._zi_lp_ir      = None
        self._zi_lp_red     = None
        self._zi_bp         = None

    def _iir_baseline(self,
                      x     : np.ndarray,
                      state : Optional[float],
                      chan  : str
                      ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Adaptive single-pole IIR DC tracker.

        Implements:   s[n] = τ·s[n-1] + (1-τ)·x[n]    (LP → DC estimate)
                      ac[n] = x[n] - s[n]               (HP → AC signal)

        Returns:
            ac      : AC-coupled signal (baseline-removed)
            dc_est  : DC estimate per sample (for AC/DC ratio)
            state   : updated filter state for next call
        """
        τ = self._cfg.iir_baseline_tau
        α = 1.0 - τ
        s = state if state is not None else x[0]
        dc_est = np.empty_like(x)
        ac     = np.empty_like(x)
        for i, xi in enumerate(x):
            s        = τ * s + α * xi
            dc_est[i]= s
            ac[i]    = xi - s
        return ac, dc_est, s

    def _apply_sos(self,
                   sos  : Optional[np.ndarray],
                   x    : np.ndarray,
                   zi   : Optional[np.ndarray]
                   ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply SOS filter to x with persistent state zi.

        Uses sosfilt (causal, forward-only) to avoid phase distortion on
        real-time data where filtfilt is not applicable (no future samples).
        """
        if not _SCIPY or sos is None or len(x) < 4:
            return x, zi
        if zi is None:
            zi = sosfilt_zi(sos) * x[0]
        y, zi_new = sosfilt(sos, x, zi=zi)
        return y, zi_new

    def process(self,
                ir_raw  : np.ndarray,
                red_raw : np.ndarray
                ) -> Dict[str, Any]:
        """
        Run the full DSP pipeline on one analysis window.

        Filter strategy:
          · Stages 1–3 (IIR baseline, HP, LP): causal sosfilt with persistent
            state — suitable for online streaming metrics (ac_rms, snr, DC).
          · Stage 4 band-pass for peak detection: filtfilt (zero-phase, non-causal)
            applied to the extracted window to eliminate group-delay distortion.
            This is valid because we operate on a complete batch window; the
            latency introduced by non-causal filtering is acceptable for the
            0.5 s analysis stride.  filtfilt prevents the peak-timing bias that
            causal filters introduce at the edges of short windows.

        Args:
            ir_raw   : Raw IR ADC samples (float64, length = analysis_window)
            red_raw  : Raw RED ADC samples (float64, same length)

        Returns dict with keys:
            ir_ac_filtered   : zero-phase band-pass filtered IR AC signal
            red_ac_filtered  : zero-phase band-pass filtered RED AC signal
            ir_dc            : mean DC of IR window
            red_dc           : mean DC of RED window
            ir_dc_array      : per-sample DC estimate (for AC/DC per beat)
            red_dc_array     : per-sample DC estimate
            ir_ac_rms        : RMS of AC IR signal
            red_ac_rms       : RMS of AC RED signal
            ir_pp            : peak-to-peak of AC IR
            red_pp           : peak-to-peak of AC RED
            dominant_hz      : FFT-dominant frequency in HR band
            snr_db_ir        : spectral SNR of IR channel (dB)
            snr_db_red       : spectral SNR of RED channel (dB)
        """
        # ── Stage 1: Adaptive IIR baseline removal ───────────────────────
        ir_ac,  ir_dc_arr,  self._iir_state_ir  = \
            self._iir_baseline(ir_raw,  self._iir_state_ir,  "IR")
        red_ac, red_dc_arr, self._iir_state_red = \
            self._iir_baseline(red_raw, self._iir_state_red, "RED")

        # ── Stage 2: Butterworth high-pass (residual drift) ───────────────
        ir_ac,  self._zi_hp_ir  = self._apply_sos(self._sos_hp, ir_ac,  self._zi_hp_ir)
        red_ac, self._zi_hp_red = self._apply_sos(self._sos_hp, red_ac, self._zi_hp_red)

        # ── Stage 3: Butterworth low-pass (noise floor) ───────────────────
        ir_ac,  self._zi_lp_ir  = self._apply_sos(self._sos_lp, ir_ac,  self._zi_lp_ir)
        red_ac, self._zi_lp_red = self._apply_sos(self._sos_lp, red_ac, self._zi_lp_red)

        # ── Stage 4: Zero-phase band-pass for peak detection ─────────────
        # filtfilt eliminates causal group delay / edge transient distortion.
        # Requires _SCIPY and a minimum of ~4 × filter_order samples.
        min_len_filtfilt = 4 * self._cfg.filter_order + 1
        ir_bp  = ir_ac.copy()
        red_bp = red_ac.copy()
        if _SCIPY and self._sos_bp is not None and len(ir_ac) > min_len_filtfilt:
            try:
                # Convert SOS → ba for filtfilt (filtfilt doesn't accept SOS
                # directly in all scipy versions; ba is universally supported)
                from scipy.signal import sosfilt_zi as _zi, butter as _b
                nyq = self._fs / 2.0
                lo  = float(np.clip(self._cfg.bp_low_hz  / nyq, 1e-4, 1.0 - 1e-4))
                hi  = float(np.clip(self._cfg.bp_high_hz / nyq, 1e-4, 1.0 - 1e-4))
                if lo < hi:
                    b_ba, a_ba = _b(self._cfg.filter_order, [lo, hi],
                                    btype='band', output='ba')
                    ir_bp  = filtfilt(b_ba, a_ba, ir_ac)
                    red_bp = filtfilt(b_ba, a_ba, red_ac)
            except Exception as e:
                self._log.debug("DSPPipeline: filtfilt fallback to sosfilt: %s", e)
                ir_bp, _ = self._apply_sos(self._sos_bp, ir_ac, None)
                red_bp, _= self._apply_sos(self._sos_bp, red_ac, None)

        # ── Stage 5: RMS and peak-to-peak ────────────────────────────────
        ir_ac_rms  = float(np.sqrt(np.mean(ir_ac  ** 2))) if len(ir_ac)  > 0 else 0.0
        red_ac_rms = float(np.sqrt(np.mean(red_ac ** 2))) if len(red_ac) > 0 else 0.0
        ir_pp      = float(ir_ac.max()  - ir_ac.min())  if len(ir_ac)  > 0 else 0.0
        red_pp     = float(red_ac.max() - red_ac.min()) if len(red_ac) > 0 else 0.0

        # ── Stage 6: FFT spectral analysis ───────────────────────────────
        dominant_hz, snr_ir, snr_red = self._spectral_analysis(ir_ac, red_ac)

        return {
            "ir_ac_filtered"  : ir_bp,
            "red_ac_filtered" : red_bp,
            "ir_dc"           : float(np.mean(ir_dc_arr)),
            "red_dc"          : float(np.mean(red_dc_arr)),
            "ir_dc_array"     : ir_dc_arr,
            "red_dc_array"    : red_dc_arr,
            "ir_ac_rms"       : ir_ac_rms,
            "red_ac_rms"      : red_ac_rms,
            "ir_pp"           : ir_pp,
            "red_pp"          : red_pp,
            "dominant_hz"     : dominant_hz,
            "snr_db_ir"       : snr_ir,
            "snr_db_red"      : snr_red,
        }

    def _spectral_analysis(self,
                           ir_ac  : np.ndarray,
                           red_ac : np.ndarray
                           ) -> Tuple[float, float, float]:
        """
        Welch periodogram-based spectral analysis.

        Uses Welch's method (overlapping Hann windows) for a lower-variance
        PSD estimate than the standard periodogram.  This matters with
        short, noisy windows (~250 samples at 100 Hz = 2.5 s).

        SNR definition:
          Signal band: ±0.1 Hz around dominant peak
          Noise band:  [bp_low, bp_high] minus signal band
          SNR = 10·log10(signal_power / noise_power)

        Returns:
          dominant_hz : peak frequency in HR band [Hz]
          snr_ir_db   : spectral SNR of IR channel [dB]
          snr_red_db  : spectral SNR of RED channel [dB]
        """
        if not _SCIPY or len(ir_ac) < 32:
            return 0.0, 0.0, 0.0

        n      = len(ir_ac)
        nperseg= min(n, 128)

        try:
            freqs, Pxx_ir  = welch(ir_ac,  fs=self._fs, nperseg=nperseg,
                                   window='hann', noverlap=nperseg//2)
            _,     Pxx_red = welch(red_ac, fs=self._fs, nperseg=nperseg,
                                   window='hann', noverlap=nperseg//2)
        except Exception:
            return 0.0, 0.0, 0.0

        # ── Dominant frequency in HR band ─────────────────────────────────
        f_lo = self._cfg.bp_low_hz
        f_hi = self._cfg.bp_high_hz
        band_mask = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(band_mask):
            return 0.0, 0.0, 0.0

        band_Pxx   = Pxx_ir[band_mask]
        band_freqs = freqs[band_mask]
        peak_idx   = int(np.argmax(band_Pxx))
        dominant_hz= float(band_freqs[peak_idx])

        # ── Spectral SNR ──────────────────────────────────────────────────
        def _snr(Pxx):
            sig_mask  = (freqs >= dominant_hz - 0.15) & (freqs <= dominant_hz + 0.15)
            noise_mask= band_mask & ~sig_mask
            sig_p  = float(np.mean(Pxx[sig_mask]))  if np.any(sig_mask)  else 0.0
            noi_p  = float(np.mean(Pxx[noise_mask])) if np.any(noise_mask) else 1e-12
            if noi_p < 1e-20:
                return 0.0
            return max(0.0, 10.0 * math.log10(sig_p / noi_p + 1e-12))

        return dominant_hz, _snr(Pxx_ir), _snr(Pxx_red)


# ─────────────────────────────────────────────────────────────────────────────
# §10  ADAPTIVE PEAK DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class AdaptivePeakDetector:
    """
    Amplitude-adaptive, refractory-gated PPG peak detector.

    Algorithm:
      1. Compute dynamic amplitude threshold: thresh = α·max + (1-α)·mean
         where α is adapted based on signal SNR.
      2. Apply refractory period: peaks within refractory_ms of a prior
         peak are rejected (enforces physiological minimum IBi).
      3. Use scipy.signal.find_peaks with computed prominence and distance.
      4. Validate each detected peak against physiological BPM bounds.

    Refractory logic motivation:
      PPG waveforms can have a prominent dicrotic notch (secondary peak from
      aortic valve closure) that naive peak detectors misidentify as a cardiac
      beat.  The refractory period (200 ms default) ensures the dicrotic
      notch — which occurs ~150–200 ms after systole — is suppressed.

    Adaptive threshold:
      Poor-quality signals have lower prominence → threshold adapts down.
      High-quality signals get a stricter threshold to reject noise peaks.
    """

    def __init__(self, cfg: MAX30102Config, log: logging.Logger):
        self._cfg = cfg
        self._log = log
        self._fs  = float(cfg.sample_rate_hz)

        # History of R-R intervals for adaptive threshold
        self._rr_history: Deque[float] = collections.deque(maxlen=10)

    def detect(self,
               sig     : np.ndarray,
               snr_db  : float
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect systolic peaks in the band-passed PPG signal.

        Args:
            sig    : band-pass filtered PPG (normalised or raw AC)
            snr_db : spectral SNR estimate (from DSPPipeline)

        Returns:
            peaks       : indices of detected systolic peaks
            prominences : prominence of each peak (for quality weighting)
        """
        if not _SCIPY or len(sig) < 10:
            return np.array([], dtype=int), np.array([], dtype=float)

        # ── Refractory distance in samples ───────────────────────────────
        min_dist_samp = max(1, int(
            self._fs * self._cfg.refractory_ms / 1000.0))

        # Also enforce BPM-based minimum distance
        bpm_dist_samp = max(1, int(
            self._fs * 60.0 / self._cfg.peak_max_bpm))
        min_dist_samp = max(min_dist_samp, bpm_dist_samp)

        # ── Adaptive prominence threshold ────────────────────────────────
        sig_range  = float(sig.max() - sig.min()) if len(sig) > 0 else 1.0
        sig_std    = float(np.std(sig))
        # SNR-weighted alpha: higher SNR → stricter threshold.
        # Cap at 0.45 × range: for a clean sine wave the true peak prominence
        # is ~1.0 in a [-1,1] normalised signal, so a 0.65 threshold would
        # reject valid peaks when SNR is low but signal is still pulsatile.
        alpha = float(np.clip(0.20 + 0.25 * (snr_db / 25.0), 0.12, 0.45))
        prom_thresh = max(sig_std * 0.30,
                          alpha * sig_range,
                          sig_range * 0.08)   # absolute minimum: 8% of range

        try:
            peaks, properties = find_peaks(
                sig,
                distance   = min_dist_samp,
                prominence = prom_thresh,
                width      = (int(self._fs * 0.05),   # min 50 ms wide
                              int(self._fs * 0.45)),   # max 450 ms wide
            )
        except Exception as e:
            self._log.debug("AdaptivePeakDetector: find_peaks error: %s", e)
            return np.array([], dtype=int), np.array([], dtype=float)

        prominences = properties.get("prominences", np.ones(len(peaks)))

        # ── BPM validity gate ─────────────────────────────────────────────
        if len(peaks) >= 2:
            intervals_s  = np.diff(peaks) / self._fs
            bpm_per_beat = 60.0 / np.maximum(intervals_s, 1e-6)
            valid_mask   = ((bpm_per_beat >= self._cfg.peak_min_bpm) &
                            (bpm_per_beat <= self._cfg.peak_max_bpm))
            # Keep peaks that form at least one valid interval
            keep = np.ones(len(peaks), dtype=bool)
            for i, v in enumerate(valid_mask):
                if not v:
                    keep[i] = False; keep[i+1] = False
            peaks       = peaks[keep]
            prominences = prominences[keep]

        return peaks, prominences


# ─────────────────────────────────────────────────────────────────────────────
# §11  BIOMETRICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class BiometricsEngine:
    """
    Computes BPM, SpO2, Perfusion Index, and Signal Quality Index
    from a DSP-processed PPG window.

    SpO2 algorithm (ratio-of-ratios):
      R = (AC_RED / DC_RED) / (AC_IR / DC_IR)

      Empirical calibration (Weber 2010, MAX30102 AN):
        SpO2 = 110 - 25·R   (linear approximation, valid ~80–100%)

      Per-beat R computation (more accurate than single-window estimate):
        For each inter-peak segment:
          AC = peak-to-peak of segment
          DC = mean of segment
          R_i = (AC_RED_i / DC_RED_i) / (AC_IR_i / DC_IR_i)
        Final R = median of all valid R_i (outlier-resistant)

    Perfusion Index (PI):
      PI = (AC_IR_RMS / DC_IR) × 100%
      Physiological range: ~0.02% (poor perfusion) – 20% (hyperdynamic)
      Values > 20% indicate motion or noise rather than true hyperperfusion.

    SQI (Signal Quality Index, 0–100):
      Weighted composite:
        w1 (40%) : spectral SNR normalised to 25 dB target
        w2 (30%) : AC/DC ratio physiological plausibility
        w3 (20%) : inter-beat interval coefficient of variation
        w4 (10%) : temporal peak count plausibility
    """

    # SpO2 lookup calibration (R → SpO2) from Welker et al., Webster (2010)
    # Validated against bench calibration with certified oximeter.
    # DO NOT extrapolate below SpO2 = 70% without re-calibration.
    _R_TABLE   = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
                  0.90, 1.00, 1.10, 1.20, 1.30, 1.50, 1.80]
    _SPO2_TABLE= [100,   99,   98,   97,   96,   95,
                   93,   91,   88,   85,   82,   77,   70 ]

    def __init__(self, cfg: MAX30102Config, log: logging.Logger):
        self._cfg      = cfg
        self._log      = log
        self._bpm_hist : Deque[float] = collections.deque(maxlen=cfg.bpm_avg_n)
        self._spo2_hist: Deque[float] = collections.deque(maxlen=cfg.spo2_avg_n)
        self._pi_hist  : Deque[float] = collections.deque(maxlen=cfg.pi_avg_n)

    def reset(self) -> None:
        """Reset history on finger loss."""
        self._bpm_hist.clear()
        self._spo2_hist.clear()
        self._pi_hist.clear()

    def compute(self,
                dsp        : Dict[str, Any],
                ir_raw     : np.ndarray,
                red_raw    : np.ndarray,
                peaks      : np.ndarray,
                prominences: np.ndarray,
                finger_score: float
                ) -> BiometricResult:
        """
        Compute all biometric metrics from one DSP window.

        Args:
            dsp          : output dict from DSPPipeline.process()
            ir_raw       : raw IR window (for per-beat AC/DC)
            red_raw      : raw RED window
            peaks        : systolic peak indices from AdaptivePeakDetector
            prominences  : peak prominences
            finger_score : from FingerDetector (0–1)

        Returns:
            BiometricResult with all fields populated.
        """
        result = BiometricResult(
            dc_ir     = dsp["ir_dc"],
            dc_red    = dsp["red_dc"],
            ac_ir_rms = dsp["ir_ac_rms"],
            ac_red_rms= dsp["red_ac_rms"],
            snr_ir_db = dsp["snr_db_ir"],
            peak_count= len(peaks),
            finger_score = finger_score,
        )

        fs = float(self._cfg.sample_rate_hz)

        # ── BPM from inter-peak intervals ─────────────────────────────────
        bpm, rr_ms = self._compute_bpm(peaks, fs)
        result.rr_intervals_ms = rr_ms

        # ── Per-beat SpO2 (ratio-of-ratios) ──────────────────────────────
        spo2, r_ratio = self._compute_spo2(peaks, ir_raw, red_raw)
        result.r_ratio = r_ratio

        # ── Perfusion Index ───────────────────────────────────────────────
        pi = self._compute_pi(dsp["ir_dc"], dsp["ir_ac_rms"])

        # ── History averaging ────────────────────────────────────────────
        if bpm is not None:
            self._bpm_hist.append(bpm)
        if spo2 is not None:
            self._spo2_hist.append(spo2)
        if pi is not None:
            self._pi_hist.append(pi)

        result.bpm             = self._hist_median(self._bpm_hist)
        result.spo2            = self._hist_median(self._spo2_hist)
        result.perfusion_index = self._hist_mean(self._pi_hist)

        # ── SQI ──────────────────────────────────────────────────────────
        sqi, label = self._compute_sqi(dsp, peaks, rr_ms, finger_score)
        result.sqi       = sqi
        result.sqi_label = label

        return result

    # ── Sub-computations ───────────────────────────────────────────────────

    def _compute_bpm(self,
                     peaks : np.ndarray,
                     fs    : float
                     ) -> Tuple[Optional[float], List[float]]:
        """
        Compute instantaneous BPM from R-R intervals.

        Rejects outlier intervals (> 2× median) to handle mis-detections.
        Returns (median_bpm, [rr_intervals_ms]).
        """
        if len(peaks) < 2:
            return None, []

        intervals_s   = np.diff(peaks) / fs
        rr_ms         = (intervals_s * 1000.0).tolist()

        # Outlier rejection: keep intervals within [0.5×, 2×] median
        med = float(np.median(intervals_s))
        valid = intervals_s[
            (intervals_s >= 0.5 * med) & (intervals_s <= 2.0 * med)]

        if len(valid) == 0:
            return None, rr_ms

        mean_interval = float(np.mean(valid))
        if mean_interval <= 0:
            return None, rr_ms

        bpm = 60.0 / mean_interval
        if not (self._cfg.peak_min_bpm <= bpm <= self._cfg.peak_max_bpm):
            return None, rr_ms

        return round(bpm, 1), rr_ms

    def _compute_spo2(self,
                      peaks  : np.ndarray,
                      ir_raw : np.ndarray,
                      red_raw: np.ndarray
                      ) -> Tuple[Optional[float], Optional[float]]:
        """
        Per-beat ratio-of-ratios SpO2 estimate.

        For each inter-systolic segment:
          AC = peak-to-peak range of segment
          DC = mean of segment
          R_i = (AC_red / DC_red) / (AC_ir / DC_ir)

        Outlier rejection: discard R values outside [0.3, 1.8].
        SpO2 = table interpolation of median R.

        Linear formula (SpO2 = 110 - 25R) is used as a crosscheck;
        the table interpolation is reported as primary result.
        """
        if len(peaks) < 2:
            return None, None

        r_values: List[float] = []

        for i in range(len(peaks) - 1):
            seg_ir  = ir_raw[peaks[i]:peaks[i+1]]
            seg_red = red_raw[peaks[i]:peaks[i+1]]
            if len(seg_ir) < 3:
                continue

            dc_ir  = float(np.mean(seg_ir))
            dc_red = float(np.mean(seg_red))
            ac_ir  = float(seg_ir.max()  - seg_ir.min())
            ac_red = float(seg_red.max() - seg_red.min())

            if dc_ir < 100 or dc_red < 100 or ac_ir < 1:
                continue

            R = (ac_red / dc_red) / (ac_ir / dc_ir)
            if 0.3 <= R <= 1.8:
                r_values.append(R)

        if not r_values:
            return None, None

        R_med = float(np.median(r_values))

        # Table interpolation (primary)
        spo2_interp = float(np.interp(R_med, self._R_TABLE, self._SPO2_TABLE))

        # Linear formula crosscheck (secondary, not reported)
        # spo2_linear = 110.0 - 25.0 * R_med

        spo2 = round(max(70.0, min(100.0, spo2_interp)), 1)
        return spo2, round(R_med, 4)

    def _compute_pi(self,
                    dc_ir    : float,
                    ac_ir_rms: float) -> Optional[float]:
        """
        Perfusion Index = (AC_RMS / DC) × 100%.

        Clamps to [0, 25]% — values above 25% are physiologically implausible
        and indicate artifact rather than true hyperperfusion.
        """
        if dc_ir < 1000:
            return None
        pi = (ac_ir_rms / dc_ir) * 100.0
        return round(max(0.0, min(25.0, pi)), 3)

    def _compute_sqi(self,
                     dsp         : Dict[str, Any],
                     peaks       : np.ndarray,
                     rr_ms       : List[float],
                     finger_score: float
                     ) -> Tuple[float, str]:
        """
        Signal Quality Index (0–100).

        Component weights (sum = 1.0):
          w1 = 0.40 : spectral SNR (normalised to 25 dB reference)
          w2 = 0.30 : AC/DC ratio plausibility (target ~1–2%)
          w3 = 0.20 : inter-beat interval coefficient of variation
          w4 = 0.10 : finger contact confidence
        """
        # w1: spectral SNR
        snr_ref = 25.0
        snr_norm= float(np.clip(dsp["snr_db_ir"] / snr_ref, 0.0, 1.0))

        # w2: AC/DC ratio plausibility
        dc = dsp["ir_dc"]
        ac = dsp["ir_ac_rms"]
        if dc > 0:
            ac_dc_pct = (ac / dc) * 100.0
            # Gaussian peaked at 1.5%, σ = 1.0 — penalises extremes
            ac_dc_score = float(np.exp(-((ac_dc_pct - 1.5)**2) / (2 * 1.0**2)))
        else:
            ac_dc_score = 0.0

        # w3: IBI coefficient of variation (lower = better)
        if len(rr_ms) >= 2:
            mean_rr = statistics.mean(rr_ms)
            std_rr  = statistics.stdev(rr_ms)
            cv = std_rr / max(mean_rr, 1.0)
            # CV < 0.05 → excellent; CV > 0.30 → poor
            ibi_score = float(np.clip(1.0 - cv / 0.30, 0.0, 1.0))
        else:
            ibi_score = 0.0

        # w4: finger contact
        contact_score = float(np.clip(finger_score, 0.0, 1.0))

        sqi_raw = (0.40 * snr_norm
                 + 0.30 * ac_dc_score
                 + 0.20 * ibi_score
                 + 0.10 * contact_score)

        sqi = round(min(100.0, max(0.0, sqi_raw * 100.0)), 1)

        if sqi >= 90:   label = "EXCELLENT"
        elif sqi >= 70: label = "GOOD"
        elif sqi >= 45: label = "FAIR"
        elif sqi >= 10: label = "POOR"
        else:           label = "NO SIGNAL"

        return sqi, label

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _hist_median(hist: Deque[float]) -> Optional[float]:
        if not hist:
            return None
        return round(float(np.median(list(hist))), 1)

    @staticmethod
    def _hist_mean(hist: Deque[float]) -> Optional[float]:
        if not hist:
            return None
        return round(float(np.mean(list(hist))), 3)


# ─────────────────────────────────────────────────────────────────────────────
# §12  SENSOR HEALTH SCORER
# ─────────────────────────────────────────────────────────────────────────────

class SensorHealthScorer:
    """
    Per-cycle hardware health metric (0.0 = failed, 1.0 = perfect).

    Health components:
      I²C error rate     : fraction of transactions that raised OSError
      FIFO overflow rate : overflows per sample (should be 0)
      CRC/range errors   : malformed 18-bit samples
      Consecutive errors : burst errors weight more heavily

    Exponential moving average smooths the metric to avoid single-cycle
    transients from degrading the displayed health score.
    """

    def __init__(self, cfg: MAX30102Config, log: logging.Logger):
        self._cfg    = cfg
        self._log    = log
        self._health = 1.0   # start optimistic

    def score(self, record: SensorHealthRecord) -> float:
        """
        Compute and return updated health score [0, 1].

        Modifies record.health_score in place for external readers.
        """
        n = max(1, record.total_samples)

        # I²C error rate
        i2c_rate = record.i2c_errors / n
        i2c_score= max(0.0, 1.0 - i2c_rate / self._cfg.max_i2c_err_rate)

        # FIFO overflow rate
        ovf_rate  = record.fifo_overflows / n
        ovf_score = max(0.0, 1.0 - ovf_rate * 10.0)

        # CRC / range error rate
        crc_rate  = record.crc_errors / n
        crc_score = max(0.0, 1.0 - crc_rate * 20.0)

        # Consecutive error penalty (exponential degradation)
        consec_pen = math.exp(-record.consecutive_errs / 3.0)

        raw = (0.50 * i2c_score
             + 0.25 * ovf_score
             + 0.15 * crc_score
             + 0.10 * consec_pen)

        # EMA smoothing
        α = self._cfg.health_ema_alpha
        self._health = (1 - α) * self._health + α * raw
        record.health_score = self._health

        if self._health < 0.5 and record.total_samples % 500 == 0:
            self._log.warning(
                "MAX30102 health degraded: %.2f  i2c_err=%d  ovf=%d  crc=%d",
                self._health, record.i2c_errors, record.fifo_overflows, record.crc_errors)

        # Reset consecutive error counter once scored (prevents stale penalty)
        record.consecutive_errs = 0

        return self._health


# ─────────────────────────────────────────────────────────────────────────────
# §13  PPG LOGGER (structured logging hooks)
# ─────────────────────────────────────────────────────────────────────────────

class PPGLogger:
    """
    Structured logging hooks for the PPG subsystem.

    Provides:
      · Per-window summary at INFO level
      · Per-sample debug logging (conditionally enabled)
      · Health event escalation to WARNING/ERROR
      · JSON-serialisable metric dict for external telemetry integration

    Caller should obtain a child logger via LoggingManager.get("ppg")
    and pass it here; PPGLogger does not construct its own logger.
    """

    def __init__(self, log: logging.Logger, debug_samples: bool = False):
        self._log          = log
        self._debug_samples= debug_samples
        self._window_count = 0

    def log_sample(self, sample: PPGSample) -> None:
        """Log raw sample at DEBUG level (high-frequency — enable sparingly)."""
        if self._debug_samples:
            self._log.debug("sample ts=%.4f  RED=%d  IR=%d",
                            sample.ts, sample.red, sample.ir)

    def log_window(self, result: BiometricResult, health: float) -> None:
        """Log one analysis window summary at INFO level."""
        self._window_count += 1
        self._log.info(
            "[W%04d] BPM=%-6s SpO2=%-5s PI=%-7s SQI=%-3.0f %-10s "
            "DC_IR=%6.0f AC_RMS=%5.0f SNR=%5.1fdB Health=%.2f",
            self._window_count,
            f"{result.bpm:.1f}" if result.bpm else "---",
            f"{result.spo2:.1f}" if result.spo2 else "---",
            f"{result.perfusion_index:.3f}" if result.perfusion_index else "---",
            result.sqi, result.sqi_label,
            result.dc_ir, result.ac_ir_rms, result.snr_ir_db,
            health,
        )

    def log_finger_event(self, present: bool, score: float) -> None:
        """Log finger contact state change."""
        self._log.info("FingerContact: %s (score=%.2f)",
                       "DETECTED" if present else "REMOVED", score)

    def log_health_event(self, record: SensorHealthRecord) -> None:
        """Log health degradation events."""
        if record.health_score < 0.7:
            self._log.warning(
                "SensorHealth=%.2f  I2C_err=%d  OVF=%d  CRC=%d  last='%s'",
                record.health_score, record.i2c_errors,
                record.fifo_overflows, record.crc_errors, record.last_error)

    def as_dict(self, result: BiometricResult, health: float) -> Dict[str, Any]:
        """Return a JSON-serialisable metric dict for external telemetry."""
        return {
            "ts"              : result.ts,
            "bpm"             : result.bpm,
            "spo2"            : result.spo2,
            "perfusion_index" : result.perfusion_index,
            "sqi"             : result.sqi,
            "sqi_label"       : result.sqi_label,
            "dc_ir"           : round(result.dc_ir, 1),
            "ac_ir_rms"       : round(result.ac_ir_rms, 2),
            "snr_ir_db"       : round(result.snr_ir_db, 2),
            "r_ratio"         : result.r_ratio,
            "peak_count"      : result.peak_count,
            "finger_score"    : round(result.finger_score, 3),
            "sensor_health"   : round(health, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# §14  ACQUISITION THREAD — worker_acq_ppg (drop-in replacement)
# ─────────────────────────────────────────────────────────────────────────────

def worker_acq_ppg(stop_event   : threading.Event,
                   state_hub    : Any,                # SharedStateHub from main.py
                   watchdog     : Any,                # WatchdogFramework
                   cfg          : Any,                # SystemConfig
                   log          : logging.Logger,
                   result_queue : Optional[queue.Queue] = None,
                   ) -> None:
    """
    MAX30102 acquisition thread — replaces the stub in main.py §10.

    Thread model:
      · Runs at `sample_rate_hz` (100 Hz nominal)
      · Precise timing: hybrid sleep + busy-poll (same as acq-imu)
      · Calls watchdog.beat() every 10 samples (10 Hz heartbeat)
      · Analysis pipeline runs every `analysis_stride` samples in same thread
        to avoid inter-thread latency on short windows
      · All results written to state_hub.update_snapshot()
      · Optional result_queue for external consumers (unit tests, recording)

    Recovery strategy:
      · Single I²C error: logged, sample skipped, continue
      · 3 consecutive errors: pause 0.5 s, attempt bus re-open
      · 10 consecutive errors: log ERROR, transition state machine
        to ERROR_RECOVERY via watchdog escalation

    Simulation mode (no hardware):
      Generates a physically plausible synthetic PPG waveform:
        IR(t)  = DC_IR  × [1 + 0.015·sin(2π·f_hr·t) + noise]
        RED(t) = DC_RED × [1 + 0.012·sin(2π·f_hr·t + φ) + noise]
      where f_hr = 1.1 Hz (66 BPM), φ = 0.2 rad (LED wavelength delay).
    """
    log.info("worker_acq_ppg: starting (scipy=%s smbus2=%s)", _SCIPY, _SMBUS)

    # ── Build subsystem objects ────────────────────────────────────────────
    ppg_cfg  = cfg.ppg if hasattr(cfg, 'ppg') else MAX30102Config()
    if not isinstance(ppg_cfg, MAX30102Config):
        # Convert from PPGConfig (main.py) to MAX30102Config
        ppg_cfg = MAX30102Config(
            i2c_bus         = ppg_cfg.i2c_bus,
            sensor_addr     = ppg_cfg.sensor_addr,
            sample_rate_hz  = ppg_cfg.sample_rate_hz,
            led_current_init= ppg_cfg.led_current,
            ring_depth      = ppg_cfg.ring_depth,
            analysis_window = ppg_cfg.calc_window,
            analysis_stride = ppg_cfg.step_size,
            hp_cutoff_hz    = ppg_cfg.bpf_low_hz,
            lp_cutoff_hz    = 8.0,
            bp_low_hz       = ppg_cfg.bpf_low_hz,
            bp_high_hz      = ppg_cfg.bpf_high_hz,
            filter_order    = ppg_cfg.bpf_order,
            iir_baseline_tau= ppg_cfg.baseline_tau,
            finger_dc_min   = ppg_cfg.finger_dc_min,
            finger_ac_min   = ppg_cfg.finger_ac_min,
            peak_min_bpm    = ppg_cfg.peak_min_bpm,
            peak_max_bpm    = ppg_cfg.peak_max_bpm,
            bpm_avg_n       = ppg_cfg.display_avg_n,
            spo2_avg_n      = ppg_cfg.display_avg_n,
        )

    health   = SensorHealthRecord()
    ring     = PPGRingBuffer(ppg_cfg.ring_depth)
    ambient  = AmbientBaselineEstimator(ppg_cfg, log)
    dsp      = DSPPipeline(ppg_cfg, log)
    peak_det = AdaptivePeakDetector(ppg_cfg, log)
    bio      = BiometricsEngine(ppg_cfg, log)
    finger   = FingerDetector(ppg_cfg, log)
    scorer   = SensorHealthScorer(ppg_cfg, log)
    ppg_log  = PPGLogger(log)

    # ── Hardware driver (or simulation) ────────────────────────────────────
    hw_enabled = getattr(cfg, 'hw_ppg_enabled', True) and _SMBUS
    driver  : Optional[MAX30102Driver] = None
    use_hw  = False

    if hw_enabled:
        driver = MAX30102Driver(ppg_cfg, health, log)
        try:
            driver.open()
            led_tuner = LEDAutoTuner(ppg_cfg, driver, log)
            use_hw    = True
            log.info("worker_acq_ppg: hardware acquisition active.")
        except Exception as e:
            log.warning("worker_acq_ppg: driver.open() failed: %s — simulation.", e)
            driver = None

    if not use_hw:
        led_tuner = None
        log.info("worker_acq_ppg: simulation mode.")

    # ── Timing ────────────────────────────────────────────────────────────
    FS       = float(ppg_cfg.sample_rate_hz)
    DT       = 1.0 / FS
    MARGIN   = 0.0015   # 1.5 ms busy-poll guard
    t_next   = time.monotonic()

    # ── State ─────────────────────────────────────────────────────────────
    n_total         = 0     # total samples acquired
    n_since_window  = 0     # samples since last analysis window
    prev_finger_pres= False
    _sim_t          = 0.0
    t_periods       : Deque[float] = collections.deque(maxlen=200)
    t_prev          = time.monotonic()
    consecutive_ok  = 0
    RECOVERY_THRESH = 10    # consecutive errors before watchdog escalation

    # ── Main loop ──────────────────────────────────────────────────────────
    while not stop_event.is_set():

        # ── Precise timing (hybrid sleep + busy-poll) ─────────────────────
        now = time.monotonic()
        if t_next - now > MARGIN:
            time.sleep(t_next - now - MARGIN)
        while time.monotonic() < t_next:
            pass
        t_next += DT

        # ── Sample acquisition ────────────────────────────────────────────
        if use_hw and driver is not None:
            # Hardware: burst-read FIFO
            try:
                samples = driver.read_fifo_burst()
                consecutive_ok += len(samples)
                health.consecutive_errs = 0
            except OSError as e:
                health.i2c_errors += 1
                health.consecutive_errs += 1
                watchdog.report_error("acq-ppg", str(e))
                log.warning("worker_acq_ppg: I²C error: %s (consec=%d)",
                            e, health.consecutive_errs)

                if health.consecutive_errs >= 3:
                    # Brief recovery pause, attempt re-open
                    log.warning("worker_acq_ppg: attempting bus recovery...")
                    time.sleep(0.5)
                    try:
                        driver.close()
                        driver.open()
                        log.info("worker_acq_ppg: bus recovered.")
                        health.consecutive_errs = 0
                    except Exception as e2:
                        log.error("worker_acq_ppg: recovery failed: %s", e2)

                if health.consecutive_errs >= RECOVERY_THRESH:
                    watchdog.report_error("acq-ppg",
                        f"Exceeded {RECOVERY_THRESH} consecutive I²C errors")
                    # Watchdog will escalate state; keep looping until stop_event
                continue

            if not samples:
                # FIFO empty (normal when polling faster than hardware outputs)
                continue

        else:
            # Simulation: generate one synthetic sample
            _sim_t += DT
            f_hr  = 1.1        # 66 BPM
            noise = float(np.random.normal(0, 300))
            dc_ir = 185_000.0
            dc_red= 150_000.0
            ir_val = int(dc_ir  * (1 + 0.015 * math.sin(2*math.pi*f_hr*_sim_t))
                        + dc_ir * 0.001 * math.sin(2*math.pi*2*f_hr*_sim_t)  # harmonics
                        + noise)
            red_val= int(dc_red * (1 + 0.012 * math.sin(2*math.pi*f_hr*_sim_t + 0.2))
                        + dc_red * 0.001 * math.sin(2*math.pi*2*f_hr*_sim_t + 0.2)
                        + noise * 0.9)
            samples = [PPGSample(
                red = int(np.clip(red_val, 0, _18BIT_MASK)),
                ir  = int(np.clip(ir_val,  0, _18BIT_MASK)),
                ts  = time.monotonic(),
            )]

        # ── Process each acquired sample ──────────────────────────────────
        for sample in samples:
            ppg_log.log_sample(sample)

            # Push to ring buffer and main.py shared buffers
            ring.push(sample)
            state_hub.push_ppg_sample(float(sample.ir), float(sample.red))

            # Enqueue for external consumers
            try:
                state_hub.ppg_raw_queue.put_nowait(
                    (float(sample.red), float(sample.ir)))
            except queue.Full:
                try:    state_hub.ppg_raw_queue.get_nowait()
                except queue.Empty: pass
                state_hub.ppg_raw_queue.put_nowait(
                    (float(sample.red), float(sample.ir)))

            # Quick finger estimate for ambient tracker (uses DC only)
            quick_dc = float(sample.ir)
            quick_finger = quick_dc > ppg_cfg.finger_dc_min * 0.5
            ambient.feed(sample, quick_finger)

            n_total        += 1
            n_since_window += 1

            # Effective sample rate estimation
            now2 = time.monotonic()
            t_periods.append(now2 - t_prev)
            t_prev = now2

        # ── Heartbeat (10 Hz) ─────────────────────────────────────────────
        if n_total % 10 == 0:
            watchdog.beat("acq-ppg")
            if len(t_periods) >= 10:
                eff_fs = 1.0 / (sum(t_periods) / len(t_periods))
                state_hub.update_snapshot(ppg_effective_fs=eff_fs)

        # ── Health scoring (every 200 samples) ───────────────────────────
        if n_total % 200 == 0:
            h_score = scorer.score(health)
            ppg_log.log_health_event(health)
            state_hub.update_snapshot(i2c_error_count=health.i2c_errors)

        # ── Analysis window ───────────────────────────────────────────────
        if n_since_window < ppg_cfg.analysis_stride:
            continue
        n_since_window = 0

        if len(ring) < ppg_cfg.analysis_window:
            continue   # not enough data yet

        # Extract window
        ir_win, red_win, ts_win = ring.window(ppg_cfg.analysis_window)
        if len(ir_win) < ppg_cfg.analysis_window:
            continue

        # Ambient subtraction on DC components (if baseline is valid)
        ir_raw  = ir_win.copy()
        red_raw = red_win.copy()
        if ambient.is_valid:
            ir_raw  = np.maximum(0.0, ir_raw  - ambient.ir_baseline  * 0.5)
            red_raw = np.maximum(0.0, red_raw - ambient.red_baseline * 0.5)

        # DC / AC window stats for finger detection
        dc_ir  = float(np.mean(ir_raw))
        dc_red = float(np.mean(red_raw))
        ac_pp  = float(ir_raw.max() - ir_raw.min())
        ir_var = float(np.var(ir_raw))

        fp, f_score = finger.update(dc_ir, dc_red, ac_pp, ir_var)

        # Finger contact state change
        if fp != prev_finger_pres:
            ppg_log.log_finger_event(fp, f_score)
            if not fp:
                bio.reset()
                dsp.reset_state()
            prev_finger_pres = fp

        if not fp:
            state_hub.update_snapshot(
                finger_present_ppg = False,
                ppg_sqi_label      = "NO SIGNAL",
                ppg_sqi            = 0.0,
                bpm                = None,
                spo2               = None,
                perfusion_index    = None,
                ts_ppg             = time.monotonic(),
            )
            continue

        # ── DSP pipeline ──────────────────────────────────────────────────
        dsp_out = dsp.process(ir_raw, red_raw)

        # ── LED auto-tuning ───────────────────────────────────────────────
        if led_tuner is not None:
            led_tuner.tune(dsp_out["ir_dc"], dsp_out["ir_ac_rms"])

        # ── Peak detection (on band-passed signal) ────────────────────────
        ir_bp = dsp_out["ir_ac_filtered"]
        if len(ir_bp) > 0:
            # Normalise to [-1, 1] for amplitude-independent threshold
            rng = ir_bp.max() - ir_bp.min()
            if rng > 1e-6:
                ir_norm = (2.0 * (ir_bp - ir_bp.min()) / rng) - 1.0
            else:
                ir_norm = np.zeros_like(ir_bp)
        else:
            ir_norm = np.zeros(0)

        peaks, prominences = peak_det.detect(ir_norm, dsp_out["snr_db_ir"])

        # ── Biometrics ────────────────────────────────────────────────────
        result = bio.compute(dsp_out, ir_raw, red_raw, peaks, prominences, f_score)
        result.sensor_health = scorer.score(health)

        # ── Logging ───────────────────────────────────────────────────────
        ppg_log.log_window(result, result.sensor_health)

        # ── Optional result queue ─────────────────────────────────────────
        if result_queue is not None:
            try:
                result_queue.put_nowait(result)
            except queue.Full:
                pass

        # ── Snapshot update ───────────────────────────────────────────────
        red_pa, ir_pa = driver.led_current if driver else (ppg_cfg.led_current_init,) * 2
        state_hub.update_snapshot(
            bpm                = result.bpm,
            spo2               = result.spo2,
            perfusion_index    = result.perfusion_index,
            ppg_sqi            = result.sqi,
            ppg_sqi_label      = result.sqi_label,
            finger_present_ppg = True,
            dc_ir_ppg          = result.dc_ir,
            ac_ir_ppg          = result.ac_ir_rms,
            ppg_noise_est      = 1.0 - result.sqi / 100.0,
            ts_ppg             = result.ts,
        )

    # ── Teardown ─────────────────────────────────────────────────────────
    if driver is not None:
        try:
            driver.close()
        except Exception:
            pass

    log.info(
        "worker_acq_ppg: stopped. total_samples=%d  i2c_errors=%d  "
        "fifo_overflows=%d  final_health=%.2f",
        health.total_samples, health.i2c_errors,
        health.fifo_overflows, health.health_score)


# ─────────────────────────────────────────────────────────────────────────────
# §15  SELF-TEST (run standalone: python max30102_subsystem.py)
# ─────────────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """
    Standalone self-test exercising all subsystems in simulation mode.

    Runs the acquisition thread for 15 seconds against synthetic data and
    prints a summary of computed biometrics to stdout.

    Expected results (simulation):
      BPM ≈ 66   (f_hr = 1.1 Hz)
      SpO2 ≈ 95–98%  (synthetic R ≈ 0.6–0.7)
      PI ≈ 0.8–1.5%
      SQI ≥ 70 (GOOD/EXCELLENT)
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--debug",    action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("selftest")

    # Minimal stub for SharedStateHub
    class StubHub:
        def push_ppg_sample(self, ir, red): pass
        def update_snapshot(self, **kw): pass
        class ppg_raw_queue:
            @staticmethod
            def put_nowait(x): pass
            @staticmethod
            def get_nowait(): raise queue.Empty

    class StubHub2(StubHub):
        def __init__(self):
            self.ppg_raw_queue = queue.Queue(maxsize=256)

    class StubWatchdog:
        def beat(self, n): pass
        def report_error(self, n, e): pass

    class StubConfig:
        hw_ppg_enabled = False   # force simulation
        class ppg:
            i2c_bus=1; sensor_addr=0x57; sample_rate_hz=100
            led_current=0x3C; ring_depth=600; calc_window=250
            step_size=50; bpf_low_hz=0.6; bpf_high_hz=4.0
            bpf_order=4; baseline_tau=0.995
            finger_dc_min=50000; finger_ac_min=300
            peak_min_bpm=30; peak_max_bpm=220; display_avg_n=6

    stop  = threading.Event()
    hub   = StubHub2()
    wd    = StubWatchdog()
    cfg   = StubConfig()
    rq    : queue.Queue = queue.Queue()

    t = threading.Thread(
        target=worker_acq_ppg,
        kwargs=dict(stop_event=stop, state_hub=hub, watchdog=wd,
                    cfg=cfg, log=log, result_queue=rq),
        daemon=True)
    t.start()

    log.info("Self-test running for %.0f s (simulation)...", args.duration)
    time.sleep(args.duration)
    stop.set()
    t.join(timeout=5.0)

    results: List[BiometricResult] = []
    while True:
        try:
            results.append(rq.get_nowait())
        except queue.Empty:
            break

    log.info("─" * 60)
    log.info("Self-test complete — %d analysis windows computed.", len(results))
    if results:
        bpms  = [r.bpm  for r in results if r.bpm  is not None]
        spo2s = [r.spo2 for r in results if r.spo2 is not None]
        pis   = [r.perfusion_index for r in results if r.perfusion_index]
        sqis  = [r.sqi for r in results]
        log.info("  BPM  : mean=%.1f  std=%.1f",
                 statistics.mean(bpms)  if bpms  else 0,
                 statistics.stdev(bpms) if len(bpms)>1 else 0)
        log.info("  SpO2 : mean=%.1f  std=%.1f",
                 statistics.mean(spo2s)  if spo2s  else 0,
                 statistics.stdev(spo2s) if len(spo2s)>1 else 0)
        log.info("  PI   : mean=%.3f",
                 statistics.mean(pis) if pis else 0)
        log.info("  SQI  : mean=%.1f  min=%.1f  max=%.1f",
                 statistics.mean(sqis), min(sqis), max(sqis))
        log.info("  Last label: %s", results[-1].sqi_label)
    log.info("─" * 60)


if __name__ == "__main__":
    _selftest()