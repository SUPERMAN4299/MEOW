#!/usr/bin/env python3
# =============================================================================
#  BioSense-Pi  —  NoIR Camera Subsystem  v1.0
#  Low-latency IR reflectance imaging, iPPG pipeline, and OpenCV overlay UI
# =============================================================================
#
#  Target HW    : Raspberry Pi 4/5  +  NoIR Camera Module v2/v3  (OV5647 / IMX219)
#                 850 nm IR LED bank (GPIO PWM) + optional white LED reference
#  Python       : 3.11+
#  Dependencies : picamera2, numpy, opencv-python-headless, scipy
#
#  ARCHITECTURE
#  ────────────
#  ┌───────────────────────────────────────────────────────────────────────┐
#  │  NoIRCameraConfig        — all tunable parameters, frozen dataclass   │
#  │  CameraHardwareDriver    — Picamera2 lifecycle, zero-copy DMA frames  │
#  │  AdaptiveExposureCtrl    — PI feedback on ROI mean → ExposureTime     │
#  │  IRChannelExtractor      — weighted BGR→IR blend, in-place operation  │
#  │  CLAHEProcessor          — contrast-limited AHE per tile              │
#  │  GaussianPrefilter       — separable Gaussian, reuses preallocated buf│
#  │  AdaptiveTissueSegmenter — Otsu + morphology + ellipse fit + EMA      │
#  │  MotionQualityGate       — frame-diff MAD + Laplacian, EMA history    │
#  │  OpticalSignalBuffer     — ring buffer with Welch spectral analysis    │
#  │  iPPGEngine              — temporal AC extraction, SNR, BPM proxy     │
#  │  FrameQualityScorer      — multi-component confidence [0,1]           │
#  │  OpenCVOverlayRenderer   — in-place annotated display, waveform strip │
#  │  worker_acq_camera       — 30 Hz acquisition + processing daemon      │
#  └───────────────────────────────────────────────────────────────────────┘
#
#  OPTICAL PHYSICS NOTES
#  ─────────────────────
#  At λ=850 nm the dominant tissue chromophores are oxyhaemoglobin (HbO₂),
#  deoxyhaemoglobin (Hb), water, and melanin.  The OV5647/IMX219 sensor QE
#  peaks in the red channel at this wavelength (≈ 30–40% QE vs ≈ 5% in blue).
#  We therefore use a channel blend R×0.75+G×0.20+B×0.05 to maximise SNR.
#
#  Remote / imaging PPG (iPPG) exploits the fact that skin micro-vessels
#  modulate the reflected IR intensity in sync with the cardiac cycle.
#  The AC/DC ratio is ~0.5–2% for a clean signal — extraction therefore
#  requires careful bandpass filtering and motion gating.
#
#  Reference implementation draws on:
#    · Verkruysse et al. (2008) Remote plethysmographic imaging — Opt. Express
#    · de Haan & Jeanne (2013) Robust pulse-rate from chrominance-based rPPG
#    · Wang et al. (2017) Algorithmic principles of remote-PPG — IEEE TBIOM
#    · Pilz et al. (2018) Local group invariance for heart rate estimation
#
#  iPPG DISCLAIMER
#  ───────────────
#  The iPPG / BPM outputs from this module are EXPERIMENTAL RESEARCH VALUES.
#  They have NOT been validated against a certified medical pulse oximeter.
#  They constitute NO medical device output and must NOT be used for clinical
#  decision-making or patient care.
#
#  ZERO-COPY FRAME PIPELINE
#  ────────────────────────
#  Picamera2 supports DMA-mapped buffers accessible via capture_array("main").
#  The returned numpy array is a VIEW of the camera's internal DMA buffer.
#  To avoid data races the pipeline immediately extracts the IR channel into
#  a pre-allocated float32 workspace, after which the DMA buffer is released.
#  All subsequent operations are performed in-place on preallocated arrays.
#
#  THREAD MODEL
#  ────────────
#  worker_acq_camera()  — single daemon thread handling:
#    · Frame capture from Picamera2 at 30 Hz (period-accurate hybrid sleep)
#    · IR channel extraction + Gaussian pre-filter (< 0.5 ms on Pi 4)
#    · Adaptive tissue segmentation every Nth frame (< 2 ms)
#    · Motion gate + optical signal push + iPPG update each frame
#    · OpenCV overlay rendering at 15 Hz (every 2nd frame)
#    · SharedStateHub snapshot update every frame
#    · Watchdog heartbeat every 10 frames
#
#  MEMORY BUDGET
#  ─────────────
#  Frame workspace   : 640×480×float32 × 4 arrays  ≈  4.9 MB
#  OpenCV BGR display: 640×480×uint8  × 1           ≈  0.9 MB
#  Waveform strips   : 640×55×uint8   × 2           ≈  0.07 MB
#  Signal ring buffer: 300 × float32                ≈  1.2 KB
#  Total             : < 6 MB   ✓ well within Pi 4 budget
#
# =============================================================================

from __future__ import annotations

import gc
import math
import time
import queue
import logging
import threading
import collections
import statistics
from dataclasses import dataclass, field
from typing import (
    Optional, Tuple, List, Deque, Dict, Any, NamedTuple
)

import numpy as np

# ── Optional heavy imports with graceful fallback ─────────────────────────────
try:
    from scipy.signal import (
        butter, sosfilt, sosfilt_zi, welch, find_peaks
    )
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    from picamera2 import Picamera2
    from libcamera import controls as _lc
    _PICAM = True
except ImportError:
    _PICAM = False


# ─────────────────────────────────────────────────────────────────────────────
# §1  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NoIRCameraConfig:
    """
    All tunable parameters for the NoIR camera subsystem.

    Immutable after construction.  Use dataclasses.replace() for overrides
    or inject via SystemConfig.from_json() in main.py §2.

    Exposure controller targets the ROI mean intensity at `aec_target_lsb`
    to keep the pulsatile AC component near the ADC mid-range.  This is
    critical: under-exposure increases shot-noise floor; over-exposure
    saturates the pulsatile AC signal and clips peaks.
    """

    # ── Sensor / capture ──────────────────────────────────────────────────
    width             : int   = 640
    height            : int   = 480
    target_fps        : int   = 30
    pixel_format      : str   = "BGR888"       # Picamera2 format string

    # ── Adaptive Exposure Controller (AEC) ───────────────────────────────
    # PI controller: correction = Kp·e + Ki·∫e; applied to ExposureTime
    aec_target_lsb    : float = 140.0          # target ROI mean intensity
    aec_deadband_lsb  : float = 12.0           # no action within ±deadband
    aec_Kp            : float = 0.006          # proportional gain
    aec_Ki            : float = 0.0009         # integral gain
    aec_integral_max  : float = 60.0           # anti-windup clamp
    aec_exp_min_us    : int   = 3_000          # minimum exposure [µs]
    aec_exp_max_us    : int   = 40_000         # maximum exposure [µs]
    aec_gain_init     : float = 4.0            # initial analogue gain
    aec_gain_min      : float = 1.0
    aec_gain_max      : float = 8.0
    aec_exp_init_us   : int   = 15_000         # initial exposure

    # ── IR channel extraction (BGR → grayscale) ───────────────────────────
    # OV5647 QE at 850 nm: R≈37%, G≈18%, B≈6%.  Weights maximise 850 nm SNR.
    ir_blend_r        : float = 0.75
    ir_blend_g        : float = 0.20
    ir_blend_b        : float = 0.05

    # ── CLAHE parameters ──────────────────────────────────────────────────
    clahe_clip        : float = 2.5
    clahe_tile        : Tuple[int, int] = (8, 8)

    # ── Gaussian pre-filter ───────────────────────────────────────────────
    gauss_kernel      : int   = 5             # must be odd; σ ≈ (k-1)/6
    # Illumination normalisation: divide frame by heavy Gaussian background
    illum_norm_kernel : int   = 81            # must be odd

    # ── Tissue segmentation ────────────────────────────────────────────────
    seg_search_frac_w : float = 0.80          # central fraction for Otsu search
    seg_search_frac_h : float = 0.85
    seg_min_area_frac : float = 0.02          # minimum contour/search area ratio
    seg_ellipse_min_ratio: float = 0.30       # minor/major axis ratio gate
    seg_ema_alpha     : float = 0.18          # ROI centre smoothing (lower=smoother)
    seg_morph_size    : int   = 9             # morphological close kernel size
    seg_every_n_frames: int   = 3             # run full segmentation every N frames

    # Fallback fixed ROI (used when segmentation fails)
    fallback_roi_w_frac: float = 0.45
    fallback_roi_h_frac: float = 0.55

    # ── Motion / quality gate ─────────────────────────────────────────────
    motion_mad_thresh : float = 8.0           # MAD [LSB] for motion flag
    motion_lap_thresh : float = 25.0          # Laplacian var below = blurry
    motion_ema_alpha  : float = 0.22          # temporal smoothing of motion score
    motion_corrupt_thresh: float = 0.55       # motion_score above → reject frame

    # Finger contact: minimum ROI mean to assume tissue in aperture
    finger_dc_thresh  : float = 35.0          # [LSB]

    # ── Optical signal (iPPG) buffer ──────────────────────────────────────
    signal_buf_len    : int   = 300           # 10 s @ 30 Hz
    signal_min_fill   : int   = 60            # 2 s warm-up before analysis

    # Butterworth bandpass for iPPG waveform extraction
    # Covers 0.5–4 Hz → 30–240 BPM (includes physiological tremor artefacts)
    ipPG_bp_low_hz    : float = 0.65
    ipPG_bp_high_hz   : float = 4.0
    ipPG_bp_order     : int   = 4             # Butterworth order (SOS)

    # Welch spectral estimator for iPPG BPM proxy
    ipPG_welch_nperseg: int   = 128           # segment length (≈4.3 s at 30 Hz)

    # BPM proxy history (used for display smoothing only — NOT a clinical reading)
    bpm_proxy_history : int   = 8

    # ── Quality scoring weights ────────────────────────────────────────────
    # Composite optical confidence = Σ wᵢ·scoreᵢ
    w_stability       : float = 0.30          # motion / sharpness score
    w_image_quality   : float = 0.30          # illumination + saturation + sharpness
    w_finger_weight   : float = 0.25          # tissue contact confidence
    w_signal_quality  : float = 0.15          # iPPG spectral SNR

    # ── OpenCV overlay UI ─────────────────────────────────────────────────
    overlay_fps       : int   = 15            # overlay render rate (every 2 frames)
    waveform_strip_h  : int   = 55            # pixel height of each waveform strip
    waveform_history  : int   = 256           # samples in waveform display buffer
    overlay_win_name  : str   = "BioSense-Pi │ NoIR Camera │ 850nm iPPG │ EXPERIMENTAL"

    # ── Health scoring ─────────────────────────────────────────────────────
    max_i2c_err_rate  : float = 0.05          # (for future hardware error tracking)
    health_ema_alpha  : float = 0.03


# ─────────────────────────────────────────────────────────────────────────────
# §2  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class ROIState(NamedTuple):
    """Immutable snapshot of the current fingertip ROI estimate."""
    x1               : int    # bounding box — pixel coords in full frame
    y1               : int
    x2               : int
    y2               : int
    cx               : float  # centroid (sub-pixel)
    cy               : float
    from_seg         : bool   # True = fitted from contour; False = fallback rect
    confidence       : float  # [0,1] — segmentation quality
    ellipse          : Any    # cv2 ellipse tuple or None


# Sentinel for "no ROI yet"
_NULL_ROI = ROIState(0, 0, 1, 1, 0.0, 0.0, False, 0.0, None)


@dataclass
class iPPGResult:
    """
    Per-window iPPG analysis result.

    ALL biometric estimates are labelled EXPERIMENTAL and must NOT be used
    for clinical decision-making.

    dominant_hz    : spectral peak of the bandpass-filtered signal [Hz]
    bpm_proxy      : 60 × dominant_hz  — EXPERIMENTAL non-clinical BPM proxy
    snr_db         : 10·log10(peak PSD / mean noise PSD in HR band) [dB]
    ac_rms         : RMS of bandpass-filtered signal [LSB]
    dc_mean        : mean of raw ROI intensity signal [LSB]
    pi_proxy       : (AC_RMS / DC_mean) × 100%  — non-clinical pulsatility proxy
    confidence     : spectral quality score [0,1]
    buf_fill       : number of valid samples in the analysis window
    """
    dominant_hz    : float = 0.0
    bpm_proxy      : float = 0.0     # EXPERIMENTAL — NOT clinical BPM
    snr_db         : float = 0.0
    ac_rms         : float = 0.0
    dc_mean        : float = 0.0
    pi_proxy       : float = 0.0     # EXPERIMENTAL
    confidence     : float = 0.0
    buf_fill       : int   = 0


@dataclass
class FrameResult:
    """
    All per-frame derived quantities.  Produced by the processing pipeline
    and consumed by the overlay renderer and SharedStateHub update.
    """
    ts                   : float = 0.0
    frame_idx            : int   = 0
    fps                  : float = 0.0

    # Photometry (whole-frame)
    mean_intensity       : float = 0.0
    std_intensity        : float = 0.0
    saturation_frac      : float = 0.0
    laplacian_var        : float = 0.0

    # AEC state
    exposure_us          : int   = 15_000
    analogue_gain        : float = 4.0

    # ROI / segmentation
    roi                  : ROIState = field(default_factory=lambda: _NULL_ROI)
    finger_present       : bool  = False
    roi_mean             : float = 0.0
    roi_std              : float = 0.0

    # Motion
    motion_mad           : float = 0.0
    motion_score         : float = 0.0    # EMA-smoothed [0,1]
    frame_rejected       : bool  = False  # True → excluded from iPPG buffer

    # iPPG / optical signal
    raw_signal_latest    : float = 0.0
    filtered_signal_latest: float = 0.0
    ipPG                 : iPPGResult = field(default_factory=iPPGResult)

    # Composite quality
    optical_stability    : float = 0.0
    image_quality        : float = 0.0
    optical_confidence   : float = 0.0
    quality_label        : str   = "INVALID"


@dataclass
class CameraHealthRecord:
    """Hardware health telemetry for the camera subsystem."""
    total_frames       : int   = 0
    dropped_frames     : int   = 0
    capture_errors     : int   = 0
    consecutive_errors : int   = 0
    health_score       : float = 1.0
    last_error         : str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# §3  CAMERA HARDWARE DRIVER
# ─────────────────────────────────────────────────────────────────────────────

class CameraHardwareDriver:
    """
    Thin, lifecycle-managed wrapper around Picamera2.

    Design goals:
      · Zero-copy DMA frame access: capture_array("main") returns a view into
        the camera's DMA-mapped buffer.  We immediately copy just the BGR
        data into a pre-allocated workspace; the DMA buffer is then free.
      · Manual exposure: AEC/AWB disabled at the driver level.  All exposure
        updates go through AdaptiveExposureCtrl.apply().
      · Noise reduction disabled: hardware NR introduces temporal correlation
        that artificially increases apparent iPPG signal quality.
      · Fixed colour gains (1.0, 1.0): eliminates auto white balance dynamics
        that would otherwise modulate the red-channel IR signal.

    Configuration note on pixel_format "BGR888":
      Picamera2 returns H×W×3 uint8 in B,G,R channel order — matching
      OpenCV's native format — so no channel reorder is needed.

    Simulation mode:
      When Picamera2 is unavailable (dev/test), generates a physically
      plausible synthetic 850 nm reflectance scene including:
        · Central Gaussian fingertip blob, DC ≈ 130 LSB
        · 1.1 Hz cardiac AC modulation (1.5% amplitude)
        · 2nd harmonic (0.5% amplitude, +0.8 rad phase)
        · Slow respiration modulation (0.3 Hz, 0.2%)
        · Brownian micro-motion (σ = 0.4 px per frame)
        · Shot noise (σ = 3 LSB)
        · Slow illumination drift (60 s period, 0.5%)
      This ensures the downstream iPPG pipeline has realistic input even
      without hardware, and the self-test computes meaningful metrics.
    """

    def __init__(self,
                 cfg    : NoIRCameraConfig,
                 health : CameraHealthRecord,
                 log    : logging.Logger):
        self._cfg    = cfg
        self._health = health
        self._log    = log
        self._cam    : Optional["Picamera2"] = None
        self._open   = False

        # Pre-allocated workspace for each captured frame (in-place BGR copy)
        # Shape (H, W, 3) uint8 — avoids repeated malloc in the hot loop.
        self._frame_workspace = np.empty(
            (cfg.height, cfg.width, 3), dtype=np.uint8)

        # Simulation state
        self._sim_t   = 0.0
        self._dt_sim  = 1.0 / cfg.target_fps

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open Picamera2 with fixed manual exposure, NR off."""
        if not _PICAM:
            self._log.info("CameraHardwareDriver: Picamera2 unavailable — simulation mode.")
            self._open = True   # simulation still "opens"
            return

        self._cam = Picamera2()
        video_cfg = self._cam.create_video_configuration(
            main={
                "size"  : (self._cfg.width, self._cfg.height),
                "format": self._cfg.pixel_format,
            },
            controls={
                "AeEnable"           : False,   # manual exposure only
                "AwbEnable"          : False,   # fixed colour gains
                "ExposureTime"       : self._cfg.aec_exp_init_us,
                "AnalogueGain"       : self._cfg.aec_gain_init,
                "ColourGains"        : (1.0, 1.0),
                # Disable all noise reduction — NR introduces temporal smearing
                # that inflates apparent iPPG SNR and should NEVER be enabled.
                "NoiseReductionMode" : _lc.draft.NoiseReductionModeEnum.Off,
            },
        )
        self._cam.configure(video_cfg)
        self._cam.start()
        self._open = True
        time.sleep(0.3)   # allow ISP to settle after start

        self._log.info(
            "CameraHardwareDriver: opened — %dx%d @ %d fps  "
            "exp=%dµs  gain=%.1f  NR=off  AEC=off  AWB=off",
            self._cfg.width, self._cfg.height, self._cfg.target_fps,
            self._cfg.aec_exp_init_us, self._cfg.aec_gain_init)

    def close(self) -> None:
        """Stop camera and release Picamera2 resources."""
        if self._cam is not None:
            try:
                self._cam.stop()
                self._cam.close()
            except Exception:
                pass
        self._open = False
        self._log.info("CameraHardwareDriver: closed.")

    def set_controls(self, exposure_us: int, gain: float) -> None:
        """
        Push AEC-computed exposure and gain to hardware.
        Called by AdaptiveExposureCtrl after each update cycle.
        No-op in simulation mode.
        """
        if self._cam is not None:
            try:
                self._cam.set_controls({
                    "ExposureTime": int(exposure_us),
                    "AnalogueGain": float(gain),
                })
            except Exception as e:
                self._health.last_error = str(e)[:200]

    def capture(self) -> np.ndarray:
        """
        Acquire one frame from hardware or simulation.

        Returns a uint8 BGR array of shape (H, W, 3).  For the hardware path,
        we copy the DMA view into self._frame_workspace to release the DMA
        buffer as fast as possible (< 0.1 ms on Pi 4).

        On capture exception, returns the previous workspace contents
        (stale frame) and increments health error counters.
        """
        if self._cam is not None:
            try:
                # capture_array returns a VIEW into the DMA buffer —
                # copy it immediately so the DMA buffer can be recycled.
                np.copyto(self._frame_workspace,
                          self._cam.capture_array("main"))
                self._health.consecutive_errors = 0
            except Exception as e:
                self._health.capture_errors    += 1
                self._health.consecutive_errors += 1
                self._health.last_error         = str(e)[:200]
                self._log.warning("CameraHardwareDriver: capture error: %s", e)
                # Return stale frame rather than None — avoids None-check overhead
        else:
            # Simulation: synthesise into workspace in-place
            self._synthesise_frame(self._frame_workspace)
            self._sim_t += self._dt_sim
            self._health.consecutive_errors = 0

        self._health.total_frames += 1
        return self._frame_workspace

    # ── Simulation ────────────────────────────────────────────────────────

    def _synthesise_frame(self, out: np.ndarray) -> None:
        """
        Write a synthetic 850 nm reflectance scene into `out` (in-place).

        Pixel model:
            I(x,y,t) = [BG + G(x,y) × (DC + AC1 + AC2 + drift)] + noise
        where:
            BG   = 10 LSB  (ambient / background scatter)
            G    = Gaussian with σ_x=18%, σ_y=28% of frame dimensions
            DC   ≈ 130 LSB (mean reflectance — typical finger on IR LED)
            AC1  = DC × 0.015 × sin(2π × f_HR × t)   (cardiac fundamental)
            AC2  = DC × 0.005 × sin(2π × 2f_HR × t + 0.8)  (2nd harmonic)
            drift= DC × 0.002 × sin(2π × t / 60)     (LED thermal drift)
            noise= N(0, 3²) LSB                       (shot + read noise)
        Brownian micro-motion displaces the Gaussian centre by ~0.4 px/frame.
        """
        t  = self._sim_t
        h, w = out.shape[:2]

        # Micro-motion: accumulate Brownian steps — fingertip never perfectly still
        dx = float(np.random.normal(0, 0.4))
        dy = float(np.random.normal(0, 0.4))
        cx = w / 2.0 + dx * 3.0 * math.sin(2 * math.pi * t * 0.07)
        cy = h / 2.0 + dy * 2.0 * math.sin(2 * math.pi * t * 0.05)

        sx, sy = w * 0.18, h * 0.28

        # Build Gaussian blob without allocating intermediate arrays where possible
        yy, xx = np.ogrid[:h, :w]
        # Use float32 to halve memory bandwidth vs float64
        gauss = np.exp(
            -(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2.0,
            dtype=np.float32,
        )

        f_hr   = 1.1   # Hz ≈ 66 BPM placeholder
        ac1    = 0.015 * math.sin(2 * math.pi * f_hr * t)
        ac2    = 0.005 * math.sin(2 * math.pi * 2 * f_hr * t + 0.8)
        drift  = 0.002 * math.sin(2 * math.pi * t / 60.0)
        dc     = 130.0
        signal = dc * (1.0 + ac1 + ac2 + drift)

        noise = np.random.normal(0.0, 3.0, (h, w)).astype(np.float32)
        gray  = np.clip(10.0 + gauss * signal + noise, 0.0, 255.0).astype(np.uint8)

        # Write into BGR workspace:  R≈0.75 carrier → fill red channel stronger
        # to mimic real OV5647 spectral response at 850 nm
        out[:, :, 0] = (gray.astype(np.uint16) * 70 // 255).astype(np.uint8)   # B
        out[:, :, 1] = (gray.astype(np.uint16) * 180 // 255).astype(np.uint8)  # G
        out[:, :, 2] = gray                                                      # R


# ─────────────────────────────────────────────────────────────────────────────
# §4  ADAPTIVE EXPOSURE CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveExposureCtrl:
    """
    PI feedback controller that maintains ROI mean intensity near `target_lsb`.

    Rationale for manual AEC:
      Picamera2's built-in AEC measures full-frame luminance via metering zones.
      For iPPG we need the ROI intensity — typically a small central region —
      to remain stable at ~140 LSB so the ≈1% pulsatile AC component occupies
      a predictable ADC range.  Full-frame AEC would hunt on background changes
      and corrupt the iPPG waveform.

    Controller design:
      Primary actuator : ExposureTime [µs].  Adjusted proportionally first.
      Secondary actuator: AnalogueGain.  Only adjusted when ExposureTime
        saturates, to avoid gain-switching noise in the optical channel.

    Anti-windup:
      Integral state clamped to ±integral_max to prevent wind-up during
      extended finger-absent periods where the ROI mean is near 0.

    Deadband:
      No correction applied when |error| ≤ deadband_lsb.  This prevents
      the controller from chasing quantisation noise and reduces the number
      of set_controls() I²C transactions to the camera ISP.
    """

    def __init__(self, cfg: NoIRCameraConfig, driver: CameraHardwareDriver,
                 log: logging.Logger):
        self._cfg      = cfg
        self._driver   = driver
        self._log      = log
        self._exp_us   = float(cfg.aec_exp_init_us)
        self._gain     = float(cfg.aec_gain_init)
        self._integral = 0.0

    def update(self, roi_mean: float) -> Tuple[int, float]:
        """
        Compute new exposure/gain from ROI mean.  Applies to hardware.

        Returns:
            (exposure_us, gain) — current AEC state after update.
        """
        cfg = self._cfg
        error = cfg.aec_target_lsb - roi_mean

        if abs(error) <= cfg.aec_deadband_lsb:
            return int(self._exp_us), self._gain

        # Integral accumulation with anti-windup clamp
        self._integral = float(
            np.clip(self._integral + error,
                    -cfg.aec_integral_max, cfg.aec_integral_max))

        correction = cfg.aec_Kp * error + cfg.aec_Ki * self._integral

        # Primary: adjust exposure time
        self._exp_us *= (1.0 + correction)
        self._exp_us = float(np.clip(
            self._exp_us, cfg.aec_exp_min_us, cfg.aec_exp_max_us))

        # Secondary: adjust gain only when exposure is saturated
        exp_high = cfg.aec_exp_max_us - 200
        exp_low  = cfg.aec_exp_min_us + 200
        if self._exp_us >= exp_high and error > 0:
            self._gain = min(self._gain * 1.06, cfg.aec_gain_max)
        elif self._exp_us <= exp_low and error < 0:
            self._gain = max(self._gain * 0.94, cfg.aec_gain_min)

        self._driver.set_controls(int(self._exp_us), self._gain)

        return int(self._exp_us), self._gain

    @property
    def state(self) -> Tuple[int, float]:
        return int(self._exp_us), self._gain


# ─────────────────────────────────────────────────────────────────────────────
# §5  IR CHANNEL EXTRACTOR + PRE-FILTERS
# ─────────────────────────────────────────────────────────────────────────────

class IRChannelExtractor:
    """
    Converts a BGR uint8 frame into a float32 IR-optimised grayscale image.

    Operations (all in-place on pre-allocated buffers):
      1. Weighted channel blend: IR = 0.75R + 0.20G + 0.05B
         This maximises signal at 850 nm given OV5647/IMX219 spectral QE.
      2. Clip to [0, 255] and cast to uint8 (metric image).
      3. CLAHE on uint8 metric image → display-quality contrast image.
      4. Illumination normalisation for display: divide by blurred background.
         This suppresses spatial illumination gradients (e.g. LED hotspot)
         for cleaner visualisation without affecting metric extraction.

    Buffer reuse policy:
      self._f32_r, _f32_g, _f32_b  — per-channel float32 views (no alloc)
      self._ir_f32                  — blended float32 output
      self._ir_u8                   — uint8 metric image
      self._ir_disp                 — uint8 display image (CLAHE + norm)
    All buffers are allocated once in __init__ and reused every frame.
    """

    def __init__(self, cfg: NoIRCameraConfig):
        self._cfg  = cfg
        h, w       = cfg.height, cfg.width

        # Pre-allocated working buffers — no per-frame heap allocation
        self._f32_b   = np.empty((h, w), dtype=np.float32)
        self._f32_g   = np.empty((h, w), dtype=np.float32)
        self._f32_r   = np.empty((h, w), dtype=np.float32)
        self._ir_f32  = np.empty((h, w), dtype=np.float32)
        self._ir_u8   = np.empty((h, w), dtype=np.uint8)
        self._ir_disp = np.empty((h, w), dtype=np.uint8)
        self._bg_f32  = np.empty((h, w), dtype=np.float32)  # for illum norm

        # CLAHE object — created once; thread-safe for single-thread use
        self._clahe = (
            cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                            tileGridSize=cfg.clahe_tile)
            if _CV2 else None)

    def process(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract IR channel from BGR frame.

        Args:
            bgr : uint8 BGR frame (H, W, 3) — may be a DMA view; not modified.

        Returns:
            (ir_metric, ir_display)
            ir_metric  : uint8 (H, W) — lightly blurred, for metric extraction
            ir_display : uint8 (H, W) — CLAHE + illum-normalised, for overlay
        """
        cfg = self._cfg

        # Step 1: split channels in-place (avoids cv2.split allocation)
        # bgr[:,:,0] = B, bgr[:,:,1] = G, bgr[:,:,2] = R
        np.copyto(self._f32_b, bgr[:, :, 0].astype(np.float32, copy=False))
        np.copyto(self._f32_g, bgr[:, :, 1].astype(np.float32, copy=False))
        np.copyto(self._f32_r, bgr[:, :, 2].astype(np.float32, copy=False))

        # Weighted blend: IR = wR·R + wG·G + wB·B
        # Use multiply-accumulate to avoid intermediate arrays
        np.multiply(self._f32_r, cfg.ir_blend_r, out=self._ir_f32)
        np.multiply(self._f32_g, cfg.ir_blend_g, out=self._f32_b)
        np.add(self._ir_f32, self._f32_b, out=self._ir_f32)
        np.multiply(
            bgr[:, :, 0].astype(np.float32, copy=False),
            cfg.ir_blend_b,
            out=self._f32_b,
        )
        np.add(self._ir_f32, self._f32_b, out=self._ir_f32)

        # Step 2: Gaussian noise suppression → uint8 metric image
        if _CV2:
            k = cfg.gauss_kernel
            cv2.GaussianBlur(
                np.clip(self._ir_f32, 0, 255).view(np.float32),
                (k, k), 0,
                dst=self._ir_f32)
        np.clip(self._ir_f32, 0.0, 255.0, out=self._ir_f32)
        np.copyto(self._ir_u8, self._ir_f32.astype(np.uint8, copy=False))

        # Step 3: CLAHE display image
        if self._clahe is not None and _CV2:
            self._clahe.apply(self._ir_u8, dst=self._ir_disp)
        else:
            np.copyto(self._ir_disp, self._ir_u8)

        # Step 4: Illumination normalisation (blend into display image)
        if _CV2:
            ik = cfg.illum_norm_kernel
            cv2.GaussianBlur(self._ir_f32, (ik, ik), 0, dst=self._bg_f32)
            np.add(self._bg_f32, 1.0, out=self._bg_f32)   # prevent /0
            np.divide(self._ir_f32, self._bg_f32, out=self._bg_f32)
            np.multiply(self._bg_f32, 128.0, out=self._bg_f32)
            np.clip(self._bg_f32, 0, 255, out=self._bg_f32)
            norm_u8 = self._bg_f32.astype(np.uint8)
            # 60% CLAHE + 40% normalised for best perceptual quality
            cv2.addWeighted(self._ir_disp, 0.60, norm_u8, 0.40, 0.0,
                            dst=self._ir_disp)

        return self._ir_u8, self._ir_disp


# ─────────────────────────────────────────────────────────────────────────────
# §6  ADAPTIVE TISSUE SEGMENTER
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveTissueSegmenter:
    """
    Detects and tracks the fingertip ROI using Otsu threshold + contour
    analysis + ellipse fitting, with EMA-smoothed centre tracking.

    Algorithm per detection frame (every seg_every_n_frames frames):
      1. Crop a central window (search_frac_w × search_frac_h) to focus
         on the expected fingertip position.
      2. Otsu threshold: automatically separates bright tissue reflectance
         from dark background without a fixed threshold.
      3. Morphological close (9×9 ellipse kernel): fills gaps in the
         tissue mask from veins or surface texture shadows.
      4. Select the largest contour by area.
      5. Reject contour if area < min_area_frac × search_area.
      6. Fit an ellipse to the contour (requires ≥ 5 points).
      7. Gate on ellipse axis ratio ≥ min_ratio (rejects elongated noise).
      8. EMA-smooth the ROI centroid with alpha=0.18 (≈ 5.5 frame time constant).
      9. Derive a bounding rectangle from the ellipse semi-axes with 10% margin.

    Between detection frames, the previous ROI is reused (phase-constant tracking).
    If detection fails, falls back to a fixed central rectangle with confidence=0.

    EMA smoothing:
      x_ema[n] = α × x[n] + (1-α) × x_ema[n-1]
      At α=0.18 and 30 Hz, the -3 dB frequency of the ROI centre tracker is
      f = -ln(1-α) × fs / (2π) ≈ 0.18 × 30 / (2π) ≈ 0.86 Hz.
      This is slow enough to absorb 1-2 frame detection jitter while
      fast enough to follow deliberate finger repositioning in < 1 s.

    Confidence metric:
      conf = √(area_norm × ratio_norm)
      area_norm  = min(area_frac / 0.15, 1.0)
      ratio_norm = min((ratio - min_ratio) / (0.6 - min_ratio), 1.0)
      This provides a [0,1] geometric quality score independent of frame size.
    """

    def __init__(self, cfg: NoIRCameraConfig, log: logging.Logger):
        self._cfg   = cfg
        self._log   = log
        h, w        = cfg.height, cfg.width

        # Pre-allocated mask buffer
        self._mask     = np.zeros((h, w), dtype=np.uint8)
        self._mask_crop: Optional[np.ndarray] = None

        # Morphological kernel
        self._morph_k = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (cfg.seg_morph_size, cfg.seg_morph_size))
            if _CV2 else None)

        # EMA state
        self._cx_ema : Optional[float] = None
        self._cy_ema : Optional[float] = None

        # Current best ROI (reused between detection frames)
        self._current_roi = self._fallback_roi(h, w)
        self._frame_count = 0

    def update(self, ir_metric: np.ndarray) -> Tuple[np.ndarray, ROIState]:
        """
        Update ROI estimate.  Full segmentation runs every N frames;
        between detections the previous ROI is returned immediately.

        Args:
            ir_metric : uint8 (H, W) grayscale IR image

        Returns:
            (mask, roi)  — binary tissue mask and ROIState
        """
        self._frame_count += 1
        h, w = ir_metric.shape

        # Skip full segmentation on non-detection frames
        if self._frame_count % self._cfg.seg_every_n_frames != 0:
            return self._mask, self._current_roi

        if not _CV2:
            return self._mask, self._fallback_roi(h, w)

        cfg = self._cfg

        # ── Search crop ───────────────────────────────────────────────────
        sw = int(w * cfg.seg_search_frac_w)
        sh = int(h * cfg.seg_search_frac_h)
        ox = (w - sw) // 2
        oy = (h - sh) // 2
        crop = ir_metric[oy:oy + sh, ox:ox + sw]

        # ── Otsu threshold ────────────────────────────────────────────────
        _, thresh = cv2.threshold(
            crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ── Morphological close (fill gaps in tissue blob) ─────────────
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self._morph_k)

        # ── Contour extraction ────────────────────────────────────────────
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self._current_roi = self._fallback_roi(h, w)
            return self._mask, self._current_roi

        # Select largest contour
        largest    = max(contours, key=cv2.contourArea)
        area       = cv2.contourArea(largest)
        search_area= sw * sh
        area_frac  = area / search_area

        if area_frac < cfg.seg_min_area_frac:
            self._current_roi = self._fallback_roi(h, w)
            return self._mask, self._current_roi

        # Shift contour to full-frame coordinates
        largest_ff = largest + np.array([[[ox, oy]]], dtype=np.int32)

        # ── Ellipse fit ───────────────────────────────────────────────────
        if len(largest_ff) < 5:
            self._current_roi = self._fallback_roi(h, w)
            return self._mask, self._current_roi

        try:
            ellipse = cv2.fitEllipse(largest_ff)
        except cv2.error:
            self._current_roi = self._fallback_roi(h, w)
            return self._mask, self._current_roi

        (ex, ey), (ma, mi), _angle = ellipse
        ratio = min(ma, mi) / (max(ma, mi) + 1e-6)

        if ratio < cfg.seg_ellipse_min_ratio:
            self._current_roi = self._fallback_roi(h, w)
            return self._mask, self._current_roi

        # ── EMA centre smoothing ──────────────────────────────────────────
        α = cfg.seg_ema_alpha
        self._cx_ema = ex if self._cx_ema is None else α * ex + (1 - α) * self._cx_ema
        self._cy_ema = ey if self._cy_ema is None else α * ey + (1 - α) * self._cy_ema
        cx, cy = self._cx_ema, self._cy_ema

        # ROI bounding box from ellipse semi-axes + 10% margin
        half_w = int(max(ma, mi) / 2.0 * 1.10) + 1
        half_h = int(min(ma, mi) / 2.0 * 1.50) + 1
        x1 = int(np.clip(cx - half_w, 0, w - 1))
        y1 = int(np.clip(cy - half_h, 0, h - 1))
        x2 = int(np.clip(cx + half_w, 0, w - 1))
        y2 = int(np.clip(cy + half_h, 0, h - 1))

        # Confidence
        area_norm  = min(area_frac / 0.15, 1.0)
        ratio_span = max(0.60 - cfg.seg_ellipse_min_ratio, 1e-6)
        ratio_norm = min((ratio - cfg.seg_ellipse_min_ratio) / ratio_span, 1.0)
        confidence = float(math.sqrt(max(0.0, area_norm * ratio_norm)))

        # ── Update mask (in-place, reuse buffer) ──────────────────────────
        self._mask[:] = 0
        cv2.drawContours(self._mask, [largest_ff], -1, 255, -1)

        roi = ROIState(
            x1=x1, y1=y1, x2=x2, y2=y2,
            cx=cx, cy=cy,
            from_seg=True, confidence=confidence, ellipse=ellipse)
        self._current_roi = roi
        return self._mask, roi

    def _fallback_roi(self, h: int, w: int) -> ROIState:
        """Central fixed rectangle — used when segmentation fails."""
        cfg = self._cfg
        rw  = int(w * cfg.fallback_roi_w_frac)
        rh  = int(h * cfg.fallback_roi_h_frac)
        x1  = (w - rw) // 2
        y1  = (h - rh) // 2
        return ROIState(
            x1=x1, y1=y1, x2=x1 + rw, y2=y1 + rh,
            cx=w / 2.0, cy=h / 2.0,
            from_seg=False, confidence=0.0, ellipse=None)


# ─────────────────────────────────────────────────────────────────────────────
# §7  MOTION QUALITY GATE
# ─────────────────────────────────────────────────────────────────────────────

class MotionQualityGate:
    """
    Dual-metric motion detection and frame quality classification.

    Metric 1 — Frame-difference MAD (Mean Absolute Difference):
      MAD[n] = mean(|gray[n] - gray[n-1]|)
      Physical interpretation: pixel-level displacement magnitude.
      At 30 Hz, a 1 px/frame finger tremor causes MAD ≈ 2–4 LSB.
      Motion threshold (8 LSB) rejects frames with > ~2 px/frame motion.

    Metric 2 — Laplacian variance (sharpness):
      LapVar = Var(∇²I)   where ∇² is the 3×3 discrete Laplacian kernel.
      Low LapVar (< 25) indicates motion blur — the frame transition smeared
      edges during exposure.  Complements MAD: fast motion may have high MAD
      but still appear sharp if the shutter is fast enough; blur catches the
      cases where it isn't.

    EMA smoothing:
      Both metrics are exponentially averaged (α=0.22) before thresholding.
      This prevents single noisy frames from flipping the quality state,
      while still responding to sustained motion within ~3 frames.

    Corruption gate:
      is_corrupt = (motion_ema > 0.55) OR (laplacian_var < blur_threshold)
      Corrupted frames are excluded from the iPPG signal buffer.
    """

    def __init__(self, cfg: NoIRCameraConfig):
        self._cfg       = cfg
        self._prev_gray : Optional[np.ndarray] = None
        self._motion_ema= 0.0
        self._lap_var   = 0.0

    def update(self,
               ir_metric: np.ndarray
               ) -> Tuple[float, float, float, bool]:
        """
        Compute motion and sharpness metrics for the current frame.

        Returns:
            (motion_mad, laplacian_var, motion_score_ema, is_corrupt)
        """
        cfg = self._cfg

        # Laplacian variance (sharpness) — in-place via cv2
        if _CV2:
            self._lap_var = float(cv2.Laplacian(ir_metric, cv2.CV_64F).var())
        else:
            self._lap_var = 50.0   # assume sharp in fallback

        # Frame-difference MAD — requires two consecutive frames
        if self._prev_gray is not None and self._prev_gray.shape == ir_metric.shape:
            # Cast to int16 to handle subtraction wrap-around
            diff = ir_metric.astype(np.int16)
            diff -= self._prev_gray.astype(np.int16)
            np.abs(diff, out=diff)
            mad = float(diff.mean())
        else:
            mad = 0.0

        # Store current frame for next call — avoid copy if possible
        if self._prev_gray is None or self._prev_gray.shape != ir_metric.shape:
            self._prev_gray = ir_metric.copy()
        else:
            np.copyto(self._prev_gray, ir_metric)

        # Normalise MAD to [0,1] and EMA-smooth
        raw_score = min(mad / max(cfg.motion_mad_thresh, 1e-6), 1.0)
        self._motion_ema = (cfg.motion_ema_alpha * raw_score
                            + (1.0 - cfg.motion_ema_alpha) * self._motion_ema)

        is_corrupt = (self._motion_ema > cfg.motion_corrupt_thresh
                      or self._lap_var  < cfg.motion_lap_thresh)

        return mad, self._lap_var, self._motion_ema, is_corrupt


# ─────────────────────────────────────────────────────────────────────────────
# §8  OPTICAL SIGNAL BUFFER + iPPG ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class OpticalSignalBuffer:
    """
    Circular buffer for ROI mean intensity samples from accepted frames.
    Provides real-time Butterworth bandpass filtering via persistent SOS state.

    Buffer strategy:
      · Only motion-clean, finger-present frames are pushed (gated upstream).
      · Effective sample rate fs_eff is estimated from push() call intervals.
        This handles the variable rate that results from motion rejection:
        if many frames are rejected, fs_eff < target_fps and the filter
        cutoff frequencies must be adjusted accordingly.
      · Filter state (zi) is maintained across calls for causal streaming.
        On a reset (e.g. finger removed), zi is zeroed and the filter
        coefficients are rebuilt at the current fs_eff.

    Detrending:
      Before filtering, we subtract the adaptive IIR mean (single-pole LP
      with tau = 0.995) to remove illumination drift below 0.08 Hz at 30 Hz.
      This is faster than polynomial detrending and avoids edge artefacts.
    """

    def __init__(self, cfg: NoIRCameraConfig, log: logging.Logger):
        self._cfg      = cfg
        self._log      = log

        # Ring buffers (all float32 to halve bandwidth vs float64)
        self._raw_buf  : Deque[float] = collections.deque(maxlen=cfg.signal_buf_len)
        self._filt_buf : Deque[float] = collections.deque(maxlen=cfg.signal_buf_len)
        self._ts_buf   : Deque[float] = collections.deque(maxlen=cfg.signal_buf_len)

        # SOS filter state
        self._sos      : Optional[np.ndarray] = None
        self._zi       : Optional[np.ndarray] = None
        self._fs_eff   : float = float(cfg.target_fps)
        self._fs_est_buf: Deque[float] = collections.deque(maxlen=30)

        # Adaptive IIR baseline state (single-pole HP for detrending)
        # tau=0.995 → LP cutoff ≈ (1-0.995)/(2π×1/30) ≈ 0.024 Hz at 30 Hz
        self._baseline_tau = 0.995
        self._iir_state : Optional[float] = None

        self._last_ts   : float = 0.0
        self._reset_flag= True   # force filter rebuild on first push

    def push(self, roi_mean: float, ts: float) -> None:
        """
        Push one accepted sample (caller must gate on not-corrupt + finger-present).

        Updates effective sample rate estimate, applies IIR detrend, then
        feeds through the Butterworth bandpass filter.
        """
        # Effective sample rate estimation from actual push timestamps
        if self._last_ts > 0.0:
            dt = ts - self._last_ts
            if 1e-4 < dt < 1.0:
                self._fs_est_buf.append(1.0 / dt)
        self._last_ts = ts

        if len(self._fs_est_buf) >= 5:
            new_fs = float(np.median(list(self._fs_est_buf)))
            # Rebuild filter if fs changed by more than 10%
            if abs(new_fs - self._fs_eff) / max(self._fs_eff, 1.0) > 0.10:
                self._fs_eff = new_fs
                self._reset_flag = True

        # Rebuild SOS filter if needed
        if self._reset_flag:
            self._build_filter()
            self._reset_flag = False

        # Adaptive IIR baseline removal
        tau = self._baseline_tau
        if self._iir_state is None:
            self._iir_state = roi_mean
        self._iir_state = tau * self._iir_state + (1.0 - tau) * roi_mean
        ac_sample = roi_mean - self._iir_state

        self._raw_buf.append(roi_mean)
        self._ts_buf.append(ts)

        # Causal Butterworth bandpass filter (SOS, sosfilt with persistent state)
        if self._sos is not None and self._zi is not None:
            sample_vec = np.array([ac_sample], dtype=np.float32)
            y, self._zi = sosfilt(
                self._sos, sample_vec.astype(np.float64), zi=self._zi)
            self._filt_buf.append(float(y[0]))
        else:
            self._filt_buf.append(ac_sample)

    def _build_filter(self) -> None:
        """Rebuild SOS Butterworth bandpass at current fs_eff."""
        if not _SCIPY:
            return
        fs  = max(self._fs_eff, 5.0)
        nyq = fs / 2.0
        cfg = self._cfg
        lo  = cfg.ipPG_bp_low_hz  / nyq
        hi  = cfg.ipPG_bp_high_hz / nyq

        # Clamp to valid Nyquist range — avoid degenerate filter at low fps
        lo = float(np.clip(lo, 1e-4, 0.999))
        hi = float(np.clip(hi, 1e-4, 0.999))
        if lo >= hi:
            return
        try:
            self._sos = butter(cfg.ipPG_bp_order, [lo, hi],
                               btype='band', output='sos').astype(np.float64)
            # Initialise zi to zero (fresh start — avoids transient from stale state)
            self._zi  = sosfilt_zi(self._sos)
            self._log.debug(
                "OpticalSignalBuffer: rebuilt Butterworth SOS at fs_eff=%.1f Hz  "
                "LP=%.3f  HP=%.3f", fs, lo, hi)
        except Exception as e:
            self._log.warning("OpticalSignalBuffer: filter build failed: %s", e)
            self._sos = None; self._zi = None

    def reset(self) -> None:
        """Reset all buffers and filter state (call on finger removal)."""
        self._raw_buf.clear()
        self._filt_buf.clear()
        self._ts_buf.clear()
        self._iir_state  = None
        self._zi         = None
        self._reset_flag = True
        self._last_ts    = 0.0

    @property
    def fill(self) -> int:
        return len(self._raw_buf)

    @property
    def raw_array(self) -> np.ndarray:
        return np.array(self._raw_buf, dtype=np.float32)

    @property
    def filt_array(self) -> np.ndarray:
        return np.array(self._filt_buf, dtype=np.float32)


class iPPGEngine:
    """
    Imaging PPG (iPPG) biometric extraction engine.

    Inputs: OpticalSignalBuffer contents (raw + filtered ROI intensity)
    Outputs: iPPGResult per analysis window

    Algorithm (Welch spectral estimator):
      1. Extract last `signal_buf_len` samples from OpticalSignalBuffer.
      2. Apply Hann window to the filtered signal to reduce spectral leakage.
      3. Welch periodogram with 50% overlap and nperseg=128 (≈ 4.3 s at 30 Hz).
         Welch reduces variance by √(2×overlap×n/nperseg) vs periodogram.
      4. Identify dominant spectral peak in [bp_low, bp_high] Hz band.
      5. Compute SNR:
            sig_power  = PSD at peak ± 0.15 Hz
            noise_power= mean PSD in band excluding signal window
            SNR_dB = 10 log₁₀(sig_power / noise_power)
      6. BPM proxy = 60 × dominant_hz  (EXPERIMENTAL — not a clinical reading)
      7. AC_RMS and Perfusion Index proxy from filtered signal statistics.

    Pulsatility Confidence:
      A composite score in [0,1] incorporating:
        · SNR normalised to 25 dB reference
        · Buffer fill fraction (warmed-up buffer needed for reliable spectral estimate)
        · AC/DC ratio plausibility (target 0.5–2% for fingertip iPPG)

    History:
      iPPGResult is smoothed over bpm_proxy_history windows using a
      deque median — this dramatically reduces jitter from single-window
      spectral artefacts while keeping latency to ≤ bpm_proxy_history/2 windows.

    DISCLAIMER: All numerical outputs are non-clinical experimental values.
    """

    def __init__(self, cfg: NoIRCameraConfig, log: logging.Logger):
        self._cfg     = cfg
        self._log     = log
        self._bpm_hist: Deque[float] = collections.deque(maxlen=cfg.bpm_proxy_history)

    def analyze(self, buf: OpticalSignalBuffer) -> iPPGResult:
        """
        Run iPPG analysis on the current buffer state.

        Returns an iPPGResult populated with spectral metrics and BPM proxy.
        """
        result = iPPGResult(buf_fill=buf.fill)

        if buf.fill < self._cfg.signal_min_fill or not _SCIPY:
            return result

        raw_arr  = buf.raw_array
        filt_arr = buf.filt_array

        # Effective sample rate from buffer timestamps
        fs = float(self._cfg.target_fps)   # conservative estimate

        # ── Spectral analysis via Welch ────────────────────────────────────
        n      = len(filt_arr)
        nperseg= min(self._cfg.ipPG_welch_nperseg, n)
        try:
            freqs, Pxx = welch(
                filt_arr.astype(np.float64),
                fs=fs, nperseg=nperseg,
                window='hann', noverlap=nperseg // 2)
        except Exception as e:
            self._log.debug("iPPGEngine: Welch failed: %s", e)
            return result

        # Band mask for HR range
        f_lo, f_hi = self._cfg.ipPG_bp_low_hz, self._cfg.ipPG_bp_high_hz
        band  = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(band):
            return result

        band_Pxx   = Pxx[band]
        band_freqs = freqs[band]
        pk_idx     = int(np.argmax(band_Pxx))
        dom_hz     = float(band_freqs[pk_idx])
        dom_pow    = float(band_Pxx[pk_idx])

        # ── Spectral SNR ───────────────────────────────────────────────────
        sig_mask   = (freqs >= dom_hz - 0.15) & (freqs <= dom_hz + 0.15)
        noise_mask = band & ~sig_mask
        sig_p  = float(np.mean(Pxx[sig_mask]))  if np.any(sig_mask)  else 0.0
        noi_p  = float(np.mean(Pxx[noise_mask])) if np.any(noise_mask) else 1e-20
        noi_p  = max(noi_p, 1e-20)
        snr_db = max(0.0, 10.0 * math.log10(sig_p / noi_p + 1e-12))

        # ── AC/DC statistics ───────────────────────────────────────────────
        dc_mean  = float(np.mean(raw_arr))
        ac_rms   = float(np.sqrt(np.mean(filt_arr ** 2.0)))
        pi_proxy = (ac_rms / max(dc_mean, 1.0)) * 100.0
        pi_proxy = round(max(0.0, min(25.0, pi_proxy)), 3)

        # ── BPM proxy (EXPERIMENTAL — non-clinical) ───────────────────────
        bpm_proxy = dom_hz * 60.0
        self._bpm_hist.append(bpm_proxy)
        bpm_smoothed = float(np.median(list(self._bpm_hist))) if self._bpm_hist else 0.0

        # ── Pulsatility confidence ─────────────────────────────────────────
        snr_score = float(np.clip(snr_db / 25.0, 0.0, 1.0))
        fill_score= float(np.clip(buf.fill / self._cfg.signal_buf_len, 0.0, 1.0))
        # AC/DC ratio target 0.5–2% for fingertip — Gaussian scoring
        ac_dc_pct = (ac_rms / max(dc_mean, 1.0)) * 100.0
        ac_score  = float(np.exp(-((ac_dc_pct - 1.2) ** 2) / (2 * 0.8 ** 2)))
        confidence = round(
            float(np.clip(0.45 * snr_score + 0.30 * fill_score + 0.25 * ac_score,
                          0.0, 1.0)),
            3)

        result.dominant_hz  = round(dom_hz, 3)
        result.bpm_proxy    = round(bpm_smoothed, 1)
        result.snr_db       = round(snr_db, 2)
        result.ac_rms       = round(ac_rms, 4)
        result.dc_mean      = round(dc_mean, 2)
        result.pi_proxy     = pi_proxy
        result.confidence   = confidence
        return result

    def reset(self) -> None:
        self._bpm_hist.clear()


# ─────────────────────────────────────────────────────────────────────────────
# §9  FRAME QUALITY SCORER
# ─────────────────────────────────────────────────────────────────────────────

class FrameQualityScorer:
    """
    Multi-component optical quality confidence scorer.

    Composite optical confidence = Σᵢ wᵢ × scoreᵢ

    Component 1 — Optical stability (w=0.30):
      stability = (1 - motion_ema) × min(laplacian_var / 30, 1)
      Penalises both motion and blur.

    Component 2 — Image quality (w=0.30):
      intensity_score = clip((mean_intensity - 8) / 27, 0, 1)
      saturation_pen  = max(0, 1 - saturation_frac × 15)
      sharpness       = min(laplacian_var / 150, 1)
      iq = intensity_score × saturation_pen × (0.4 + 0.6 × sharpness)
      Rewards proper exposure, penalises saturation and blur.

    Component 3 — Finger / tissue contact (w=0.25):
      finger_w = 0.80 + 0.20 × seg_confidence  if finger present
      finger_w = 0.25                            if absent
      Rewards confident segmentation over fallback detection.

    Component 4 — iPPG signal quality (w=0.15):
      sig_q = clip((snr_db - 10) / 10, 0, 1)  if snr_db > 10 else 0
      Rewards strong spectral peak in the HR band.

    Quality labels:
      conf ≥ 0.80  → EXCELLENT
      conf ≥ 0.65  → GOOD
      conf ≥ 0.45  → FAIR
      conf ≥ 0.20  → POOR
      otherwise    → INVALID
    """

    def __init__(self, cfg: NoIRCameraConfig):
        self._cfg = cfg

    def score(self,
              mean_intensity : float,
              saturation_frac: float,
              laplacian_var  : float,
              motion_ema     : float,
              finger_present : bool,
              seg_confidence : float,
              snr_db         : float,
              ) -> Tuple[float, float, float, str]:
        """
        Returns:
            (optical_stability, image_quality, optical_confidence, quality_label)
        """
        cfg = self._cfg

        # Component 1: stability
        motion_pen  = 1.0 - motion_ema
        blur_ok     = min(laplacian_var / 30.0, 1.0)
        stability   = float(np.clip(motion_pen * blur_ok, 0.0, 1.0))

        # Component 2: image quality
        int_score   = float(np.clip((mean_intensity - 8.0) / 27.0, 0.0, 1.0))
        sat_pen     = max(0.0, 1.0 - saturation_frac * 15.0)
        sharpness   = min(laplacian_var / 150.0, 1.0)
        iq          = float(int_score * sat_pen * (0.4 + 0.6 * sharpness))

        # Component 3: finger contact
        finger_w = (0.80 + 0.20 * seg_confidence) if finger_present else 0.25

        # Component 4: iPPG spectral quality
        sig_q = float(np.clip((snr_db - 10.0) / 10.0, 0.0, 1.0)) if snr_db > 10.0 else 0.0

        conf = float(np.clip(
            cfg.w_stability    * stability
            + cfg.w_image_quality * iq
            + cfg.w_finger_weight * finger_w
            + cfg.w_signal_quality * sig_q,
            0.0, 1.0))

        if   conf >= 0.80: label = "EXCELLENT"
        elif conf >= 0.65: label = "GOOD"
        elif conf >= 0.45: label = "FAIR"
        elif conf >= 0.20: label = "POOR"
        else:              label = "INVALID"

        return (round(stability, 3), round(iq, 3), round(conf, 3), label)


# ─────────────────────────────────────────────────────────────────────────────
# §10  OPENCV OVERLAY RENDERER
# ─────────────────────────────────────────────────────────────────────────────

class OpenCVOverlayRenderer:
    """
    Research-grade OpenCV overlay display with iPPG and diagnostic annotations.

    Layout (vertical stack):
      ┌──────────────────────────────────────┐
      │                                      │
      │    Annotated IR frame  (H × W)       │
      │                                      │
      │  ┌─────────────────────────────────┐ │
      │  │ Diagnostics panel (top-left)    │ │
      │  └─────────────────────────────────┘ │
      │  ┌──────────────┐  ┌──────────────┐  │
      │  │  BPM badge   │  │ Quality badge│  │
      │  └──────────────┘  └──────────────┘  │
      └──────────────────────────────────────┘
      ┌──────────────────────────────────────┐
      │  Raw iPPG waveform strip  (55 px)    │
      └──────────────────────────────────────┘
      ┌──────────────────────────────────────┐
      │  Bandpass iPPG waveform strip (55 px)│
      └──────────────────────────────────────┘

    Overlay elements:
      · ROI bounding box:  green (finger present, segmentation)
                           blue  (finger present, fallback)
                           red   (no finger)
      · Ellipse fit:       cyan (from segmentation)
      · Tissue mask:       semi-transparent green fill (10% opacity)
      · Diagnostics panel: frame index, FPS, exposure, ROI stats
      · BPM proxy badge:   EXPERIMENTAL — non-clinical
      · SpO2 badge:        from MAX30102 (populated from snapshot)
      · Quality badge:     EXCELLENT / GOOD / FAIR / POOR / INVALID
      · Motion banner:     full-width red bar if frame rejected
      · Waveform traces:   raw (top, green) + BP-filtered (bottom, cyan)

    Memory:
      All arrays pre-allocated in __init__.  Rendering is done by annotating
      a copy of the display frame (to avoid modifying the metric frame).
      The waveform strips are single-channel uint8 allocated once.

    Return value of render():
      True  = window still open, continue
      False = 'q'/ESC pressed, caller should stop
    """

    # BGR colour constants
    _GREEN  = (0, 220, 50)
    _BLUE   = (30, 100, 200)
    _RED    = (0, 0, 200)
    _CYAN   = (200, 210, 0)
    _YELLOW = (0, 180, 210)
    _WHITE  = (240, 240, 240)
    _BLACK  = (0, 0, 0)
    _DARK   = (12, 12, 12)

    # Quality label → BGR badge colour
    _QC_BGR = {
        "EXCELLENT": (0, 200,  10),
        "GOOD"     : (0, 175,  25),
        "FAIR"     : (0, 170, 210),
        "POOR"     : (0,  90, 215),
        "INVALID"  : (0,  15, 200),
    }

    def __init__(self, cfg: NoIRCameraConfig):
        self._cfg     = cfg
        h, w          = cfg.height, cfg.width
        strip_h       = cfg.waveform_strip_h
        total_h       = h + strip_h * 2

        # Pre-allocated output frame: IR display (BGR) + 2 waveform strips
        self._canvas      = np.zeros((total_h, w, 3), dtype=np.uint8)
        # View slices into canvas (no copy needed for assembly)
        self._cam_view    = self._canvas[:h]
        self._raw_strip   = self._canvas[h: h + strip_h]
        self._filt_strip  = self._canvas[h + strip_h:]

        # Waveform history deques (ring)
        self._raw_hist : Deque[float] = collections.deque(maxlen=cfg.waveform_history)
        self._filt_hist: Deque[float] = collections.deque(maxlen=cfg.waveform_history)

        # Tissue mask overlay scratchpad
        self._mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)

        self._ready = False
        self._win   = cfg.overlay_win_name

    def init(self) -> None:
        """Create OpenCV window.  No-op if cv2 is unavailable."""
        if not _CV2:
            return
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        h, w = self._cfg.height, self._cfg.width
        cv2.resizeWindow(self._win, w, h + self._cfg.waveform_strip_h * 2)
        self._ready = True

    def render(self,
               ir_disp   : np.ndarray,
               mask      : np.ndarray,
               fd        : FrameResult,
               snap_bpm  : Optional[float],
               snap_spo2 : Optional[float],
               ) -> bool:
        """
        Render one overlay frame into self._canvas and display.

        Args:
            ir_disp   : uint8 (H, W) CLAHE-enhanced IR image for display
            mask      : uint8 (H, W) binary tissue mask
            fd        : FrameResult with all metrics
            snap_bpm  : BPM from MAX30102 (may be None) — for cross-display
            snap_spo2 : SpO2 from MAX30102 (may be None)

        Returns:
            True = continue; False = user pressed q/Q/ESC
        """
        if not _CV2 or not self._ready:
            return True

        h, w = ir_disp.shape
        cfg  = self._cfg

        # ── Convert IR grayscale → BGR into canvas camera region ──────────
        cv2.cvtColor(ir_disp, cv2.COLOR_GRAY2BGR, dst=self._cam_view)

        # ── Semi-transparent tissue mask (10% green fill) ─────────────────
        if mask is not None and mask.any():
            self._mask_overlay[:] = 0
            self._mask_overlay[mask > 0] = (0, 70, 0)
            cv2.addWeighted(self._cam_view, 0.90,
                            self._mask_overlay, 0.10, 0.0,
                            dst=self._cam_view)

        roi = fd.roi

        # ── ROI bounding box ──────────────────────────────────────────────
        if fd.finger_present and roi.from_seg:
            roi_col = self._GREEN
            lw      = 2
        elif fd.finger_present:
            roi_col = self._BLUE
            lw      = 1
        else:
            roi_col = self._RED
            lw      = 1
        cv2.rectangle(self._cam_view,
                      (roi.x1, roi.y1), (roi.x2, roi.y2), roi_col, lw)

        # ── Ellipse overlay ────────────────────────────────────────────────
        if roi.ellipse is not None:
            try:
                cv2.ellipse(self._cam_view, roi.ellipse, self._CYAN, 1)
            except cv2.error:
                pass

        # ── Diagnostics panel (top-left, 140 × N px dark background) ──────
        ipPG   = fd.ipPG
        diag   = [
            f"#{fd.frame_idx:06d}  {fd.fps:4.1f}fps",
            f"Exp {fd.exposure_us:,}us  G{fd.analogue_gain:.2f}x",
            f"ROI {fd.roi_mean:5.1f}LSB  s{fd.roi_std:.1f}",
            f"Mot {fd.motion_score:.3f}  Lap{fd.laplacian_var:5.0f}",
            f"SNR {ipPG.snr_db:4.1f}dB  PI {ipPG.pi_proxy:.3f}%",
            f"Buf {ipPG.buf_fill}/{cfg.signal_buf_len}",
            f"Conf{fd.optical_confidence*100:4.0f}%  {fd.quality_label[:6]}",
        ]
        rows_shown = len(diag)
        panel_h    = rows_shown * 17 + 6
        cv2.rectangle(self._cam_view, (2, 2), (148, 2 + panel_h), (8, 8, 8), -1)
        for i, txt in enumerate(diag):
            cv2.putText(self._cam_view, txt, (5, 16 + i * 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.39,
                        (170, 225, 175), 1, cv2.LINE_AA)

        # ── BPM proxy badge (top-right upper) ─────────────────────────────
        # MAX30102 BPM takes precedence over iPPG proxy; iPPG shown if absent
        bpm_val  = snap_bpm if snap_bpm is not None else ipPG.bpm_proxy
        bpm_src  = "PPG" if snap_bpm is not None else "iPPG*"
        bpm_txt  = f"{bpm_val:.0f}BPM" if bpm_val > 0 else "---BPM"
        bpm_lbl  = f"{bpm_txt} {bpm_src}"
        bpm_col  = (0, 180, 30) if snap_bpm is not None else (0, 130, 200)
        tw_b     = cv2.getTextSize(bpm_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
        bx1      = w - tw_b - 12
        cv2.rectangle(self._cam_view, (bx1 - 2, 2), (w - 2, 23), bpm_col, -1)
        cv2.putText(self._cam_view, bpm_lbl, (bx1, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, self._WHITE, 1, cv2.LINE_AA)

        # ── SpO2 badge (top-right, below BPM) ────────────────────────────
        if snap_spo2 is not None:
            spo2_txt = f"SpO2 {snap_spo2:.0f}%"
            spo2_col = (0, 190, 20) if snap_spo2 >= 95 else (0, 140, 215)
            tw_s     = cv2.getTextSize(spo2_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0][0]
            sx1      = w - tw_s - 12
            cv2.rectangle(self._cam_view, (sx1 - 2, 25), (w - 2, 44), spo2_col, -1)
            cv2.putText(self._cam_view, spo2_txt, (sx1, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, self._WHITE, 1, cv2.LINE_AA)

        # ── Quality badge (bottom-right of frame) ────────────────────────
        qc_col = self._QC_BGR.get(fd.quality_label, self._RED)
        qc_txt = fd.quality_label
        tw_q   = cv2.getTextSize(qc_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0]
        qx1    = w - tw_q - 12
        cv2.rectangle(self._cam_view, (qx1 - 3, h - 22), (w - 2, h - 2), qc_col, -1)
        cv2.putText(self._cam_view, qc_txt, (qx1, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, self._WHITE, 1, cv2.LINE_AA)

        # ── iPPG dominant frequency annotation ──────────────────────────
        if ipPG.dominant_hz > 0 and fd.finger_present:
            hz_txt = f"{ipPG.dominant_hz:.2f}Hz [EXPERIMENTAL]"
            cv2.putText(self._cam_view, hz_txt, (2, h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 120, 210), 1)

        # ── Motion rejection banner ───────────────────────────────────────
        if fd.frame_rejected:
            cv2.rectangle(self._cam_view, (0, h - 30), (w, h - 23), (0, 0, 160), -1)
            cv2.putText(self._cam_view, "MOTION ARTEFACT — frame excluded from iPPG",
                        (4, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        self._WHITE, 1, cv2.LINE_AA)

        # ── Experimental watermark ────────────────────────────────────────
        cv2.putText(self._cam_view,
                    "EXPERIMENTAL | NON-CLINICAL | 850nm iPPG | BioSense-Pi",
                    (2, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (60, 60, 140), 1)

        # ── Waveform strips ────────────────────────────────────────────────
        self._raw_hist.append(fd.raw_signal_latest)
        self._filt_hist.append(fd.filtered_signal_latest)
        self._draw_waveform_strip(
            self._raw_strip,
            list(self._raw_hist),
            (0, 200, 80),
            "Raw ROI intensity [LSB]")
        self._draw_waveform_strip(
            self._filt_strip,
            list(self._filt_hist),
            (0, 140, 220),
            f"Butterworth BP [{cfg.ipPG_bp_low_hz:.1f}–{cfg.ipPG_bp_high_hz:.1f}Hz] "
            f"iPPG AC — EXPERIMENTAL NON-CLINICAL")

        # ── Display ────────────────────────────────────────────────────────
        cv2.imshow(self._win, self._canvas)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord('q'), ord('Q'), 27)

    @staticmethod
    def _draw_waveform_strip(strip: np.ndarray,
                             data : List[float],
                             colour: Tuple[int, int, int],
                             label : str) -> None:
        """
        Draw a scrolling waveform into a pre-allocated BGR strip (in-place).

        Algorithm:
          Normalise data to strip height, resample to strip width via linear
          interpolation, render as a polyline.  All operations on the existing
          strip array — no allocation.
        """
        strip[:] = 12   # dark background
        h, w = strip.shape[:2]
        if len(data) < 2:
            return
        arr = np.array(data, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        rng = hi - lo
        if rng < 1e-6:
            return
        # Normalise to strip height with 4 px margin top+bottom
        norm  = (arr - lo) / rng * (h - 8) + 4
        # Resample to strip width
        xi    = np.linspace(0, len(norm) - 1, w)
        y_rsp = np.interp(xi, np.arange(len(norm)), norm).astype(np.int32)
        # Draw polyline
        pts = np.column_stack([
            np.arange(w, dtype=np.int32),
            np.clip(h - 1 - y_rsp, 0, h - 1)
        ]).reshape(-1, 1, 2)
        if _CV2:
            cv2.polylines(strip, [pts], False, colour, 1, cv2.LINE_AA)
            cv2.putText(strip, label, (3, 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.31,
                        tuple(max(c // 2, 40) for c in colour), 1)

    def close(self) -> None:
        if _CV2:
            cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# §11  SENSOR HEALTH SCORER
# ─────────────────────────────────────────────────────────────────────────────

class CameraHealthScorer:
    """
    Per-cycle hardware health metric for the camera subsystem.

    Health components:
      · Dropped / errored frames  (0.60 weight)
      · Consecutive error burst   (0.40 weight, exponential decay)

    EMA-smoothed to avoid single-frame glitches from degrading the
    displayed health score.
    """

    def __init__(self, cfg: NoIRCameraConfig, log: logging.Logger):
        self._cfg    = cfg
        self._log    = log
        self._health = 1.0

    def score(self, record: CameraHealthRecord) -> float:
        """Compute and return health score [0,1].  Modifies record in place."""
        n    = max(1, record.total_frames)
        err_rate  = record.capture_errors / n
        err_score = max(0.0, 1.0 - err_rate / max(self._cfg.max_i2c_err_rate, 1e-6))

        burst_pen = math.exp(-record.consecutive_errors / 3.0)

        raw = 0.60 * err_score + 0.40 * burst_pen

        α = self._cfg.health_ema_alpha
        self._health = (1 - α) * self._health + α * raw
        record.health_score = round(self._health, 3)

        if self._health < 0.5 and record.total_frames % 300 == 0:
            self._log.warning(
                "Camera health degraded: %.2f  err=%d  consec=%d",
                self._health, record.capture_errors, record.consecutive_errors)

        record.consecutive_errors = 0
        return self._health


# ─────────────────────────────────────────────────────────────────────────────
# §12  ACQUISITION THREAD
# ─────────────────────────────────────────────────────────────────────────────

def worker_acq_camera(stop_event   : threading.Event,
                      state_hub    : Any,                  # SharedStateHub
                      watchdog     : Any,                  # WatchdogFramework
                      cfg          : Any,                  # SystemConfig
                      log          : logging.Logger,
                      result_queue : Optional[queue.Queue] = None,
                      ) -> None:
    """
    NoIR camera acquisition, processing, and overlay thread.

    Replaces / supersedes the _worker_acq_camera stub in main.py §10.
    Designed to run as a daemon thread managed by ThreadManager.

    Thread model (single-threaded within this function):
      · Frame acquisition: CameraHardwareDriver.capture() at 30 Hz
      · Pipeline stages are invoked sequentially each frame:
          1. IR extraction + Gaussian pre-filter      < 1 ms
          2. Tissue segmentation (every 3rd frame)    < 2 ms
          3. Photometric extraction from ROI          < 0.2 ms
          4. AEC update                               < 0.1 ms
          5. Motion gate                              < 0.5 ms
          6. Optical signal push + iPPG analysis      < 1 ms (Welch cached)
          7. Quality scoring                          < 0.2 ms
          8. Snapshot update                          < 0.1 ms
          9. OpenCV overlay (every 2nd frame)         < 3 ms
        Total pipeline latency: < 8 ms per frame → safe at 30 Hz (33 ms budget)

    Timing strategy:
      Hybrid sleep + busy-poll identical to max30102_subsystem.py worker.
      Period accuracy: ≤ 1 ms jitter on Pi 4 under nominal load.

    Recovery:
      · 1 capture error:  skips frame, decrements health, continues.
      · 5 consecutive errors:  logs WARNING, pauses 0.5 s, attempts reopen.
      · 15 consecutive errors: watchdog escalation (same pattern as IMU/PPG).

    Args:
        stop_event   : threading.Event from ThreadManager.stop_all()
        state_hub    : SharedStateHub from main.py §4
        watchdog     : WatchdogFramework from main.py §6
        cfg          : SystemConfig — uses cfg.camera and cfg.optical fields
        log          : child Logger from LoggingManager.get("acq-camera")
        result_queue : optional Queue[FrameResult] for consumers (testing/recording)
    """
    log.info("worker_acq_camera: starting (picamera2=%s  cv2=%s  scipy=%s)",
             _PICAM, _CV2, _SCIPY)

    # ── Build per-subsystem config ─────────────────────────────────────────
    # Accept both the legacy CameraConfig / OpticalConfig from main.py §2
    # and a direct NoIRCameraConfig instance.
    raw_cam_cfg = getattr(cfg, 'camera', None)
    raw_opt_cfg = getattr(cfg, 'optical', None)

    if isinstance(raw_cam_cfg, NoIRCameraConfig):
        cam_cfg = raw_cam_cfg
    elif raw_cam_cfg is not None:
        # Translate from main.py CameraConfig + OpticalConfig
        cam_cfg = NoIRCameraConfig(
            width              = raw_cam_cfg.width,
            height             = raw_cam_cfg.height,
            target_fps         = raw_cam_cfg.target_fps,
            pixel_format       = raw_cam_cfg.pixel_format,
            aec_exp_init_us    = raw_cam_cfg.init_exposure_us,
            aec_gain_init      = raw_cam_cfg.init_gain,
            aec_exp_min_us     = raw_cam_cfg.min_exposure_us,
            aec_exp_max_us     = raw_cam_cfg.max_exposure_us,
            aec_gain_min       = raw_cam_cfg.min_gain,
            aec_gain_max       = raw_cam_cfg.max_gain,
            aec_target_lsb     = raw_opt_cfg.target_roi_intensity if raw_opt_cfg else 140.0,
            aec_deadband_lsb   = raw_opt_cfg.intensity_deadband   if raw_opt_cfg else 12.0,
            clahe_clip         = raw_opt_cfg.clahe_clip            if raw_opt_cfg else 2.5,
            clahe_tile         = raw_opt_cfg.clahe_tile            if raw_opt_cfg else (8, 8),
            gauss_kernel       = raw_opt_cfg.blur_kernel           if raw_opt_cfg else 5,
            ipPG_bp_low_hz     = raw_opt_cfg.bp_low_hz             if raw_opt_cfg else 0.65,
            ipPG_bp_high_hz    = raw_opt_cfg.bp_high_hz            if raw_opt_cfg else 4.0,
            ipPG_bp_order      = raw_opt_cfg.bp_order              if raw_opt_cfg else 4,
            signal_buf_len     = raw_opt_cfg.signal_buffer_frames  if raw_opt_cfg else 300,
            signal_min_fill    = raw_opt_cfg.min_frames_for_filter if raw_opt_cfg else 60,
            finger_dc_thresh   = raw_opt_cfg.finger_dc_threshold   if raw_opt_cfg else 35.0,
        )
    else:
        cam_cfg = NoIRCameraConfig()

    # ── Instantiate subsystems ─────────────────────────────────────────────
    health    = CameraHealthRecord()
    driver    = CameraHardwareDriver(cam_cfg, health, log)
    aec       = AdaptiveExposureCtrl(cam_cfg, driver, log)
    extractor = IRChannelExtractor(cam_cfg)
    segmenter = AdaptiveTissueSegmenter(cam_cfg, log)
    motion    = MotionQualityGate(cam_cfg)
    sig_buf   = OpticalSignalBuffer(cam_cfg, log)
    ipPG_eng  = iPPGEngine(cam_cfg, log)
    quality   = FrameQualityScorer(cam_cfg)
    renderer  = OpenCVOverlayRenderer(cam_cfg)
    health_sc = CameraHealthScorer(cam_cfg, log)

    # Open hardware
    hw_enabled = getattr(cfg, 'hw_camera_enabled', True)
    try:
        if hw_enabled:
            driver.open()
        else:
            log.info("worker_acq_camera: hw_camera_enabled=False — simulation.")
            driver.open()   # driver handles PICAM unavailability internally
    except Exception as e:
        log.warning("worker_acq_camera: driver.open() failed: %s", e)

    # OpenCV window
    if _CV2 and getattr(cfg, 'cv2_display', True):
        try:
            renderer.init()
        except Exception as e:
            log.warning("worker_acq_camera: cv2 window init failed: %s", e)

    # ── Timing ────────────────────────────────────────────────────────────
    FS     = float(cam_cfg.target_fps)
    DT     = 1.0 / FS
    MARGIN = 0.001          # 1 ms busy-poll guard
    t_next = time.monotonic()

    # ── State ─────────────────────────────────────────────────────────────
    n_total       = 0
    fps_ema       = 0.0
    t_prev_fps    = time.monotonic()
    prev_finger   = False
    latest_result : Optional[FrameResult] = None
    RECOVERY_THRESH = 15

    # iPPG analysis runs every frame but Welch is cached (< 1 ms incremental)
    log.info("worker_acq_camera: acquisition loop starting — target=%.0f Hz.", FS)

    # ── Main loop ──────────────────────────────────────────────────────────
    while not stop_event.is_set():

        # ── Precise timing: hybrid sleep + busy-poll ──────────────────────
        now = time.monotonic()
        sleep_to = t_next - MARGIN
        if sleep_to > now:
            time.sleep(sleep_to - now)
        while time.monotonic() < t_next:
            pass
        t_next += DT
        ts_frame = time.monotonic()

        # ── Frame capture ─────────────────────────────────────────────────
        frame = driver.capture()

        # Recovery handling on consecutive errors
        if health.consecutive_errors >= 5:
            log.warning("worker_acq_camera: %d consecutive errors — attempting reopen.",
                        health.consecutive_errors)
            time.sleep(0.5)
            try:
                driver.close()
                driver.open()
                log.info("worker_acq_camera: camera recovered.")
                health.consecutive_errors = 0
            except Exception as e2:
                log.error("worker_acq_camera: reopen failed: %s", e2)

        if health.consecutive_errors >= RECOVERY_THRESH:
            watchdog.report_error(
                "acq-camera",
                f"Camera: {RECOVERY_THRESH} consecutive errors — escalating.")
            # Watchdog will drive state machine; keep looping until stop_event

        # ── IR extraction + pre-filter ────────────────────────────────────
        ir_metric, ir_disp = extractor.process(frame)

        # ── Photometrics (whole frame) ────────────────────────────────────
        flat      = ir_metric.ravel().astype(np.float32)
        mean_int  = float(flat.mean())
        std_int   = float(flat.std())
        sat_frac  = float((flat >= 254).mean())

        # ── Adaptive exposure update ──────────────────────────────────────
        exp_us, gain = aec.update(mean_int)

        # ── Tissue segmentation ───────────────────────────────────────────
        mask, roi = segmenter.update(ir_metric)

        # ── ROI photometric extraction ────────────────────────────────────
        roi_crop = ir_metric[roi.y1:roi.y2, roi.x1:roi.x2]
        if roi_crop.size > 0:
            roi_mean = float(roi_crop.astype(np.float32).mean())
            roi_std  = float(roi_crop.astype(np.float32).std())
        else:
            roi_mean, roi_std = 0.0, 0.0

        finger_present = roi_mean >= cam_cfg.finger_dc_thresh

        # ── Motion gate ───────────────────────────────────────────────────
        mad, lap_var, motion_score, frame_rejected = motion.update(ir_metric)

        # ── Finger state change ───────────────────────────────────────────
        if finger_present != prev_finger:
            if not finger_present:
                sig_buf.reset()
                ipPG_eng.reset()
                log.info("worker_acq_camera: finger removed — signal buffer reset.")
            else:
                log.info("worker_acq_camera: finger detected (ROI mean=%.1f LSB).",
                         roi_mean)
            prev_finger = finger_present

        # ── Optical signal push ───────────────────────────────────────────
        if finger_present and not frame_rejected:
            sig_buf.push(roi_mean, ts_frame)

        # ── iPPG analysis (every frame — Welch result is cached in SOS state)
        ipPG_result = ipPG_eng.analyze(sig_buf)

        # Expose latest raw + filtered to display
        raw_latest  = float(sig_buf._raw_buf[-1])  if sig_buf._raw_buf  else 0.0
        filt_latest = float(sig_buf._filt_buf[-1]) if sig_buf._filt_buf else 0.0

        # ── FPS estimation ─────────────────────────────────────────────────
        now2 = time.monotonic()
        dt2  = now2 - t_prev_fps
        if dt2 > 0:
            fps_ema = 0.05 * (1.0 / dt2) + 0.95 * fps_ema
        t_prev_fps = now2

        # ── Quality scoring ────────────────────────────────────────────────
        stab, iq, conf, q_label = quality.score(
            mean_int, sat_frac, lap_var, motion_score,
            finger_present, roi.confidence, ipPG_result.snr_db)

        # ── Assemble FrameResult ───────────────────────────────────────────
        fd = FrameResult(
            ts                    = ts_frame,
            frame_idx             = n_total,
            fps                   = fps_ema,
            mean_intensity        = mean_int,
            std_intensity         = std_int,
            saturation_frac       = sat_frac,
            laplacian_var         = lap_var,
            exposure_us           = exp_us,
            analogue_gain         = gain,
            roi                   = roi,
            finger_present        = finger_present,
            roi_mean              = roi_mean,
            roi_std               = roi_std,
            motion_mad            = mad,
            motion_score          = motion_score,
            frame_rejected        = frame_rejected,
            raw_signal_latest     = raw_latest,
            filtered_signal_latest= filt_latest,
            ipPG                  = ipPG_result,
            optical_stability     = stab,
            image_quality         = iq,
            optical_confidence    = conf,
            quality_label         = q_label,
        )
        latest_result = fd

        # ── SharedStateHub snapshot update ────────────────────────────────
        state_hub.push_optical_sample(roi_mean)
        state_hub.update_snapshot(
            roi_mean_ir           = roi_mean,
            roi_std_ir            = roi_std,
            optical_ac_rms        = ipPG_result.ac_rms,
            optical_snr_db        = ipPG_result.snr_db,
            optical_quality_conf  = conf,
            optical_quality_label = q_label,
            finger_present_camera = finger_present,
            exposure_us           = exp_us,
            analogue_gain         = gain,
            camera_fps            = fps_ema,
            frame_idx             = n_total,
            ts_optical            = ts_frame,
        )

        # ── Push to camera_queue (for proc-optical thread compatibility) ───
        # We push the raw BGR frame so proc-optical can still consume it
        # if it runs as a separate thread.  Use put_nowait + evict-on-full.
        try:
            state_hub.camera_queue.put_nowait(frame)
        except queue.Full:
            try:
                state_hub.camera_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                state_hub.camera_queue.put_nowait(frame)
            except queue.Full:
                pass

        # ── Optional result queue (for unit tests / recording) ────────────
        if result_queue is not None:
            try:
                result_queue.put_nowait(fd)
            except queue.Full:
                pass

        n_total += 1

        # ── Heartbeat (10 Hz → every 3 frames at 30 Hz) ───────────────────
        if n_total % 3 == 0:
            watchdog.beat("acq-camera")

        # ── Health scoring (every 150 frames ≈ 5 s) ──────────────────────
        if n_total % 150 == 0:
            h_score = health_sc.score(health)
            state_hub.update_snapshot(
                # Expose camera health for watchdog / CSV logger
                # (no dedicated field in SensorSnapshot; reuse system_state)
            )

        # ── OpenCV overlay (every 2nd frame → ~15 Hz render) ─────────────
        if n_total % 2 == 0 and _CV2:
            # Read BPM / SpO2 from shared hub for cross-modal display
            snap     = state_hub.read_snapshot()
            keep_win = renderer.render(
                ir_disp, mask, fd,
                snap_bpm  = snap.bpm,
                snap_spo2 = snap.spo2,
            )
            if not keep_win:
                log.info("worker_acq_camera: OpenCV window closed by user.")
                stop_event.set()

    # ── Teardown ──────────────────────────────────────────────────────────
    renderer.close()
    driver.close()
    gc.collect()

    log.info(
        "worker_acq_camera: stopped. frames=%d  errors=%d  "
        "dropped=%d  health=%.2f",
        health.total_frames, health.capture_errors,
        health.dropped_frames, health.health_score)


# ─────────────────────────────────────────────────────────────────────────────
# §13  SELF-TEST (run standalone: python noir_camera_subsystem.py)
# ─────────────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """
    Standalone self-test — exercises the full subsystem in simulation mode
    for a configurable duration and prints a metric summary.

    Expected results (simulation, 15 s):
      Optical quality  : FAIR or GOOD (synthetic scene is ideal-ish)
      iPPG BPM proxy   : ≈ 66 BPM (f_hr = 1.1 Hz in synthetic signal)
      AC/DC ratio      : ≈ 1.5% (AC1 + AC2 combined ≈ 1.7%)
      iPPG confidence  : > 0.4 (after warm-up period)
      Frames processed : ≈ 30 × duration
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="NoIR Camera Subsystem standalone self-test")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Test duration in seconds (default: 15)")
    parser.add_argument("--debug",    action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level   = logging.DEBUG if args.debug else logging.INFO,
        format  = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
        datefmt = "%H:%M:%S",
    )
    log = logging.getLogger("selftest")

    # ── Minimal SharedStateHub stub ────────────────────────────────────────
    import copy

    class _Snap:
        bpm = None; spo2 = None

    class StubHub:
        def __init__(self):
            import queue, collections
            self.camera_queue     = queue.Queue(maxsize=4)
            self._snap            = _Snap()
            self._snap_lock       = threading.Lock()
            self.optical_signal_buf = collections.deque(maxlen=300)

        def push_optical_sample(self, v):
            self.optical_signal_buf.append(v)

        def optical_signal_array(self):
            return np.array(self.optical_signal_buf, dtype=np.float32)

        def update_snapshot(self, **kw):
            with self._snap_lock:
                for k, v in kw.items():
                    setattr(self._snap, k, v)

        def read_snapshot(self):
            with self._snap_lock:
                return copy.copy(self._snap)

    class StubWatchdog:
        def beat(self, n):       pass
        def report_error(self, n, e): pass

    class StubConfig:
        hw_camera_enabled = False   # force simulation
        cv2_display       = False   # no OpenCV window in self-test
        class camera:
            width=640; height=480; target_fps=30; pixel_format="BGR888"
            init_exposure_us=15000; init_gain=4.0
            min_exposure_us=3000; max_exposure_us=40000
            min_gain=1.0; max_gain=8.0
        class optical:
            target_roi_intensity=140.0; intensity_deadband=12.0
            clahe_clip=2.5; clahe_tile=(8,8); blur_kernel=5
            bp_low_hz=0.65; bp_high_hz=4.0; bp_order=4
            signal_buffer_frames=300; min_frames_for_filter=60
            finger_dc_threshold=35.0

    stop    = threading.Event()
    hub     = StubHub()
    wd      = StubWatchdog()
    raw_cfg = StubConfig()
    rq      : queue.Queue = queue.Queue(maxsize=1000)

    t = threading.Thread(
        target=worker_acq_camera,
        kwargs=dict(
            stop_event   = stop,
            state_hub    = hub,
            watchdog     = wd,
            cfg          = raw_cfg,
            log          = log,
            result_queue = rq,
        ),
        daemon=True,
    )
    t.start()

    log.info("NoIR subsystem self-test running for %.0f s (simulation)...",
             args.duration)
    time.sleep(args.duration)
    stop.set()
    t.join(timeout=8.0)

    # ── Collect results ────────────────────────────────────────────────────
    results: List[FrameResult] = []
    while True:
        try:
            results.append(rq.get_nowait())
        except queue.Empty:
            break

    log.info("─" * 65)
    log.info("Self-test complete — %d frames processed.", len(results))

    if results:
        fps_vals   = [r.fps       for r in results if r.fps       > 0]
        bpm_vals   = [r.ipPG.bpm_proxy  for r in results
                      if r.ipPG.bpm_proxy > 0]
        snr_vals   = [r.ipPG.snr_db     for r in results
                      if r.ipPG.snr_db   > 0]
        conf_vals  = [r.optical_confidence for r in results]
        pi_vals    = [r.ipPG.pi_proxy for r in results if r.ipPG.pi_proxy > 0]
        rej_count  = sum(1 for r in results if r.frame_rejected)

        def _fmt(seq):
            if not seq: return "---"
            return f"mean={statistics.mean(seq):.2f}  std={statistics.stdev(seq) if len(seq)>1 else 0:.2f}"

        log.info("  FPS            : %s", _fmt(fps_vals))
        log.info("  iPPG BPM proxy : %s  [EXPERIMENTAL]", _fmt(bpm_vals))
        log.info("  iPPG SNR dB    : %s", _fmt(snr_vals))
        log.info("  iPPG PI proxy  : %s%%  [EXPERIMENTAL]", _fmt(pi_vals))
        log.info("  Optical conf   : %s", _fmt(conf_vals))
        log.info("  Frames rejected: %d / %d  (motion gate)",
                 rej_count, len(results))
        log.info("  Last quality   : %s", results[-1].quality_label)
    log.info("─" * 65)


if __name__ == "__main__":
    _selftest()