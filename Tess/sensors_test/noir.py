#!/usr/bin/env python3
"""
=============================================================================
 NoIR Biomedical Optical Sensing System
 Raspberry Pi NoIR Camera — Infrared Tissue Reflection Analysis Platform
=============================================================================

 Author        : Biomedical Optics Research Prototype
 Target HW     : Raspberry Pi 4/5 + NoIR Camera Module v2/v3
 Illumination  : 850 nm IR LEDs + White LED (optional)
 Enclosure     : Black optical isolation chamber
 Python        : 3.9+
 License       : Research / Experimental Use Only

 SCIENTIFIC DISCLAIMER
 ---------------------
 This system observes *reflected near-infrared light* (850 nm) from the
 surface and shallow subsurface of biological tissue.  It is NOT capable
 of:  X-ray imaging, MRI, deep-tissue scanning, or clinical diagnosis.
 All outputs are labelled "experimental" and must NOT be used for any
 medical or clinical decision.

 OPTICAL PHYSICS BACKGROUND
 --------------------------
 At 850 nm the dominant tissue chromophores are oxy- and deoxy-haemoglobin,
 water, and melanin.  Skin optical penetration depth for CW 850 nm light
 is typically 1–3 mm (depending on tissue type and pigmentation).  The
 NoIR camera detects the backscattered/reflected flux, which encodes
 information about local optical properties (µa, µs') within that shallow
 volume.  This is the physical basis of near-infrared diffuse reflectance
 spectroscopy and reflectance-mode photoplethysmography (rPPG).

 HARDWARE SETUP
 --------------
 1. Attach NoIR Camera to CSI ribbon connector.
 2. Mount 2–4 × 850 nm IR LEDs symmetrically around the lens (≈15 mm
    stand-off from tissue surface).  Drive at ≈50 mA (verify datasheet).
 3. (Optional) Add one white LED for reference/visible-light mode.
 4. Enclose in a black-anodised aluminium or black-painted ABS chamber to
    reject ambient light (critical for SNR).
 5. Aperture for fingertip placement ≈ 15 × 30 mm.

 PIPELINE OVERVIEW
 -----------------
   [Picamera2 CSI stream]
       │
       ▼
   [Frame pre-processing]   ← grayscale, IR extraction, exposure norm.
       │
       ▼
   [Quality gate]           ← motion score, sharpness, saturation check
       │
       ▼
   [ROI detection / lock]   ← fingertip localisation within frame
       │
       ▼
   [Optical analysis]       ← intensity, variance, stability, PPG proxy
       │
       ▼
   [AI interpretation]      ← rule-based optical quality feedback
       │
       ▼
   [Terminal dashboard  +  OpenCV overlay visualisation]

=============================================================================
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import sys
import os
import time
import math
import threading
import queue
import signal
import logging
import collections
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Deque

# ---------------------------------------------------------------------------
# Third-party (must be installed on the Pi):
#   pip install numpy opencv-python-headless picamera2
# ---------------------------------------------------------------------------
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARN] opencv-python not found.  Visualisation disabled.")

try:
    from picamera2 import Picamera2 # sudo apt install python3-picamera2 
    from libcamera import controls as libcam_controls # pip install opencv-python-headless
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("[WARN] picamera2 not found.  Running in DEMO/SIMULATION mode.")

# ---------------------------------------------------------------------------
# Logging — structured, engineering-style
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("NoIR-BioSensor")


# ===========================================================================
# § 1  CONFIGURATION
# ===========================================================================

@dataclass
class CameraConfig:
    """Picamera2 acquisition parameters optimised for 850 nm IR reflectance."""

    # Frame dimensions — VGA is a good trade-off between resolution and CPU
    width: int = 640
    height: int = 480

    # Target frame rate (fps).  30 is achievable at VGA on Pi 4.
    target_fps: int = 30

    # Exposure time bounds (µs).  IR isolation chamber → longer exposures OK.
    min_exposure_us: int = 5_000      # 5 ms  — motion blur floor
    max_exposure_us: int = 33_000     # 33 ms — ≈ 1 frame at 30 fps

    # Analogue gain — keep low to minimise shot noise on slow signals
    analogue_gain: float = 4.0

    # Auto-white-balance off: NoIR in grayscale, AWB is irrelevant
    awb_mode: str = "off"

    # AEC/AGC — we use manual exposure for signal stability
    auto_exposure: bool = False

    # Camera format — BGR888 from Picamera2, we convert to gray immediately
    pixel_format: str = "BGR888"


@dataclass
class OpticalConfig:
    """Physical/optical parameters of the 850 nm sensing system."""

    # Near-IR wavelength (nm) — informational; used in log headers
    ir_wavelength_nm: int = 850

    # Expected tissue reflectance range [0–255 grayscale].
    # Lightly pigmented skin at 850 nm typically reflects 40–80 % of incident.
    expected_intensity_min: int = 40
    expected_intensity_max: int = 220

    # Gaussian blur kernel (px) for speckle / shot-noise suppression
    blur_kernel_size: int = 5

    # CLAHE clip limit for adaptive histogram equalisation
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # Adaptive threshold block size & constant
    adapt_thresh_block: int = 21
    adapt_thresh_c: int = 4


@dataclass
class ROIConfig:
    """Region-of-interest parameters for fingertip optical sensing."""

    # Fraction of frame to use as the central ROI (width, height)
    roi_fraction_w: float = 0.45
    roi_fraction_h: float = 0.55

    # Minimum mean intensity to accept an ROI frame as "finger present"
    presence_intensity_threshold: int = 35

    # Temporal smoothing window (frames) for optical signal
    signal_window: int = 90           # 3 s at 30 fps


@dataclass
class MotionConfig:
    """Frame-difference–based motion artefact detection thresholds."""

    # Mean absolute difference [0–255] above which frame is flagged as "motion"
    motion_flag_threshold: float = 8.0

    # Laplacian variance below which the frame is "blurry" (motion smear)
    blur_threshold: float = 25.0

    # Exponential moving-average alpha for motion score
    ema_alpha: float = 0.25


# ===========================================================================
# § 2  DATA STRUCTURES
# ===========================================================================

@dataclass
class FrameMetrics:
    """All per-frame quantitative optical metrics — populated by the pipeline."""

    timestamp: float = 0.0
    frame_index: int = 0

    # Whole-frame metrics
    mean_intensity: float = 0.0
    std_intensity: float = 0.0
    min_intensity: float = 0.0
    max_intensity: float = 0.0
    saturation_fraction: float = 0.0   # fraction of pixels at 255

    # Sharpness (Laplacian variance — higher = sharper)
    laplacian_variance: float = 0.0

    # Motion (frame-difference MAD)
    motion_mad: float = 0.0
    motion_score: float = 0.0          # 0 (still) … 1 (heavy motion)

    # ROI metrics
    roi_mean: float = 0.0
    roi_std: float = 0.0
    roi_variance: float = 0.0
    finger_present: bool = False

    # Derived optical scores [0–1]
    optical_stability: float = 0.0
    image_quality: float = 0.0
    optical_confidence: float = 0.0

    # Instantaneous FPS measured by the acquisition loop
    fps: float = 0.0


@dataclass
class OpticalSignalBuffer:
    """
    Circular buffer of per-frame ROI mean intensity values.

    This sequence is the 'optical signal' — in reflectance-mode PPG research
    it would be bandpass-filtered to extract cardiac pulsatility.  Here we
    expose it as a raw experimental signal without any clinical inference.
    """
    maxlen: int = 300   # 10 s at 30 fps

    _buf: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=300))
    _timestamps: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=300))

    def push(self, value: float, t: float) -> None:
        self._buf.append(value)
        self._timestamps.append(t)

    def as_array(self) -> np.ndarray:
        return np.array(self._buf, dtype=np.float32)

    def signal_ac_rms(self) -> float:
        """AC RMS of the detrended optical signal (proxy for pulsatility)."""
        a = self.as_array()
        if len(a) < 10:
            return 0.0
        detrended = a - np.mean(a)
        return float(np.sqrt(np.mean(detrended ** 2)))

    def signal_snr_db(self) -> float:
        """Estimated signal-to-noise ratio in dB (DC/AC ratio in log scale)."""
        a = self.as_array()
        if len(a) < 10:
            return 0.0
        dc = np.mean(a)
        ac_rms = self.signal_ac_rms()
        if ac_rms < 1e-6:
            return 60.0   # effectively infinite SNR → very stable
        return 20.0 * math.log10(dc / ac_rms + 1e-9)

    def __len__(self) -> int:
        return len(self._buf)


# ===========================================================================
# § 3  CAMERA ACQUISITION LAYER
# ===========================================================================

class CameraAcquisition:
    """
    Wraps Picamera2 (or a simulation fallback) to deliver BGR frames via a
    thread-safe queue.  The acquisition thread runs independently at the
    requested frame rate so that downstream processing never blocks the
    camera pipeline.
    """

    def __init__(self, cfg: CameraConfig):
        self._cfg = cfg
        self._camera: Optional["Picamera2"] = None
        self._frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=4)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._last_ts = time.monotonic()
        self._fps_ema = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise camera hardware and start the acquisition thread."""
        if PICAMERA2_AVAILABLE:
            self._init_picamera2()
        else:
            log.warning("Picamera2 unavailable — using synthetic IR test signal.")

        self._running = True
        self._thread = threading.Thread(
            target=self._acquisition_loop, daemon=True, name="cam-acq"
        )
        self._thread.start()
        log.info("Camera acquisition thread started.")

    def stop(self) -> None:
        """Gracefully stop acquisition and release hardware."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._camera:
            self._camera.stop()
            self._camera.close()
            log.info("Camera hardware released.")

    def get_frame(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """
        Retrieve the latest frame from the queue (BGR uint8 ndarray).
        Returns None on timeout.
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def fps(self) -> float:
        return self._fps_ema

    # ------------------------------------------------------------------
    # Internal — hardware init
    # ------------------------------------------------------------------

    def _init_picamera2(self) -> None:
        """Configure Picamera2 for fixed-exposure IR reflectance imaging."""
        self._camera = Picamera2()

        # Build a still-capture–style config at video speed
        cfg = self._camera.create_video_configuration(
            main={
                "size": (self._cfg.width, self._cfg.height),
                "format": self._cfg.pixel_format,
            },
            controls={
                # Manual exposure — essential for optical signal stability
                "AeEnable": False,
                "ExposureTime": self._cfg.min_exposure_us,
                "AnalogueGain": self._cfg.analogue_gain,
                # Disable AWB — irrelevant for grayscale IR
                "AwbEnable": False,
                # Colour gains 1,1 → neutral (camera still outputs BGR)
                "ColourGains": (1.0, 1.0),
                # No noise reduction — we do our own
                "NoiseReductionMode": libcam_controls.draft.NoiseReductionModeEnum.Off,
            },
        )
        self._camera.configure(cfg)
        self._camera.start()
        # Allow auto-exposure to settle before locking (won't apply — AeEnable=False)
        time.sleep(0.5)
        log.info(
            "Picamera2 configured: %dx%d @ %dfps, exposure=%dµs, gain=%.1f",
            self._cfg.width, self._cfg.height, self._cfg.target_fps,
            self._cfg.min_exposure_us, self._cfg.analogue_gain,
        )

    # ------------------------------------------------------------------
    # Internal — acquisition loop
    # ------------------------------------------------------------------

    def _acquisition_loop(self) -> None:
        """
        Hot loop: capture frames as fast as possible (up to target_fps) and
        push them onto the queue.  Drop oldest frame if downstream is slow.
        """
        frame_period = 1.0 / self._cfg.target_fps

        while self._running:
            t0 = time.monotonic()

            frame = self._capture_frame()

            if frame is not None:
                # Non-blocking put: discard oldest frame if queue full
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put_nowait(frame)

                # EMA FPS estimator
                now = time.monotonic()
                dt = now - self._last_ts
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self._fps_ema = 0.1 * inst_fps + 0.9 * self._fps_ema
                self._last_ts = now
                self._frame_count += 1

            # Pace to target FPS
            elapsed = time.monotonic() - t0
            sleep_t = frame_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture one frame from hardware or synthesise a test frame."""
        if self._camera is not None:
            try:
                arr = self._camera.capture_array("main")
                return arr  # BGR uint8
            except Exception as exc:
                log.error("Frame capture error: %s", exc)
                return None
        else:
            return self._synthetic_ir_frame()

    # ------------------------------------------------------------------
    # Simulation / demo mode
    # ------------------------------------------------------------------

    def _synthetic_ir_frame(self) -> np.ndarray:
        """
        Generate a physically plausible synthetic IR reflectance frame for
        development / testing without hardware.

        The frame simulates:
          - Gaussian-illuminated fingertip region (central ROI)
          - Slight pulsatile intensity modulation (≈1 % AC)
          - Gaussian shot noise (σ ≈ 3 LSB)
          - Slow illumination drift to test stability scoring
        """
        h, w = self._cfg.height, self._cfg.width
        t = time.monotonic()

        # Background: very low IR reflection (optical isolation chamber)
        frame = np.full((h, w), 12, dtype=np.float32)

        # Fingertip Gaussian blob — centred, elliptical
        cx, cy = w // 2, h // 2
        sigma_x, sigma_y = w * 0.18, h * 0.28

        yy, xx = np.ogrid[:h, :w]
        gauss = np.exp(
            -((xx - cx) ** 2 / (2 * sigma_x ** 2) +
              (yy - cy) ** 2 / (2 * sigma_y ** 2))
        )

        # DC reflectance ≈ 130 LSB (≈ 51 % of 255)
        dc_level = 130.0

        # AC pulsatility: 1 % of DC at simulated ~1.1 Hz (66 bpm)
        ac_component = dc_level * 0.01 * math.sin(2 * math.pi * 1.1 * t)

        # Slow illumination drift (< 0.5 % over 60 s)
        drift = dc_level * 0.005 * math.sin(2 * math.pi * t / 60.0)

        frame += gauss * (dc_level + ac_component + drift)

        # Shot noise (Poisson-like approximation via Gaussian)
        noise = np.random.normal(0, 3.0, (h, w)).astype(np.float32)
        frame += noise

        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Return as 3-channel BGR (pipeline expects colour input)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if CV2_AVAILABLE \
            else np.stack([frame, frame, frame], axis=-1)


# ===========================================================================
# § 4  IMAGE PROCESSING PIPELINE
# ===========================================================================

class IRImageProcessor:
    """
    Research-grade image processing pipeline for 850 nm reflectance frames.

    Processing stages (in order):
      1. BGR → grayscale extraction (luminance-weighted)
      2. Gaussian blur for shot-noise suppression
      3. CLAHE adaptive histogram equalisation for visibility
      4. Optional adaptive thresholding for structural analysis
      5. Per-frame metric extraction
    """

    def __init__(self, cam_cfg: CameraConfig, opt_cfg: OpticalConfig,
                 roi_cfg: ROIConfig, mot_cfg: MotionConfig):
        self._cam = cam_cfg
        self._opt = opt_cfg
        self._roi = roi_cfg
        self._mot = mot_cfg

        # CLAHE engine — contrast-limited adaptive histogram equalisation
        # improves low-contrast IR tissue features without over-amplifying noise
        if CV2_AVAILABLE:
            self._clahe = cv2.createCLAHE(
                clipLimit=opt_cfg.clahe_clip_limit,
                tileGridSize=opt_cfg.clahe_tile_grid,
            )
        else:
            self._clahe = None

        # Previous frame for motion detection
        self._prev_gray: Optional[np.ndarray] = None

        # Exponential moving-average motion score
        self._motion_ema: float = 0.0

        # ROI pixel coordinates (computed once from frame size)
        self._roi_slice: Optional[Tuple[slice, slice]] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, FrameMetrics]:
        """
        Full pipeline for one frame.

        Returns:
            gray_raw    — grayscale, blur-suppressed (for metric extraction)
            gray_disp   — CLAHE-enhanced grayscale (for visualisation)
            metrics     — populated FrameMetrics dataclass
        """
        metrics = FrameMetrics(timestamp=time.monotonic())

        # --- Stage 1: extract IR-channel grayscale ---
        # For a NoIR camera, the red channel carries the most 850 nm signal
        # (the Bayer filter passes ~50 % of IR to R pixels vs ~10–20 % for B).
        # Using cv2.cvtColor gives luminance weighting (0.114 B + 0.587 G + 0.299 R),
        # which is sub-optimal for pure IR.  We instead weight heavily toward R.
        gray_raw = self._extract_ir_channel(bgr_frame)

        # --- Stage 2: Gaussian blur (σ ≈ 1 px) ---
        # Suppresses pixel-level shot noise before metric extraction.
        k = self._opt.blur_kernel_size
        if CV2_AVAILABLE:
            gray_blurred = cv2.GaussianBlur(gray_raw, (k, k), 0)
        else:
            gray_blurred = gray_raw  # fallback: no-op

        # --- Stage 3: CLAHE for display ---
        if self._clahe is not None:
            gray_disp = self._clahe.apply(gray_blurred)
        else:
            gray_disp = gray_blurred.copy()

        # --- Stage 4: whole-frame photometric metrics ---
        self._compute_frame_metrics(gray_blurred, metrics)

        # --- Stage 5: motion / sharpness detection ---
        self._compute_motion_metrics(gray_blurred, metrics)

        # --- Stage 6: ROI optical analysis ---
        self._compute_roi_metrics(gray_blurred, bgr_frame, metrics)

        # --- Stage 7: composite optical scores ---
        self._compute_optical_scores(metrics)

        return gray_raw, gray_disp, metrics

    def compute_roi_coords(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """
        Return (x1, y1, x2, y2) pixel coordinates of the central ROI.
        Cached after first call.
        """
        rw = int(w * self._roi.roi_fraction_w)
        rh = int(h * self._roi.roi_fraction_h)
        x1 = (w - rw) // 2
        y1 = (h - rh) // 2
        return x1, y1, x1 + rw, y1 + rh

    # ------------------------------------------------------------------
    # Internal — channel extraction
    # ------------------------------------------------------------------

    def _extract_ir_channel(self, bgr: np.ndarray) -> np.ndarray:
        """
        Extract the IR-dominant channel from a BGR NoIR frame.

        Physical rationale:
          The OmniVision OV5647 / Sony IMX219 Bayer filter transmits visible
          and NIR light to all colour channels, but with different spectral
          sensitivities.  At 850 nm the red-channel quantum efficiency is
          approximately 2–4× higher than blue.  Weighting toward R gives
          better SNR for the IR signal.

        Weights: B=0.05, G=0.20, R=0.75  (empirically tuned for 850 nm)
        """
        if not CV2_AVAILABLE:
            return np.mean(bgr, axis=2).astype(np.uint8)

        b = bgr[:, :, 0].astype(np.float32)
        g = bgr[:, :, 1].astype(np.float32)
        r = bgr[:, :, 2].astype(np.float32)

        ir_weighted = 0.05 * b + 0.20 * g + 0.75 * r
        return np.clip(ir_weighted, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Internal — metrics
    # ------------------------------------------------------------------

    def _compute_frame_metrics(self, gray: np.ndarray, m: FrameMetrics) -> None:
        """Compute whole-frame photometric statistics."""
        flat = gray.ravel().astype(np.float32)
        m.mean_intensity = float(np.mean(flat))
        m.std_intensity = float(np.std(flat))
        m.min_intensity = float(np.min(flat))
        m.max_intensity = float(np.max(flat))
        m.saturation_fraction = float(np.mean(flat >= 254))

    def _compute_motion_metrics(self, gray: np.ndarray, m: FrameMetrics) -> None:
        """
        Frame-difference motion detection + Laplacian sharpness.

        Motion MAD (mean absolute difference between consecutive frames) is a
        simple but effective motion artefact detector for slow-moving objects
        such as a finger resting on the sensor aperture.  Blurry frames
        (low Laplacian variance) indicate either motion smear or defocus.
        """
        alpha = self._mot.ema_alpha

        # Laplacian variance — sharpness estimator
        if CV2_AVAILABLE:
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            m.laplacian_variance = float(lap.var())
        else:
            m.laplacian_variance = 50.0  # assume sharp in fallback

        # Frame difference MAD
        if self._prev_gray is not None and gray.shape == self._prev_gray.shape:
            diff = np.abs(gray.astype(np.int16) - self._prev_gray.astype(np.int16))
            m.motion_mad = float(np.mean(diff))
        else:
            m.motion_mad = 0.0

        # EMA-smoothed motion score [0–1]
        raw_motion = min(m.motion_mad / self._mot.motion_flag_threshold, 1.0)
        self._motion_ema = alpha * raw_motion + (1 - alpha) * self._motion_ema
        m.motion_score = self._motion_ema

        self._prev_gray = gray.copy()

    def _compute_roi_metrics(self, gray: np.ndarray, bgr: np.ndarray,
                             m: FrameMetrics) -> None:
        """
        Extract optical metrics from the central fingertip ROI.

        The ROI is a fixed central rectangle — in a calibrated system you
        would adaptively fit an ellipse to the finger boundary.
        """
        h, w = gray.shape
        x1, y1, x2, y2 = self.compute_roi_coords(h, w)
        roi = gray[y1:y2, x1:x2].astype(np.float32)

        if roi.size == 0:
            return

        m.roi_mean = float(np.mean(roi))
        m.roi_std = float(np.std(roi))
        m.roi_variance = float(np.var(roi))
        m.finger_present = m.roi_mean >= self._roi.presence_intensity_threshold

    def _compute_optical_scores(self, m: FrameMetrics) -> None:
        """
        Compute three composite optical quality scores, each in [0, 1].

        optical_stability:
          Penalises high motion score and low sharpness.

        image_quality:
          Rewards proper exposure (not too dark, not saturated) and sharpness.

        optical_confidence:
          Combined score gating whether downstream analysis is trustworthy.
        """
        # --- Motion / stability ---
        stability = 1.0 - m.motion_score
        if m.laplacian_variance < self._mot.blur_threshold:
            blur_penalty = m.laplacian_variance / self._mot.blur_threshold
        else:
            blur_penalty = 1.0
        m.optical_stability = float(np.clip(stability * blur_penalty, 0.0, 1.0))

        # --- Image quality ---
        # Penalise darkness (mean < 40) and saturation (> 5 % pixels at 255)
        intensity_ok = np.clip(
            (m.mean_intensity - 10.0) / (self._opt.expected_intensity_min - 10.0),
            0.0, 1.0,
        )
        saturation_penalty = max(0.0, 1.0 - m.saturation_fraction * 20.0)
        sharpness_norm = np.clip(m.laplacian_variance / 200.0, 0.0, 1.0)
        m.image_quality = float(intensity_ok * saturation_penalty * (0.5 + 0.5 * sharpness_norm))

        # --- Overall optical confidence ---
        finger_weight = 1.0 if m.finger_present else 0.3
        m.optical_confidence = float(
            m.optical_stability * 0.4
            + m.image_quality * 0.4
            + finger_weight * 0.2
        )


# ===========================================================================
# § 5  REGION-OF-INTEREST OPTICAL SIGNAL TRACKER
# ===========================================================================

class OpticalSignalTracker:
    """
    Accumulates per-frame ROI mean intensities into a circular buffer and
    computes derived temporal optical metrics.

    The buffer is the raw experimental optical signal.  It intentionally
    contains no bandpass filtering or cardiac inference — this is left as a
    research task for downstream analysis.
    """

    def __init__(self, roi_cfg: ROIConfig):
        self._cfg = roi_cfg
        self._signal = OpticalSignalBuffer(maxlen=roi_cfg.signal_window)
        self._good_frames = 0
        self._total_frames = 0

    def update(self, metrics: FrameMetrics) -> None:
        self._total_frames += 1
        if metrics.finger_present and metrics.motion_score < 0.3:
            self._signal.push(metrics.roi_mean, metrics.timestamp)
            self._good_frames += 1

    @property
    def signal_ac_rms(self) -> float:
        return self._signal.signal_ac_rms()

    @property
    def signal_snr_db(self) -> float:
        return self._signal.signal_snr_db()

    @property
    def buffer_fill(self) -> int:
        return len(self._signal)

    @property
    def good_frame_ratio(self) -> float:
        if self._total_frames == 0:
            return 0.0
        return self._good_frames / self._total_frames


# ===========================================================================
# § 6  AI-ASSISTED OPTICAL INTERPRETATION
# ===========================================================================

class OpticalInterpreter:
    """
    Lightweight rule-based optical quality interpreter.

    IMPORTANT: This module produces EXPERIMENTAL feedback on optical signal
    quality only.  It makes no clinical, diagnostic, or physiological claims.
    All outputs are labelled "non-clinical experimental analysis".
    """

    @staticmethod
    def interpret(m: FrameMetrics, tracker: OpticalSignalTracker) -> List[str]:
        """
        Return a list of human-readable interpretation strings based on current
        optical metrics.  Each string begins with a severity tag:
          [OK]   — nominal
          [INFO] — advisory
          [WARN] — quality concern
          [ERR]  — critical quality failure
        """
        messages: List[str] = []

        # Finger presence
        if not m.finger_present:
            messages.append("[WARN] No tissue detected in ROI — place fingertip on aperture.")
        else:
            messages.append("[OK]   Tissue presence confirmed in optical ROI.")

        # Motion artefacts
        if m.motion_score > 0.6:
            messages.append("[ERR]  Severe motion artefact — hold fingertip still.")
        elif m.motion_score > 0.3:
            messages.append("[WARN] Moderate motion detected — stabilise placement.")
        else:
            messages.append("[OK]   Low motion — optical coupling is stable.")

        # Illumination / exposure
        if m.mean_intensity < 20:
            messages.append("[ERR]  Frame underexposed — check IR LED drive current.")
        elif m.mean_intensity < 40:
            messages.append("[WARN] Low IR illumination — increase LED current or exposure.")
        elif m.saturation_fraction > 0.1:
            messages.append("[WARN] Partial saturation detected — reduce IR LED drive or gain.")
        else:
            messages.append("[OK]   Illumination level within experimental range.")

        # Sharpness
        if m.laplacian_variance < 10:
            messages.append("[ERR]  Severe blur — check focus, possible lens contamination.")
        elif m.laplacian_variance < 25:
            messages.append("[WARN] Low sharpness — verify optical alignment.")

        # Signal quality (requires ≥ 2 s of data)
        if tracker.buffer_fill >= 30:
            snr = tracker.signal_snr_db
            ac = tracker.signal_ac_rms
            if snr < 20:
                messages.append(
                    f"[WARN] Signal SNR = {snr:.1f} dB — noisy optical signal."
                )
            else:
                messages.append(
                    f"[OK]   Signal SNR = {snr:.1f} dB, AC-RMS = {ac:.2f} LSB "
                    f"(experimental optical signal)."
                )

        # Overall confidence
        conf = m.optical_confidence
        if conf > 0.75:
            messages.append(f"[OK]   Optical confidence: {conf*100:.0f}% — analysis reliable.")
        elif conf > 0.45:
            messages.append(f"[INFO] Optical confidence: {conf*100:.0f}% — marginal quality.")
        else:
            messages.append(f"[WARN] Optical confidence: {conf*100:.0f}% — poor coupling.")

        return messages


# ===========================================================================
# § 7  TERMINAL DASHBOARD
# ===========================================================================

class TerminalDashboard:
    """
    Engineering-style real-time terminal monitor.

    Uses ANSI escape sequences for in-place updating (no external dependency
    on curses).  Refreshes at a fixed rate independent of the processing loop.
    """

    # ANSI codes
    _CLR = "\033[2J\033[H"   # clear + home
    _BOLD = "\033[1m"
    _RST = "\033[0m"
    _GRN = "\033[32m"
    _YLW = "\033[33m"
    _RED = "\033[31m"
    _CYN = "\033[36m"
    _WHT = "\033[37m"

    def __init__(self, opt_cfg: OpticalConfig, cam_cfg: CameraConfig):
        self._opt = opt_cfg
        self._cam = cam_cfg
        self._start_time = time.monotonic()

    def render(self, m: FrameMetrics, tracker: OpticalSignalTracker,
               interpreter: OpticalInterpreter, cam_ok: bool) -> None:
        """Clear screen and redraw the full dashboard."""
        elapsed = time.monotonic() - self._start_time
        mins, secs = divmod(int(elapsed), 60)

        lines: List[str] = []
        B, R, G, Y, C, W = self._BOLD, self._RST, self._GRN, self._YLW, self._CYN, self._WHT

        # ── Header ────────────────────────────────────────────────────────────
        lines.append(f"{B}{C}{'═'*70}{R}")
        lines.append(
            f"{B}{C}  NoIR BIOMEDICAL OPTICAL SENSING SYSTEM  "
            f"│  λ={self._opt.ir_wavelength_nm} nm  │  {mins:02d}:{secs:02d}{R}"
        )
        lines.append(f"{B}{C}  Experimental Research Platform — NON-CLINICAL{R}")
        lines.append(f"{B}{C}{'─'*70}{R}")

        # ── Camera status ──────────────────────────────────────────────────────
        cam_str = f"{G}CONNECTED (hw){R}" if cam_ok else f"{Y}SIMULATION{R}"
        lines.append(f"  Camera Status   : {cam_str}")
        lines.append(f"  Frame Rate      : {B}{m.fps:5.1f}{R} fps  "
                     f"(target {self._cam.target_fps} fps)")
        lines.append(f"  Frame Index     : {m.frame_index:,}")

        lines.append(f"{C}{'─'*70}{R}")

        # ── Illumination & exposure ────────────────────────────────────────────
        exp_colour = G if 40 <= m.mean_intensity <= 210 else Y if m.mean_intensity > 0 else R
        lines.append(f"  Mean IR Intensity   : {exp_colour}{m.mean_intensity:6.1f}{R} / 255 LSB")
        lines.append(f"  Std Dev Intensity   : {m.std_intensity:6.1f} LSB")
        sat_c = R if m.saturation_fraction > 0.05 else G
        lines.append(f"  Saturation Fraction : {sat_c}{m.saturation_fraction*100:5.1f}%{R}")
        lines.append(f"  Illumination Status : "
                     + (f"{G}NOMINAL{R}" if 40 <= m.mean_intensity <= 210 else f"{Y}CHECK LEDs{R}"))

        lines.append(f"{C}{'─'*70}{R}")

        # ── ROI analysis ───────────────────────────────────────────────────────
        fp_str = f"{G}PRESENT{R}" if m.finger_present else f"{R}ABSENT {R}"
        lines.append(f"  ROI Status      : {fp_str}")
        lines.append(f"  ROI Mean        : {B}{m.roi_mean:6.1f}{R} LSB")
        lines.append(f"  ROI Std Dev     : {m.roi_std:6.2f} LSB")
        lines.append(f"  ROI Variance    : {m.roi_variance:8.1f} LSB²")

        lines.append(f"{C}{'─'*70}{R}")

        # ── Motion & stability ────────────────────────────────────────────────
        mot_c = G if m.motion_score < 0.2 else Y if m.motion_score < 0.5 else R
        lines.append(f"  Motion Score    : {mot_c}{m.motion_score:5.3f}{R}  (0=still, 1=heavy)")
        lines.append(f"  Motion MAD      : {m.motion_mad:5.1f} LSB")
        lines.append(f"  Laplacian Var   : {m.laplacian_variance:7.1f}  (sharpness proxy)")

        lines.append(f"{C}{'─'*70}{R}")

        # ── Composite optical scores ───────────────────────────────────────────
        def bar(v: float, w: int = 20) -> str:
            filled = int(v * w)
            return f"[{'█'*filled}{'░'*(w-filled)}] {v*100:5.1f}%"

        stab_c = G if m.optical_stability > 0.7 else Y if m.optical_stability > 0.4 else R
        qual_c = G if m.image_quality > 0.7 else Y if m.image_quality > 0.4 else R
        conf_c = G if m.optical_confidence > 0.7 else Y if m.optical_confidence > 0.4 else R

        lines.append(f"  Optical Stability : {stab_c}{bar(m.optical_stability)}{R}")
        lines.append(f"  Image Quality     : {qual_c}{bar(m.image_quality)}{R}")
        lines.append(f"  Optical Confidence: {conf_c}{bar(m.optical_confidence)}{R}")

        lines.append(f"{C}{'─'*70}{R}")

        # ── Optical signal buffer ──────────────────────────────────────────────
        lines.append(f"  Signal Buffer   : {tracker.buffer_fill:3d} / {tracker._signal.maxlen} frames")
        lines.append(f"  Signal AC-RMS   : {tracker.signal_ac_rms:6.3f} LSB  (optical pulsatility proxy)")
        lines.append(f"  Signal SNR      : {tracker.signal_snr_db:6.1f} dB  (experimental)")
        lines.append(f"  Good Frame Ratio: {tracker.good_frame_ratio*100:5.1f}%")

        lines.append(f"{C}{'─'*70}{R}")

        # ── AI interpretation ──────────────────────────────────────────────────
        lines.append(f"{B}  AI-ASSISTED OPTICAL INTERPRETATION  (non-clinical){R}")
        interp = interpreter.interpret(m, tracker)
        for msg in interp[:6]:  # show up to 6 messages
            tag = msg[:6]
            colour = G if "[OK]" in tag else Y if "[INFO]" in tag or "[WARN]" in tag else R
            lines.append(f"    {colour}{msg}{R}")

        lines.append(f"{B}{C}{'═'*70}{R}")
        lines.append("  Press  Q  in the OpenCV window  or  Ctrl-C  to exit.")

        # Print all at once after clearing screen
        print(self._CLR + "\n".join(lines), end="", flush=True)


# ===========================================================================
# § 8  OPENCV VISUALISATION
# ===========================================================================

class IRVisualiser:
    """
    Real-time OpenCV window displaying the processed IR frame with research-
    grade overlays.

    Overlays:
      • Central ROI bounding rectangle
      • Optical signal time-series mini-plot (bottom strip)
      • Per-frame metric text
      • Motion / quality colour-coded badge
    """

    WINDOW_NAME = "NoIR BioSensor | 850nm IR Reflectance | EXPERIMENTAL"

    def __init__(self, cam_cfg: CameraConfig, processor: IRImageProcessor):
        self._cam = cam_cfg
        self._proc = processor
        self._initialized = False
        self._signal_history: Deque[float] = collections.deque(maxlen=200)

    def init(self) -> None:
        if CV2_AVAILABLE:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, self._cam.width, self._cam.height + 80)
            self._initialized = True

    def render(self, gray_disp: np.ndarray, metrics: FrameMetrics) -> bool:
        """
        Draw overlays and display frame.

        Returns False if the user pressed Q (quit signal).
        """
        if not CV2_AVAILABLE or not self._initialized:
            return True

        # Convert grayscale to BGR for coloured overlays
        vis = cv2.cvtColor(gray_disp, cv2.COLOR_GRAY2BGR)

        h, w = vis.shape[:2]

        # ── ROI rectangle ──────────────────────────────────────────────────────
        x1, y1, x2, y2 = self._proc.compute_roi_coords(h, w)
        roi_colour = (0, 220, 0) if metrics.finger_present else (0, 80, 200)
        cv2.rectangle(vis, (x1, y1), (x2, y2), roi_colour, 2)
        cv2.putText(vis, "ROI", (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_colour, 1)

        # ── Top-left metric overlay ────────────────────────────────────────────
        overlay_lines = [
            f"FPS: {metrics.fps:4.1f}",
            f"IR mean: {metrics.mean_intensity:5.1f} LSB",
            f"ROI mean: {metrics.roi_mean:5.1f} LSB",
            f"Stability: {metrics.optical_stability*100:4.0f}%",
            f"Confidence: {metrics.optical_confidence*100:4.0f}%",
            f"Motion: {metrics.motion_score:4.3f}",
        ]
        bg_h = len(overlay_lines) * 18 + 8
        overlay_bg = vis[4:4+bg_h, 4:180].copy()
        cv2.rectangle(vis, (4, 4), (180, 4 + bg_h), (10, 10, 10), -1)
        for i, txt in enumerate(overlay_lines):
            cv2.putText(vis, txt, (8, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 230, 200), 1)

        # ── Motion badge (top-right) ───────────────────────────────────────────
        if metrics.motion_score > 0.5:
            badge_colour = (0, 0, 210)
            badge_txt = "MOTION"
        elif not metrics.finger_present:
            badge_colour = (0, 120, 210)
            badge_txt = "NO TISSUE"
        else:
            badge_colour = (0, 150, 0)
            badge_txt = "STABLE"
        cv2.rectangle(vis, (w - 110, 4), (w - 4, 26), badge_colour, -1)
        cv2.putText(vis, badge_txt, (w - 108, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)

        # ── Bottom: optical signal strip plot ──────────────────────────────────
        self._signal_history.append(metrics.roi_mean)
        vis = self._draw_signal_strip(vis, w)

        # ── Watermark ─────────────────────────────────────────────────────────
        cv2.putText(vis, "EXPERIMENTAL — NON-CLINICAL", (4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 180), 1)

        cv2.imshow(self.WINDOW_NAME, vis)

        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), ord("Q"), 27)  # ESC or Q → quit

    def _draw_signal_strip(self, vis: np.ndarray, w: int) -> np.ndarray:
        """
        Append a 60-pixel-tall optical signal trace below the camera frame,
        forming a continuous combined image.
        """
        strip_h = 60
        strip = np.zeros((strip_h, w, 3), dtype=np.uint8)
        strip[:] = (15, 15, 15)  # near-black background

        sig = np.array(self._signal_history, dtype=np.float32)
        if len(sig) < 2:
            return np.vstack([vis, strip])

        # Normalise to strip height
        sig_min, sig_max = sig.min(), sig.max()
        if sig_max - sig_min < 1.0:
            sig_norm = np.full_like(sig, strip_h // 2)
        else:
            sig_norm = (sig - sig_min) / (sig_max - sig_min) * (strip_h - 8) + 4

        # Resample to frame width
        x_src = np.linspace(0, len(sig_norm) - 1, w)
        y_vals = np.interp(x_src, np.arange(len(sig_norm)), sig_norm).astype(int)

        # Draw polyline
        pts = np.array(
            [[xi, strip_h - 1 - yi] for xi, yi in enumerate(y_vals)],
            dtype=np.int32,
        )
        cv2.polylines(strip, [pts], False, (0, 200, 120), 1)

        # Label
        cv2.putText(strip, "Optical Signal (ROI mean, experimental)", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 180, 120), 1)

        return np.vstack([vis, strip])

    def close(self) -> None:
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()


# ===========================================================================
# § 9  MAIN SYSTEM CONTROLLER
# ===========================================================================

class NoIRBioSensorSystem:
    """
    Top-level coordinator that wires together all subsystems and runs the
    main processing loop.

    Thread model:
      • Main thread   — image processing, scoring, visualisation, dashboard
      • cam-acq thread — Picamera2 acquisition (started by CameraAcquisition)
    """

    def __init__(self) -> None:
        # Instantiate configuration objects
        self._cam_cfg = CameraConfig()
        self._opt_cfg = OpticalConfig()
        self._roi_cfg = ROIConfig()
        self._mot_cfg = MotionConfig()

        # Subsystems
        self._acquisition = CameraAcquisition(self._cam_cfg)
        self._processor = IRImageProcessor(
            self._cam_cfg, self._opt_cfg, self._roi_cfg, self._mot_cfg
        )
        self._tracker = OpticalSignalTracker(self._roi_cfg)
        self._interpreter = OpticalInterpreter()
        self._dashboard = TerminalDashboard(self._opt_cfg, self._cam_cfg)
        self._visualiser = IRVisualiser(self._cam_cfg, self._processor)

        # State
        self._running = False
        self._frame_index = 0
        self._last_dashboard_t = 0.0
        self._dashboard_interval = 0.25   # update terminal at 4 Hz

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._handle_sigint)
        signal.signal(signal.SIGTERM, self._handle_sigint)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start all subsystems and enter the main processing loop."""
        log.info("=" * 60)
        log.info("NoIR Biomedical Optical Sensing System  —  STARTING")
        log.info("IR wavelength: %d nm  |  Resolution: %dx%d  |  FPS target: %d",
                 self._opt_cfg.ir_wavelength_nm,
                 self._cam_cfg.width, self._cam_cfg.height,
                 self._cam_cfg.target_fps)
        log.info("DISCLAIMER: Experimental optical sensing only.  Non-clinical.")
        log.info("=" * 60)

        self._acquisition.start()
        self._visualiser.init()
        self._running = True

        last_metrics = FrameMetrics()

        try:
            while self._running:
                frame = self._acquisition.get_frame(timeout=0.5)
                if frame is None:
                    continue

                self._frame_index += 1

                # Full image processing pipeline
                gray_raw, gray_disp, metrics = self._processor.process(frame)
                metrics.frame_index = self._frame_index
                metrics.fps = self._acquisition.fps

                # Update optical signal tracker
                self._tracker.update(metrics)

                # Update terminal dashboard at reduced rate (CPU saving)
                now = time.monotonic()
                if now - self._last_dashboard_t >= self._dashboard_interval:
                    self._dashboard.render(metrics, self._tracker,
                                           self._interpreter,
                                           PICAMERA2_AVAILABLE)
                    self._last_dashboard_t = now

                # Visualisation (every frame)
                if CV2_AVAILABLE:
                    keep_open = self._visualiser.render(gray_disp, metrics)
                    if not keep_open:
                        log.info("User requested exit from visualisation window.")
                        self._running = False

                last_metrics = metrics

        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        log.info("Shutting down NoIR Optical Sensing System …")
        self._running = False
        self._acquisition.stop()
        self._visualiser.close()
        log.info("Session ended.  Total frames processed: %d", self._frame_index)

    def _handle_sigint(self, sig, frame) -> None:
        log.info("Interrupt received — initiating graceful shutdown.")
        self._running = False


# ===========================================================================
# § 10  ENTRY POINT
# ===========================================================================

def print_startup_banner() -> None:
    """Print a professional startup banner to stdout."""
    banner = r"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║         NoIR BIOMEDICAL OPTICAL SENSING SYSTEM                  ║
  ║         Raspberry Pi NoIR Camera — 850 nm IR Reflectance        ║
  ║         Research Prototype  ·  NON-CLINICAL USE ONLY            ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  Illumination : 850 nm IR LEDs                                  ║
  ║  Modality     : Diffuse reflectance optical sensing             ║
  ║  Target tissue: Fingertip (surface + shallow subsurface)        ║
  ║  Penetration  : ~1–3 mm (physically realistic NIR)              ║
  ║  Output       : Optical signal + image quality metrics          ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  SCIENTIFIC LIMITATIONS                                         ║
  ║  • NOT deep-tissue imaging   • NOT clinical diagnosis           ║
  ║  • NOT X-ray or MRI          • Surface optical reflection ONLY  ║
  ╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_dependencies() -> bool:
    """Verify required libraries and emit warnings for missing ones."""
    ok = True
    if not CV2_AVAILABLE:
        log.warning("OpenCV not available — install with:  pip install opencv-python-headless")
        ok = False
    if not PICAMERA2_AVAILABLE:
        log.warning(
            "Picamera2 not available — running in SIMULATION mode.  "
            "Install on Raspberry Pi with:  sudo apt install python3-picamera2"
        )
    return ok


if __name__ == "__main__":
    print_startup_banner()
    check_dependencies()

    system = NoIRBioSensorSystem()
    system.run()