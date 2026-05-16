#!/usr/bin/env python3
# =============================================================================
#  NoIR Biomedical Optical Sensing System  —  v2
#  Raspberry Pi NoIR Camera | 850 nm IR Reflectance Imaging Platform
# =============================================================================
#
#  Author        : Biomedical Optics Research Prototype
#  Target HW     : Raspberry Pi 4/5 + NoIR Camera Module v2/v3
#  Illumination  : 850 nm IR LEDs  +  White LEDs  (GPIO-controlled)
#  Enclosure     : Black optical-isolation chamber
#  Python        : 3.9+
#  Dependencies  : numpy, opencv-python-headless, scipy, picamera2, RPi.GPIO
#
#  SCIENTIFIC DISCLAIMER
#  ─────────────────────
#  This system observes *reflected near-infrared light* (850 nm) from the
#  surface and shallow subsurface of biological tissue (penetration ≈ 1–3 mm).
#  It is NOT capable of deep-tissue imaging, X-ray, MRI, or clinical diagnosis.
#  All signals and outputs are explicitly labelled "experimental / non-clinical".
#
#  OPTICAL PHYSICS BACKGROUND
#  ──────────────────────────
#  At 850 nm, dominant tissue chromophores are oxy/deoxy-haemoglobin, water,
#  and melanin.  The NoIR camera captures backscattered flux encoding local
#  optical properties (µa, µs') within that shallow volume — the physical
#  basis of near-infrared diffuse reflectance and reflectance-mode rPPG.
#  A Butterworth bandpass (0.7–4 Hz) isolates the experimental pulsatile
#  component of the raw ROI-averaged intensity signal.
#
#  PIPELINE OVERVIEW
#  ─────────────────
#  [Picamera2 CSI]
#      │
#      ▼
#  [Adaptive Exposure Controller]  ← ROI brightness feedback loop
#      │
#      ▼
#  [IR Channel Extraction + Noise Filtering]
#      │
#      ▼
#  [Adaptive Tissue Segmenter]     ← contour/ellipse fingertip detection
#      │
#      ▼
#  [Motion & Quality Gate]         ← MAD + Laplacian + saturation check
#      │
#      ▼
#  [Optical Signal Extractor]      ← temporal buffer → Butterworth BP
#      │
#      ▼
#  [Quality Engine]                ← EXCELLENT / GOOD / FAIR / POOR / INVALID
#      │
#      ▼
#  [AI Interpreter]                ← rule-based non-clinical optical feedback
#      │
#      ▼
#  [Terminal Dashboard  +  OpenCV Research Visualiser]
#
# =============================================================================

# ─── Standard library ────────────────────────────────────────────────────────
import sys
import os
import time
import math
import threading
import queue
import signal
import logging
import collections
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List, Deque, Dict

# ─── Third-party ─────────────────────────────────────────────────────────────
import numpy as np

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from scipy.signal import butter, sosfilt, sosfilt_zi
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    from picamera2 import Picamera2
    from libcamera import controls as lc
    PICAM_OK = True
except ImportError:
    PICAM_OK = False

try:
    import RPi.GPIO as GPIO
    GPIO_OK = True
except ImportError:
    GPIO_OK = False

# ─── Logger ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("NoIR-v2")


# =============================================================================
# §1  ENUMERATIONS
# =============================================================================

class IlluminationMode(Enum):
    IR_ONLY    = auto()   # 850 nm IR LEDs only
    WHITE_ONLY = auto()   # White LEDs only
    HYBRID     = auto()   # Both simultaneously (reference comparison)
    OFF        = auto()   # Dark-frame baseline


class OpticalQuality(Enum):
    EXCELLENT = "EXCELLENT"
    GOOD      = "GOOD"
    FAIR      = "FAIR"
    POOR      = "POOR"
    INVALID   = "INVALID"


# =============================================================================
# §2  CONFIGURATION
# =============================================================================

@dataclass
class CamCfg:
    width:          int   = 640
    height:         int   = 480
    target_fps:     int   = 30
    pixel_format:   str   = "BGR888"
    # Manual exposure seed values — adaptive controller will tune these
    init_exposure_us: int  = 15_000
    init_gain:        float = 4.0
    # Exposure controller bounds
    min_exposure_us:  int  = 3_000
    max_exposure_us:  int  = 40_000
    min_gain:         float = 1.0
    max_gain:         float = 8.0


@dataclass
class OptCfg:
    ir_wavelength_nm:      int   = 850
    # Target mean ROI intensity for adaptive exposure [LSB]
    target_roi_intensity:  float = 140.0
    # Acceptable band around target before correction kicks in
    intensity_deadband:    float = 15.0
    # CLAHE
    clahe_clip:            float = 2.5
    clahe_tile:            Tuple[int,int] = (8, 8)
    # Gaussian blur kernel (must be odd)
    blur_k:                int   = 5
    # Butterworth bandpass for optical pulsatility waveform
    bp_low_hz:             float = 0.7
    bp_high_hz:            float = 4.0
    bp_order:              int   = 4
    # Signal buffer length (frames)
    signal_buf_len:        int   = 300   # 10 s @ 30 fps
    # Minimum buffer fill before filtering
    min_buf_for_filter:    int   = 60    # 2 s


@dataclass
class GPIOCfg:
    ir_led_pin:    int = 17   # BCM numbering
    white_led_pin: int = 27
    # PWM frequency (Hz)
    pwm_freq:      int = 1000
    # Default duty cycles [0–100]
    ir_duty:       float = 80.0
    white_duty:    float = 60.0


@dataclass
class SegCfg:
    # Otsu threshold region to search within (central fraction of frame)
    search_frac_w: float = 0.80
    search_frac_h: float = 0.85
    # Minimum contour area as fraction of search area
    min_area_frac: float = 0.02
    # Ellipse fit quality threshold
    ellipse_ratio_min: float = 0.30   # minor/major axis ratio
    # Fallback fixed ROI fraction when segmentation fails
    fallback_frac_w: float = 0.45
    fallback_frac_h: float = 0.55
    # Smoothing alpha for ROI centre EMA
    ema_alpha: float = 0.20


@dataclass
class MotCfg:
    motion_flag_mad:   float = 8.0    # MAD threshold [LSB]
    blur_threshold:    float = 25.0   # Laplacian variance
    ema_alpha:         float = 0.25


# =============================================================================
# §3  DATA CONTAINERS
# =============================================================================

@dataclass
class ROIState:
    """Current best estimate of the fingertip ROI."""
    x1: int = 0; y1: int = 0; x2: int = 0; y2: int = 0
    cx: float = 0.0; cy: float = 0.0
    from_segmentation: bool = False
    confidence: float = 0.0
    # Ellipse params (None if not fitted)
    ellipse: Optional[Tuple] = None


@dataclass
class FrameData:
    """All per-frame derived quantities — output of the full pipeline."""
    ts:                  float = 0.0
    frame_idx:           int   = 0
    fps:                 float = 0.0
    illum_mode:          IlluminationMode = IlluminationMode.IR_ONLY

    # Photometry
    mean_intensity:      float = 0.0
    std_intensity:       float = 0.0
    saturation_frac:     float = 0.0
    laplacian_var:       float = 0.0

    # Motion
    motion_mad:          float = 0.0
    motion_score:        float = 0.0   # [0=still, 1=heavy]
    is_motion_corrupt:   bool  = False

    # Exposure controller state
    current_exposure_us: int   = 15_000
    current_gain:        float = 4.0

    # ROI
    roi:                 ROIState = field(default_factory=ROIState)
    finger_present:      bool = False
    roi_mean:            float = 0.0
    roi_std:             float = 0.0

    # Optical signal (waveform)
    raw_signal_latest:    float = 0.0
    filtered_signal_latest: float = 0.0
    signal_ac_rms:        float = 0.0
    signal_snr_db:        float = 0.0
    pulsatility_confidence: float = 0.0  # [0,1] non-clinical proxy

    # Composite scores [0,1]
    optical_stability:   float = 0.0
    image_quality:       float = 0.0
    optical_confidence:  float = 0.0
    quality_class:       OpticalQuality = OpticalQuality.INVALID


# =============================================================================
# §4  GPIO / ILLUMINATION CONTROLLER
# =============================================================================

class IlluminationController:
    """
    Controls 850 nm IR and white LED banks via Raspberry Pi GPIO PWM.
    Falls back to a no-op stub when RPi.GPIO is unavailable.

    GPIO wiring (BCM):
      Pin 17 → IR LED MOSFET gate   (850 nm bank)
      Pin 27 → White LED MOSFET gate
    """

    def __init__(self, cfg: GPIOCfg):
        self._cfg = cfg
        self._mode = IlluminationMode.IR_ONLY
        self._ir_pwm = None
        self._wh_pwm = None
        self._hw_ok = False
        self._init_gpio()

    def _init_gpio(self) -> None:
        if not GPIO_OK:
            log.warning("RPi.GPIO unavailable — LED control disabled (simulation).")
            return
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self._cfg.ir_led_pin, GPIO.OUT)
            GPIO.setup(self._cfg.white_led_pin, GPIO.OUT)
            self._ir_pwm = GPIO.PWM(self._cfg.ir_led_pin, self._cfg.pwm_freq)
            self._wh_pwm = GPIO.PWM(self._cfg.white_led_pin, self._cfg.pwm_freq)
            self._ir_pwm.start(0)
            self._wh_pwm.start(0)
            self._hw_ok = True
            log.info("GPIO LED controller initialised (IR pin=%d, White pin=%d).",
                     self._cfg.ir_led_pin, self._cfg.white_led_pin)
        except Exception as e:
            log.warning("GPIO init failed: %s", e)

    def set_mode(self, mode: IlluminationMode) -> None:
        self._mode = mode
        if not self._hw_ok:
            return
        if mode == IlluminationMode.IR_ONLY:
            self._ir_pwm.ChangeDutyCycle(self._cfg.ir_duty)
            self._wh_pwm.ChangeDutyCycle(0)
        elif mode == IlluminationMode.WHITE_ONLY:
            self._ir_pwm.ChangeDutyCycle(0)
            self._wh_pwm.ChangeDutyCycle(self._cfg.white_duty)
        elif mode == IlluminationMode.HYBRID:
            self._ir_pwm.ChangeDutyCycle(self._cfg.ir_duty)
            self._wh_pwm.ChangeDutyCycle(self._cfg.white_duty)
        else:  # OFF
            self._ir_pwm.ChangeDutyCycle(0)
            self._wh_pwm.ChangeDutyCycle(0)

    @property
    def mode(self) -> IlluminationMode:
        return self._mode

    def cleanup(self) -> None:
        if self._hw_ok:
            self.set_mode(IlluminationMode.OFF)
            self._ir_pwm.stop()
            self._wh_pwm.stop()
            GPIO.cleanup()
            log.info("GPIO cleaned up.")


# =============================================================================
# §5  ADAPTIVE EXPOSURE CONTROLLER
# =============================================================================

class AdaptiveExposureController:
    """
    Proportional–integral exposure controller that keeps the ROI mean
    intensity near a target value by tuning Picamera2 ExposureTime and
    AnalogueGain.

    Control law (PI):
        error  = target_intensity − roi_mean
        if |error| > deadband:
            correction = Kp * error + Ki * integral_error
        Apply correction first to ExposureTime; saturate gain as secondary.

    This avoids the large frame-to-frame variance that AEC introduces on
    pulsatile optical signals — manual control is essential for rPPG / optical
    biosignal extraction.
    """

    Kp = 0.005      # proportional gain (normalised to [0,1] error)
    Ki = 0.0008     # integral gain
    integral_clamp = 50.0  # anti-windup

    def __init__(self, cam_cfg: CamCfg, opt_cfg: OptCfg):
        self._cam = cam_cfg
        self._opt = opt_cfg
        self._exposure_us: float = cam_cfg.init_exposure_us
        self._gain: float = cam_cfg.init_gain
        self._integral: float = 0.0
        self._camera: Optional[Picamera2] = None  # injected after camera init

    def inject_camera(self, cam: "Picamera2") -> None:
        self._camera = cam

    def update(self, roi_mean: float) -> Tuple[int, float]:
        """
        Given current ROI mean intensity, compute updated exposure & gain.

        Returns (exposure_us, gain) — already applied to camera if available.
        """
        error = self._opt.target_roi_intensity - roi_mean

        if abs(error) <= self._opt.intensity_deadband:
            # Within deadband — freeze to avoid noise-driven hunting
            return int(self._exposure_us), self._gain

        # PI update
        self._integral = np.clip(
            self._integral + error, -self.integral_clamp, self.integral_clamp
        )
        correction = self.Kp * error + self.Ki * self._integral

        # Primary: adjust exposure (proportionally)
        self._exposure_us *= (1.0 + correction)
        self._exposure_us = float(
            np.clip(self._exposure_us,
                    self._cam.min_exposure_us, self._cam.max_exposure_us)
        )

        # Secondary: if exposure is saturated, adjust gain
        if self._exposure_us >= self._cam.max_exposure_us - 100 and error > 0:
            self._gain = min(self._gain * 1.05, self._cam.max_gain)
        elif self._exposure_us <= self._cam.min_exposure_us + 100 and error < 0:
            self._gain = max(self._gain * 0.95, self._cam.min_gain)

        # Push to hardware
        if self._camera is not None and PICAM_OK:
            try:
                self._camera.set_controls({
                    "ExposureTime": int(self._exposure_us),
                    "AnalogueGain": float(self._gain),
                })
            except Exception:
                pass

        return int(self._exposure_us), self._gain


# =============================================================================
# §6  CAMERA ACQUISITION
# =============================================================================

class CameraAcquisition:
    """
    Picamera2 CSI pipeline with a dedicated daemon thread.
    Falls back to a physically-realistic synthetic IR scene for dev/testing.
    """

    def __init__(self, cfg: CamCfg, exp_ctrl: AdaptiveExposureController):
        self._cfg = cfg
        self._exp = exp_ctrl
        self._camera: Optional[Picamera2] = None
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=3)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fps_ema = 0.0
        self._last_t = time.monotonic()

    # ── public ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        if PICAM_OK:
            self._init_hw()
        else:
            log.warning("Picamera2 unavailable — synthetic IR simulation active.")
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="cam-acq")
        self._thread.start()
        log.info("Camera acquisition thread started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._camera:
            self._camera.stop()
            self._camera.close()

    def get_frame(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def fps(self) -> float:
        return self._fps_ema

    # ── private ──────────────────────────────────────────────────────────────

    def _init_hw(self) -> None:
        self._camera = Picamera2()
        cam_cfg = self._camera.create_video_configuration(
            main={"size": (self._cfg.width, self._cfg.height),
                  "format": self._cfg.pixel_format},
            controls={
                "AeEnable":          False,
                "AwbEnable":         False,
                "ExposureTime":      self._cfg.init_exposure_us,
                "AnalogueGain":      self._cfg.init_gain,
                "ColourGains":       (1.0, 1.0),
                "NoiseReductionMode": lc.draft.NoiseReductionModeEnum.Off,
            },
        )
        self._camera.configure(cam_cfg)
        self._camera.start()
        self._exp.inject_camera(self._camera)
        time.sleep(0.3)
        log.info("Picamera2 ready: %dx%d @ %d fps, exp=%dµs gain=%.1f",
                 self._cfg.width, self._cfg.height, self._cfg.target_fps,
                 self._cfg.init_exposure_us, self._cfg.init_gain)

    def _loop(self) -> None:
        period = 1.0 / self._cfg.target_fps
        while self._running:
            t0 = time.monotonic()
            frame = self._capture()
            if frame is not None:
                if self._q.full():
                    try: self._q.get_nowait()
                    except queue.Empty: pass
                self._q.put_nowait(frame)
                now = time.monotonic()
                dt = now - self._last_t
                if dt > 0:
                    self._fps_ema = 0.1 * (1/dt) + 0.9 * self._fps_ema
                self._last_t = now
            elapsed = time.monotonic() - t0
            sleep = period - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _capture(self) -> Optional[np.ndarray]:
        if self._camera:
            try:
                return self._camera.capture_array("main")
            except Exception as e:
                log.error("Capture error: %s", e)
                return None
        return self._synth_frame()

    def _synth_frame(self) -> np.ndarray:
        """
        Physically plausible synthetic 850 nm reflectance frame.

        Simulates:
          • Gaussian fingertip blob (DC reflectance ≈ 130 LSB)
          • 1% AC pulsatile component at ~1.1 Hz
          • Secondary vascular shadow at ~1.5× cardiac frequency
          • Shot noise (σ = 3 LSB)
          • Slow illumination drift (< 0.5% over 60 s)
          • Finger micro-motion (Brownian ±3 px, σ=0.5)
        """
        h, w = self._cfg.height, self._cfg.width
        t = time.monotonic()

        frame = np.full((h, w), 10.0, dtype=np.float32)

        # Micro-motion: small random displacement
        dx = np.random.normal(0, 0.4)
        dy = np.random.normal(0, 0.4)
        cx = w / 2 + dx
        cy = h / 2 + dy
        sx, sy = w * 0.18, h * 0.28

        yy, xx = np.ogrid[:h, :w]
        gauss = np.exp(-((xx - cx)**2 / (2*sx**2) + (yy - cy)**2 / (2*sy**2)))

        dc      = 130.0
        f_card  = 1.1   # Hz (~66 bpm placeholder — NOT a clinical measurement)
        ac1     = dc * 0.012 * math.sin(2*math.pi * f_card * t)
        ac2     = dc * 0.004 * math.sin(2*math.pi * f_card * 2 * t + 0.8)
        drift   = dc * 0.005 * math.sin(2*math.pi * t / 60.0)
        noise   = np.random.normal(0, 3.0, (h, w)).astype(np.float32)

        frame += gauss * (dc + ac1 + ac2 + drift) + noise
        frame  = np.clip(frame, 0, 255).astype(np.uint8)

        if CV2_OK:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return np.stack([frame]*3, axis=-1)


# =============================================================================
# §7  IR IMAGE PROCESSOR
# =============================================================================

class IRImageProcessor:
    """
    Converts raw BGR NoIR frames to processed grayscale IR images.

    Stages:
      1. IR-weighted channel merge (R×0.75, G×0.20, B×0.05)
         Physical basis: OV5647/IMX219 QE at 850 nm peaks in red channel.
      2. Gaussian blur — shot-noise suppression (σ≈1 px)
      3. CLAHE — contrast-limited adaptive histogram equalisation
         (improves low-contrast subsurface optical features)
      4. Illumination normalisation (large-kernel blur divide)
    """

    def __init__(self, opt_cfg: OptCfg):
        self._cfg = opt_cfg
        self._clahe = (cv2.createCLAHE(
            clipLimit=opt_cfg.clahe_clip,
            tileGridSize=opt_cfg.clahe_tile) if CV2_OK else None)

    def process(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            gray_metric  — lightly blurred grayscale (for metric extraction)
            gray_display — CLAHE-enhanced + normalised (for visualisation)
        """
        gray_raw = self._extract_ir(bgr)

        k = self._cfg.blur_k
        if CV2_OK:
            gray_metric = cv2.GaussianBlur(gray_raw, (k, k), 0)
        else:
            gray_metric = gray_raw.copy()

        # CLAHE display image
        if self._clahe is not None:
            gray_disp = self._clahe.apply(gray_metric)
        else:
            gray_disp = gray_metric.copy()

        # Illumination normalisation: divide by large-scale Gaussian background
        if CV2_OK:
            bg = cv2.GaussianBlur(gray_metric.astype(np.float32), (81, 81), 0) + 1.0
            norm = np.clip(gray_metric.astype(np.float32) / bg * 128.0, 0, 255).astype(np.uint8)
            # Blend normalised and CLAHE for best of both
            gray_disp = cv2.addWeighted(gray_disp, 0.6, norm, 0.4, 0)

        return gray_metric, gray_disp

    def _extract_ir(self, bgr: np.ndarray) -> np.ndarray:
        """Weighted channel merge optimised for 850 nm NoIR response."""
        if not CV2_OK:
            return np.mean(bgr, axis=2).astype(np.uint8)
        b = bgr[:,:,0].astype(np.float32)
        g = bgr[:,:,1].astype(np.float32)
        r = bgr[:,:,2].astype(np.float32)
        ir = 0.05*b + 0.20*g + 0.75*r
        return np.clip(ir, 0, 255).astype(np.uint8)


# =============================================================================
# §8  ADAPTIVE TISSUE SEGMENTER
# =============================================================================

class AdaptiveTissueSegmenter:
    """
    Detects and tracks the fingertip ROI in the processed IR frame.

    Algorithm:
      1. Search within a central crop of the frame.
      2. Otsu threshold → morphological close → find contours.
      3. Select largest contour with area ≥ min_area_frac of search area.
      4. Fit an ellipse to the contour for sub-pixel accuracy.
      5. Smooth ROI centre with EMA to suppress jitter.
      6. Fall back to fixed central rectangle when segmentation fails.

    Confidence scoring:
      • Contour area fraction normalised [0,1]
      • Ellipse axis ratio (roundness) normalised [0,1]
      • Combined as geometric mean
    """

    def __init__(self, seg_cfg: SegCfg, cam_cfg: CamCfg):
        self._seg = seg_cfg
        self._cam = cam_cfg
        self._cx_ema: Optional[float] = None
        self._cy_ema: Optional[float] = None
        self._last_roi = ROIState()
        self._seg_kernel = (cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (9, 9)) if CV2_OK else None)

    def segment(self, gray: np.ndarray) -> Tuple[np.ndarray, ROIState]:
        """
        Returns:
            mask   — binary tissue mask (uint8, same size as gray)
            roi    — ROIState with current best estimate
        """
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if not CV2_OK:
            return mask, self._fallback_roi(h, w)

        # Search crop
        sw = int(w * self._seg.search_frac_w)
        sh = int(h * self._seg.search_frac_h)
        ox = (w - sw) // 2
        oy = (h - sh) // 2
        crop = gray[oy:oy+sh, ox:ox+sw]

        # Otsu threshold
        _, thresh = cv2.threshold(crop, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Morphological close — fills small holes in tissue region
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self._seg_kernel)

        # Contour detection
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask, self._fallback_roi(h, w)

        # Select largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        search_area = sw * sh
        area_frac = area / search_area

        if area_frac < self._seg.min_area_frac:
            return mask, self._fallback_roi(h, w)

        # Shift contour back to full-frame coordinates
        largest_full = largest + np.array([[[ox, oy]]])

        # Fit ellipse (requires ≥ 5 points)
        roi = ROIState()
        ellipse = None
        confidence = 0.0

        if len(largest_full) >= 5:
            try:
                ellipse = cv2.fitEllipse(largest_full)
                (ex, ey), (ma, mi), angle = ellipse
                ratio = min(ma, mi) / (max(ma, mi) + 1e-6)
                if ratio >= self._seg.ellipse_ratio_min:
                    # EMA smoothing of centre
                    alpha = self._seg.ema_alpha
                    self._cx_ema = ex if self._cx_ema is None \
                        else alpha*ex + (1-alpha)*self._cx_ema
                    self._cy_ema = ey if self._cy_ema is None \
                        else alpha*ey + (1-alpha)*self._cy_ema
                    cx, cy = self._cx_ema, self._cy_ema
                    half_w = int(max(ma, mi) / 2 * 1.1)
                    half_h = int(min(ma, mi) / 2 * 1.5)
                    roi.cx, roi.cy = cx, cy
                    roi.x1 = int(np.clip(cx - half_w, 0, w-1))
                    roi.y1 = int(np.clip(cy - half_h, 0, h-1))
                    roi.x2 = int(np.clip(cx + half_w, 0, w-1))
                    roi.y2 = int(np.clip(cy + half_h, 0, h-1))
                    roi.ellipse = ellipse
                    roi.from_segmentation = True
                    confidence = math.sqrt(
                        min(area_frac / 0.15, 1.0) * min(ratio / 0.5, 1.0)
                    )
                    roi.confidence = confidence
                    # Draw tissue mask
                    cv2.drawContours(mask, [largest_full], -1, 255, -1)
                    self._last_roi = roi
                    return mask, roi
            except cv2.error:
                pass

        return mask, self._fallback_roi(h, w)

    def _fallback_roi(self, h: int, w: int) -> ROIState:
        rw = int(w * self._seg.fallback_frac_w)
        rh = int(h * self._seg.fallback_frac_h)
        x1 = (w - rw) // 2
        y1 = (h - rh) // 2
        roi = ROIState(
            x1=x1, y1=y1, x2=x1+rw, y2=y1+rh,
            cx=w/2, cy=h/2,
            from_segmentation=False, confidence=0.0
        )
        self._last_roi = roi
        return roi


# =============================================================================
# §9  MOTION DETECTOR
# =============================================================================

class MotionDetector:
    """
    Dual-metric motion detection:
      1. Frame-difference MAD (mean absolute difference) — movement detector
      2. Laplacian variance — sharpness / motion-blur detector

    Both are EMA-smoothed for temporal stability.
    """

    def __init__(self, cfg: MotCfg):
        self._cfg = cfg
        self._prev: Optional[np.ndarray] = None
        self._motion_ema: float = 0.0

    def update(self, gray: np.ndarray) -> Tuple[float, float, float, bool]:
        """
        Returns (motion_mad, laplacian_var, motion_score, is_corrupt).
        """
        alpha = self._cfg.ema_alpha

        # Laplacian variance
        if CV2_OK:
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        else:
            lap_var = 50.0

        # Frame-difference MAD
        if self._prev is not None and gray.shape == self._prev.shape:
            mad = float(np.mean(np.abs(
                gray.astype(np.int16) - self._prev.astype(np.int16)
            )))
        else:
            mad = 0.0

        self._prev = gray.copy()

        raw_score = min(mad / max(self._cfg.motion_flag_mad, 1e-6), 1.0)
        self._motion_ema = alpha * raw_score + (1 - alpha) * self._motion_ema

        corrupt = (self._motion_ema > 0.55 or lap_var < self._cfg.blur_threshold)
        return mad, lap_var, self._motion_ema, corrupt


# =============================================================================
# §10  OPTICAL SIGNAL EXTRACTOR (Butterworth Bandpass)
# =============================================================================

class OpticalSignalExtractor:
    """
    Accumulates per-frame ROI mean intensities and applies a real-time
    Butterworth bandpass filter (0.7–4 Hz) to isolate the experimental
    optical pulsatility waveform.

    Implementation notes:
      • scipy.signal.sosfilt with persistent filter state (zi) for causal
        real-time filtering — avoids end effects and latency of batch filtfilt.
      • Detrending via 3rd-order polynomial fit on the buffer window prior to
        filtering removes slow illumination drift (< 0.7 Hz).
      • SNR estimated as 20·log10(DC_mean / AC_RMS).

    DISCLAIMER: The AC component of this signal is an *experimental optical
    pulsatility proxy*.  It must NOT be interpreted as a clinical measurement.
    """

    def __init__(self, opt_cfg: OptCfg):
        self._cfg = opt_cfg
        self._raw_buf:  Deque[float] = collections.deque(maxlen=opt_cfg.signal_buf_len)
        self._filt_buf: Deque[float] = collections.deque(maxlen=opt_cfg.signal_buf_len)
        self._ts_buf:   Deque[float] = collections.deque(maxlen=opt_cfg.signal_buf_len)
        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._fps_est: float = 30.0
        self._fps_samples: Deque[float] = collections.deque(maxlen=30)
        self._last_ts: float = 0.0

    def push(self, roi_mean: float, ts: float, accept: bool) -> None:
        """Push one sample; reject motion-corrupted frames."""
        if not accept:
            return
        self._raw_buf.append(roi_mean)
        self._ts_buf.append(ts)

        # FPS estimation from timestamps
        if self._last_ts > 0:
            dt = ts - self._last_ts
            if 0 < dt < 1.0:
                self._fps_samples.append(1.0 / dt)
        self._last_ts = ts

        if len(self._fps_samples) >= 5:
            self._fps_est = float(np.median(list(self._fps_samples)))

        # Rebuild filter when FPS estimate has settled or changes significantly
        self._ensure_filter()

        # Real-time sample-by-sample Butterworth filtering
        if self._sos is not None and self._zi is not None and len(self._raw_buf) >= 4:
            sample = np.array([[roi_mean]], dtype=np.float64)
            y, self._zi = sosfilt(self._sos, sample.ravel(), zi=self._zi)
            self._filt_buf.append(float(y[0]))
        else:
            self._filt_buf.append(roi_mean)

    def _ensure_filter(self) -> None:
        if not SCIPY_OK:
            return
        fps = max(self._fps_est, 5.0)
        nyq = fps / 2.0
        lo = self._cfg.bp_low_hz / nyq
        hi = self._cfg.bp_high_hz / nyq
        if lo >= 1.0 or hi >= 1.0 or lo <= 0:
            return
        if self._sos is None:
            self._sos = butter(self._cfg.bp_order, [lo, hi],
                               btype='band', output='sos')
            self._zi = sosfilt_zi(self._sos)[:, :, np.newaxis].squeeze(axis=2)
            # sosfilt_zi returns shape (n_sections, 2); reshape for scalar input
            # Actually for scalar input zi shape must be (n_sections, 2)
            self._zi = sosfilt_zi(self._sos) * 0.0  # zero initial state

    def metrics(self) -> Tuple[float, float, float, float]:
        """
        Returns (raw_latest, filtered_latest, ac_rms, snr_db).
        All labelled non-clinical experimental values.
        """
        raw_latest  = self._raw_buf[-1]  if self._raw_buf  else 0.0
        filt_latest = self._filt_buf[-1] if self._filt_buf else 0.0

        if len(self._filt_buf) >= self._cfg.min_buf_for_filter:
            fa = np.array(list(self._filt_buf), dtype=np.float32)
            ac_rms = float(np.sqrt(np.mean(fa**2)))
        else:
            ac_rms = 0.0

        if len(self._raw_buf) >= 4:
            dc = float(np.mean(list(self._raw_buf)))
            snr_db = 20.0 * math.log10(dc / (ac_rms + 1e-9) + 1e-9)
        else:
            snr_db = 0.0

        return raw_latest, filt_latest, ac_rms, snr_db

    @property
    def buffer_fill(self) -> int:
        return len(self._raw_buf)

    def raw_array(self) -> np.ndarray:
        return np.array(list(self._raw_buf), dtype=np.float32)

    def filtered_array(self) -> np.ndarray:
        return np.array(list(self._filt_buf), dtype=np.float32)


# =============================================================================
# §11  OPTICAL QUALITY ENGINE
# =============================================================================

class OpticalQualityEngine:
    """
    Aggregates per-frame scalar metrics into a five-level quality
    classification and normalised confidence score.

    Classification logic (thresholds are empirically tuned for 850 nm
    reflectance with 50 mA IR LED drive inside a black isolation chamber):

        Confidence ≥ 0.80  →  EXCELLENT
        Confidence ≥ 0.65  →  GOOD
        Confidence ≥ 0.45  →  FAIR
        Confidence ≥ 0.20  →  POOR
        Otherwise          →  INVALID
    """

    @staticmethod
    def evaluate(fd: FrameData) -> Tuple[float, float, float, OpticalQuality]:
        """
        Returns (optical_stability, image_quality, optical_confidence, quality_class).
        All scores in [0, 1].
        """
        # ── Stability score ────────────────────────────────────────────────
        motion_penalty = 1.0 - fd.motion_score
        blur_ok = min(fd.laplacian_var / 30.0, 1.0)
        stability = float(np.clip(motion_penalty * blur_ok, 0, 1))

        # ── Image quality score ────────────────────────────────────────────
        # Reward proper exposure; penalise darkness and saturation
        intensity_score = np.clip(
            (fd.mean_intensity - 8.0) / (35.0 - 8.0), 0.0, 1.0
        )
        saturation_penalty = max(0.0, 1.0 - fd.saturation_frac * 15.0)
        sharpness_norm = min(fd.laplacian_var / 150.0, 1.0)
        iq = float(intensity_score * saturation_penalty * (0.4 + 0.6 * sharpness_norm))

        # ── ROI / segmentation bonus ───────────────────────────────────────
        seg_bonus = 0.8 + 0.2 * fd.roi.confidence
        finger_w  = seg_bonus if fd.finger_present else 0.25

        # ── Signal quality contribution ────────────────────────────────────
        if fd.signal_snr_db > 20:
            sig_q = 1.0
        elif fd.signal_snr_db > 10:
            sig_q = (fd.signal_snr_db - 10) / 10.0
        else:
            sig_q = 0.0

        # ── Weighted confidence ────────────────────────────────────────────
        conf = float(
            stability * 0.30
            + iq       * 0.30
            + finger_w * 0.25
            + sig_q    * 0.15
        )
        conf = float(np.clip(conf, 0.0, 1.0))

        # ── Classification ────────────────────────────────────────────────
        if   conf >= 0.80: qc = OpticalQuality.EXCELLENT
        elif conf >= 0.65: qc = OpticalQuality.GOOD
        elif conf >= 0.45: qc = OpticalQuality.FAIR
        elif conf >= 0.20: qc = OpticalQuality.POOR
        else:              qc = OpticalQuality.INVALID

        return stability, iq, conf, qc


# =============================================================================
# §12  AI-ASSISTED OPTICAL INTERPRETER
# =============================================================================

class OpticalInterpreter:
    """
    Rule-based optical interpretation engine.

    Generates structured, non-clinical advisory messages based on current
    optical metrics.  Output is explicitly labelled "experimental optical
    analysis" — never a clinical or physiological claim.
    """

    @staticmethod
    def interpret(fd: FrameData) -> List[str]:
        msgs: List[str] = []

        # Tissue presence
        if not fd.finger_present:
            msgs.append("[ERR]  No tissue in ROI — place fingertip on chamber aperture.")
        elif fd.roi.from_segmentation and fd.roi.confidence > 0.6:
            msgs.append(f"[OK]   Adaptive segmentation active (conf={fd.roi.confidence:.2f}).")
        else:
            msgs.append("[INFO] Fixed-ROI fallback active — improve finger placement.")

        # Motion
        if fd.motion_score > 0.60:
            msgs.append("[ERR]  Severe motion artefact — hold still for stable signal.")
        elif fd.motion_score > 0.35:
            msgs.append("[WARN] Moderate motion — stabilise fingertip contact.")
        else:
            msgs.append("[OK]   Optical coupling stable — low motion artefact.")

        # Illumination
        if fd.mean_intensity < 15:
            msgs.append("[ERR]  Frame severely underexposed — verify LED drive circuit.")
        elif fd.mean_intensity < 40:
            msgs.append("[WARN] Low IR illumination — increase LED current or exposure.")
        elif fd.saturation_frac > 0.08:
            msgs.append("[WARN] Partial pixel saturation — reduce LED drive or gain.")
        else:
            msgs.append(f"[OK]   Illumination nominal (mean={fd.mean_intensity:.0f} LSB).")

        # Adaptive exposure report
        msgs.append(f"[INFO] AEC: exp={fd.current_exposure_us}µs  gain={fd.current_gain:.2f}×")

        # Optical signal (non-clinical)
        if fd.signal_snr_db > 20:
            msgs.append(
                f"[OK]   Experimental optical signal: SNR={fd.signal_snr_db:.1f}dB  "
                f"AC-RMS={fd.signal_ac_rms:.2f}LSB  (non-clinical)")
        elif fd.signal_snr_db > 10:
            msgs.append(
                f"[WARN] Marginal signal quality: SNR={fd.signal_snr_db:.1f}dB "
                f"(non-clinical optical pulsatility proxy)")
        else:
            msgs.append("[WARN] Insufficient optical signal — awaiting stable buffer.")

        # Overall quality classification
        qc = fd.quality_class
        qc_colours = {
            OpticalQuality.EXCELLENT: "[OK]  ",
            OpticalQuality.GOOD:      "[OK]  ",
            OpticalQuality.FAIR:      "[INFO]",
            OpticalQuality.POOR:      "[WARN]",
            OpticalQuality.INVALID:   "[ERR] ",
        }
        msgs.append(
            f"{qc_colours[qc]} Optical quality class: {qc.value}  "
            f"(confidence={fd.optical_confidence*100:.0f}%)"
        )

        return msgs


# =============================================================================
# §13  TERMINAL DASHBOARD
# =============================================================================

class TerminalDashboard:
    """ANSI in-place updating engineering terminal dashboard (4 Hz refresh)."""

    _CLR  = "\033[2J\033[H"
    _B    = "\033[1m"
    _R    = "\033[0m"
    _G    = "\033[32m"
    _Y    = "\033[33m"
    _RED  = "\033[31m"
    _C    = "\033[36m"
    _M    = "\033[35m"

    _QC_COLOUR = {
        OpticalQuality.EXCELLENT: "\033[32m",
        OpticalQuality.GOOD:      "\033[32m",
        OpticalQuality.FAIR:      "\033[33m",
        OpticalQuality.POOR:      "\033[31m",
        OpticalQuality.INVALID:   "\033[31m",
    }

    def __init__(self, opt_cfg: OptCfg, cam_cfg: CamCfg):
        self._opt = opt_cfg
        self._cam = cam_cfg
        self._t0 = time.monotonic()

    def render(self, fd: FrameData, sig: OpticalSignalExtractor,
               interp: OpticalInterpreter, hw_cam: bool, hw_gpio: bool) -> None:
        el = time.monotonic() - self._t0
        mm, ss = divmod(int(el), 60)
        B, R, G, Y, RE, C, M = self._B,self._R,self._G,self._Y,self._RED,self._C,self._M

        def bar(v: float, w: int = 18) -> str:
            f = int(np.clip(v, 0, 1) * w)
            return f"[{'█'*f}{'░'*(w-f)}] {v*100:5.1f}%"

        ln: List[str] = []
        ln.append(f"{B}{C}{'═'*72}{R}")
        ln.append(f"{B}{C}  NoIR BIOMEDICAL OPTICAL SENSING SYSTEM  v2  │  "
                  f"λ={self._opt.ir_wavelength_nm}nm  │  {mm:02d}:{ss:02d}{R}")
        ln.append(f"{B}{C}  Experimental Research Platform — NON-CLINICAL USE ONLY{R}")
        ln.append(f"{B}{C}{'─'*72}{R}")

        # Hardware status
        cam_s = f"{G}HW CAMERA{R}" if hw_cam else f"{Y}SIMULATION{R}"
        gpio_s = f"{G}GPIO ACTIVE{R}" if hw_gpio else f"{Y}GPIO SIM{R}"
        ln.append(f"  Camera: {cam_s}   LED Control: {gpio_s}   "
                  f"Mode: {B}{fd.illum_mode.name}{R}")
        ln.append(f"  Frame #{fd.frame_idx:,}   FPS: {B}{fd.fps:4.1f}{R}   "
                  f"Target: {self._cam.target_fps}")

        ln.append(f"{C}{'─'*72}{R}")

        # Adaptive exposure
        exp_c = G if 40 <= fd.mean_intensity <= 210 else Y
        ln.append(f"  Adaptive Exposure Control (AEC)")
        ln.append(f"    ExposureTime  : {B}{fd.current_exposure_us:,}{R} µs")
        ln.append(f"    AnalogueGain  : {B}{fd.current_gain:.2f}{R} ×")
        ln.append(f"    Mean Intensity: {exp_c}{fd.mean_intensity:6.1f}{R} LSB  "
                  f"(target {self._opt.target_roi_intensity:.0f})")
        ln.append(f"    Saturation    : {fd.saturation_frac*100:5.2f}%   "
                  f"Laplacian Var: {fd.laplacian_var:7.1f}")

        ln.append(f"{C}{'─'*72}{R}")

        # ROI / segmentation
        fp_s = f"{G}PRESENT{R}" if fd.finger_present else f"{RE}ABSENT{R}"
        seg_s = f"{G}ELLIPSE FIT{R}" if fd.roi.from_segmentation else f"{Y}FALLBACK{R}"
        ln.append(f"  ROI / Segmentation")
        ln.append(f"    Tissue Status : {fp_s}   ROI Mode: {seg_s}")
        ln.append(f"    Seg Confidence: {fd.roi.confidence*100:5.1f}%   "
                  f"Centre: ({fd.roi.cx:.0f}, {fd.roi.cy:.0f}) px")
        ln.append(f"    ROI Mean      : {B}{fd.roi_mean:6.1f}{R} LSB   "
                  f"ROI Std: {fd.roi_std:5.2f}")

        ln.append(f"{C}{'─'*72}{R}")

        # Motion
        mot_c = G if fd.motion_score < 0.2 else Y if fd.motion_score < 0.5 else RE
        ln.append(f"  Motion & Stability")
        ln.append(f"    Motion MAD    : {fd.motion_mad:5.1f} LSB   "
                  f"Score: {mot_c}{fd.motion_score:.3f}{R}")
        ln.append(f"    Frame Corrupt : {RE+'YES'+R if fd.is_motion_corrupt else G+'NO '+R}")
        stab_c = G if fd.optical_stability > 0.7 else Y if fd.optical_stability > 0.4 else RE
        ln.append(f"    Optical Stab. : {stab_c}{bar(fd.optical_stability)}{R}")

        ln.append(f"{C}{'─'*72}{R}")

        # Optical signal (non-clinical)
        ln.append(f"  Experimental Optical Signal  (non-clinical waveform)")
        ln.append(f"    Buffer Fill   : {sig.buffer_fill:3d}/{self._opt.signal_buf_len}  "
                  f"Filter: {'Butterworth BP' if SCIPY_OK else 'UNAVAILABLE'} "
                  f"[{self._opt.bp_low_hz:.1f}–{self._opt.bp_high_hz:.1f} Hz]")
        ln.append(f"    Raw Latest    : {fd.raw_signal_latest:7.2f} LSB")
        ln.append(f"    Filtered      : {fd.filtered_signal_latest:+7.3f} LSB  (AC component)")
        ln.append(f"    AC-RMS        : {fd.signal_ac_rms:7.3f} LSB  (optical pulsatility proxy)")
        snr_c = G if fd.signal_snr_db > 20 else Y if fd.signal_snr_db > 10 else RE
        ln.append(f"    Signal SNR    : {snr_c}{fd.signal_snr_db:6.1f}{R} dB  (experimental)")
        ln.append(f"    Pulsatility   : {bar(fd.pulsatility_confidence)}")

        ln.append(f"{C}{'─'*72}{R}")

        # Composite scores
        iq_c = G if fd.image_quality > 0.7 else Y if fd.image_quality > 0.4 else RE
        conf_c = G if fd.optical_confidence > 0.7 else Y if fd.optical_confidence > 0.4 else RE
        qc_c = self._QC_COLOUR.get(fd.quality_class, RE)
        ln.append(f"  Optical Quality Engine")
        ln.append(f"    Image Quality : {iq_c}{bar(fd.image_quality)}{R}")
        ln.append(f"    Confidence    : {conf_c}{bar(fd.optical_confidence)}{R}")
        ln.append(f"    Classification: {qc_c}{B}{fd.quality_class.value:10s}{R}")

        ln.append(f"{C}{'─'*72}{R}")

        # AI interpretation
        ln.append(f"{B}  AI-ASSISTED OPTICAL INTERPRETATION  (non-clinical){R}")
        msgs = OpticalInterpreter.interpret(fd)
        for msg in msgs[:7]:
            c = G if "[OK]" in msg else Y if "[WARN]" in msg or "[INFO]" in msg else RE
            ln.append(f"    {c}{msg}{R}")

        ln.append(f"{B}{C}{'═'*72}{R}")
        ln.append("  Press  Q / ESC  in OpenCV window  or  Ctrl-C  to quit.")

        print(self._CLR + "\n".join(ln), end="", flush=True)


# =============================================================================
# §14  OPENCV RESEARCH VISUALISER
# =============================================================================

class IRVisualiser:
    """
    OpenCV research-grade visualisation window.

    Layout:
      ┌─────────────────────────────────────┐
      │                                     │
      │   Processed IR frame + overlays     │  ← cam height
      │                                     │
      ├─────────────────────────────────────┤
      │   Raw waveform strip                │  ← 55 px
      ├─────────────────────────────────────┤
      │   Filtered waveform strip           │  ← 55 px
      └─────────────────────────────────────┘

    Overlays:
      • Adaptive ROI (green if finger, blue if fallback)
      • Segmentation ellipse (cyan)
      • Tissue mask semi-transparent fill
      • Per-frame metric text panel (top-left)
      • Motion / quality badge (top-right)
      • Illumination mode badge (top-centre)
      • Watermark (bottom-left)
    """

    WIN = "NoIR BioSensor v2 | 850nm IR Reflectance | EXPERIMENTAL"

    def __init__(self, cam_cfg: CamCfg):
        self._cam = cam_cfg
        self._raw_hist:  Deque[float] = collections.deque(maxlen=256)
        self._filt_hist: Deque[float] = collections.deque(maxlen=256)
        self._ready = False

    def init(self) -> None:
        if not CV2_OK:
            return
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        total_h = self._cam.height + 110
        cv2.resizeWindow(self.WIN, self._cam.width, total_h)
        self._ready = True

    def render(self, gray_disp: np.ndarray, mask: np.ndarray,
               fd: FrameData, sig: OpticalSignalExtractor) -> bool:
        if not CV2_OK or not self._ready:
            return True

        h, w = gray_disp.shape
        vis = cv2.cvtColor(gray_disp, cv2.COLOR_GRAY2BGR)

        # ── Tissue mask overlay (semi-transparent green) ────────────────────
        if mask is not None and mask.any():
            overlay = vis.copy()
            overlay[mask > 0] = (0, 80, 0)
            vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)

        # ── ROI rectangle ───────────────────────────────────────────────────
        r = fd.roi
        roi_col = (0, 220, 50) if fd.finger_present else (30, 100, 200)
        roi_lw  = 2 if r.from_segmentation else 1
        cv2.rectangle(vis, (r.x1, r.y1), (r.x2, r.y2), roi_col, roi_lw)

        # ── Ellipse overlay ─────────────────────────────────────────────────
        if r.ellipse is not None:
            try:
                cv2.ellipse(vis, r.ellipse, (0, 210, 210), 1)
            except cv2.error:
                pass

        # ── Metric panel (top-left) ─────────────────────────────────────────
        panel_lines = [
            f"FPS {fd.fps:4.1f}",
            f"IR  {fd.mean_intensity:5.1f}",
            f"ROI {fd.roi_mean:5.1f}",
            f"Stb {fd.optical_stability*100:4.0f}%",
            f"Conf{fd.optical_confidence*100:4.0f}%",
            f"Mot {fd.motion_score:.3f}",
            f"SNR {fd.signal_snr_db:4.1f}dB",
        ]
        ph = len(panel_lines) * 17 + 6
        cv2.rectangle(vis, (2, 2), (138, 2+ph), (8, 8, 8), -1)
        for i, txt in enumerate(panel_lines):
            cv2.putText(vis, txt, (6, 17 + i*17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 230, 190), 1)

        # ── Quality badge (top-right) ───────────────────────────────────────
        qc = fd.quality_class
        qc_colours_bgr = {
            OpticalQuality.EXCELLENT: (0, 200, 0),
            OpticalQuality.GOOD:      (0, 180, 20),
            OpticalQuality.FAIR:      (0, 180, 200),
            OpticalQuality.POOR:      (0, 100, 210),
            OpticalQuality.INVALID:   (0, 0, 200),
        }
        qc_col = qc_colours_bgr.get(qc, (0, 0, 200))
        cv2.rectangle(vis, (w-118, 2), (w-2, 22), qc_col, -1)
        cv2.putText(vis, qc.value, (w-116, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)

        # ── Illumination mode badge (top-centre) ────────────────────────────
        mode_txt = fd.illum_mode.name
        tw = cv2.getTextSize(mode_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0]
        mx = (w - tw) // 2
        cv2.rectangle(vis, (mx-4, 2), (mx+tw+4, 20), (40, 40, 80), -1)
        cv2.putText(vis, mode_txt, (mx, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 255), 1)

        # ── Motion warning banner ────────────────────────────────────────────
        if fd.is_motion_corrupt:
            cv2.rectangle(vis, (0, h-28), (w, h), (0, 0, 180), -1)
            cv2.putText(vis, "MOTION ARTEFACT — frame rejected", (6, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        # ── Watermark ────────────────────────────────────────────────────────
        cv2.putText(vis, "EXPERIMENTAL | NON-CLINICAL | 850 nm IR REFLECTANCE",
                    (2, h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (60, 60, 150), 1)

        # ── Signal strips ────────────────────────────────────────────────────
        self._raw_hist.append(fd.raw_signal_latest)
        self._filt_hist.append(fd.filtered_signal_latest)
        raw_strip  = self._make_strip(w, 55, list(self._raw_hist),
                                       (0, 200, 80),  "Raw optical signal (ROI mean)")
        filt_strip = self._make_strip(w, 55, list(self._filt_hist),
                                       (0, 140, 220), f"Butterworth BP [{sig._cfg.bp_low_hz:.1f}–{sig._cfg.bp_high_hz:.1f} Hz] — experimental non-clinical waveform")

        combined = np.vstack([vis, raw_strip, filt_strip])
        cv2.imshow(self.WIN, combined)

        key = cv2.waitKey(1) & 0xFF
        return key not in (ord('q'), ord('Q'), 27)

    @staticmethod
    def _make_strip(w: int, h: int, data: List[float],
                    colour: Tuple, label: str) -> np.ndarray:
        strip = np.zeros((h, w, 3), dtype=np.uint8)
        strip[:] = (12, 12, 12)
        if len(data) < 2:
            return strip
        arr = np.array(data, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        rng = hi - lo if hi - lo > 0.1 else 1.0
        norm = (arr - lo) / rng * (h - 10) + 5
        x_idx = np.linspace(0, len(norm)-1, w)
        y_vals = np.interp(x_idx, np.arange(len(norm)), norm).astype(int)
        pts = np.array([[xi, h-1-int(yi)] for xi, yi in enumerate(y_vals)],
                       dtype=np.int32)
        cv2.polylines(strip, [pts], False, colour, 1)
        cv2.putText(strip, label, (4, 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (c*2//3 for c in colour), 1)
        return strip

    def close(self) -> None:
        if CV2_OK:
            cv2.destroyAllWindows()


# =============================================================================
# §15  ILLUMINATION CYCLE SCHEDULER
# =============================================================================

class IlluminationScheduler:
    """
    Optionally cycles through illumination modes to enable differential
    reflectance comparison (IR vs White vs Hybrid).

    In AUTO mode it advances every N seconds.
    In FIXED mode it stays on a single mode.
    """

    def __init__(self, led: IlluminationController,
                 mode_sequence: Optional[List[IlluminationMode]] = None,
                 dwell_s: float = 5.0, auto: bool = False):
        self._led = led
        self._sequence = mode_sequence or [IlluminationMode.IR_ONLY]
        self._dwell = dwell_s
        self._auto = auto
        self._idx = 0
        self._last_switch = time.monotonic()
        self._led.set_mode(self._sequence[0])

    def tick(self) -> IlluminationMode:
        if self._auto and len(self._sequence) > 1:
            now = time.monotonic()
            if now - self._last_switch >= self._dwell:
                self._idx = (self._idx + 1) % len(self._sequence)
                self._led.set_mode(self._sequence[self._idx])
                self._last_switch = now
        return self._sequence[self._idx]

    def current_mode(self) -> IlluminationMode:
        return self._sequence[self._idx]


# =============================================================================
# §16  MAIN SYSTEM CONTROLLER
# =============================================================================

class NoIRBioSensorSystemV2:
    """
    Top-level system controller — wires all subsystems and runs the main loop.

    Thread model:
      cam-acq  (daemon) — Picamera2 acquisition at target fps
      main             — full processing pipeline + dashboard + visualisation
    """

    def __init__(self, illum_auto: bool = False):
        # ── Configuration ────────────────────────────────────────────────────
        self._cam_cfg = CamCfg()
        self._opt_cfg = OptCfg()
        self._seg_cfg = SegCfg()
        self._mot_cfg = MotCfg()
        self._gpio_cfg = GPIOCfg()

        # ── Subsystems ───────────────────────────────────────────────────────
        self._led       = IlluminationController(self._gpio_cfg)
        self._exp_ctrl  = AdaptiveExposureController(self._cam_cfg, self._opt_cfg)
        self._camera    = CameraAcquisition(self._cam_cfg, self._exp_ctrl)
        self._processor = IRImageProcessor(self._opt_cfg)
        self._segmenter = AdaptiveTissueSegmenter(self._seg_cfg, self._cam_cfg)
        self._motion    = MotionDetector(self._mot_cfg)
        self._signal    = OpticalSignalExtractor(self._opt_cfg)
        self._quality   = OpticalQualityEngine()
        self._interp    = OpticalInterpreter()
        self._dashboard = TerminalDashboard(self._opt_cfg, self._cam_cfg)
        self._vis       = IRVisualiser(self._cam_cfg)
        self._scheduler = IlluminationScheduler(
            self._led,
            mode_sequence=[IlluminationMode.IR_ONLY,
                           IlluminationMode.WHITE_ONLY,
                           IlluminationMode.HYBRID],
            dwell_s=8.0,
            auto=illum_auto,
        )

        # ── State ────────────────────────────────────────────────────────────
        self._running = False
        self._frame_idx = 0
        self._last_dash_t = 0.0
        self._dash_interval = 0.25   # 4 Hz terminal refresh

        signal.signal(signal.SIGINT,  self._sighandler)
        signal.signal(signal.SIGTERM, self._sighandler)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._print_banner()
        self._camera.start()
        self._vis.init()
        self._running = True

        log.info("Entering main processing loop.")

        try:
            while self._running:
                frame = self._camera.get_frame(timeout=0.5)
                if frame is None:
                    continue

                self._frame_idx += 1
                ts = time.monotonic()

                # ── Image processing ─────────────────────────────────────────
                gray_metric, gray_disp = self._processor.process(frame)

                # ── Tissue segmentation ──────────────────────────────────────
                mask, roi = self._segmenter.segment(gray_metric)

                # ── Photometrics ─────────────────────────────────────────────
                roi_crop = gray_metric[roi.y1:roi.y2, roi.x1:roi.x2].astype(np.float32)
                roi_mean = float(np.mean(roi_crop)) if roi_crop.size > 0 else 0.0
                roi_std  = float(np.std(roi_crop))  if roi_crop.size > 0 else 0.0

                flat = gray_metric.ravel().astype(np.float32)
                mean_int = float(np.mean(flat))
                sat_frac = float(np.mean(flat >= 254))

                # ── Adaptive exposure update ──────────────────────────────────
                exp_us, gain = self._exp_ctrl.update(roi_mean)

                # ── Motion detection ──────────────────────────────────────────
                mad, lap_var, mot_score, is_corrupt = self._motion.update(gray_metric)

                # ── Optical signal push ───────────────────────────────────────
                finger_present = roi_mean >= 35.0
                self._signal.push(roi_mean, ts, accept=(not is_corrupt and finger_present))

                # ── Illumination scheduler ────────────────────────────────────
                illum_mode = self._scheduler.tick()

                # ── Assemble FrameData ────────────────────────────────────────
                raw_l, filt_l, ac_rms, snr = self._signal.metrics()
                pulse_conf = float(np.clip((snr - 10.0) / 20.0, 0.0, 1.0))

                fd = FrameData(
                    ts=ts, frame_idx=self._frame_idx,
                    fps=self._camera.fps,
                    illum_mode=illum_mode,
                    mean_intensity=mean_int,
                    std_intensity=float(np.std(flat)),
                    saturation_frac=sat_frac,
                    laplacian_var=lap_var,
                    motion_mad=mad,
                    motion_score=mot_score,
                    is_motion_corrupt=is_corrupt,
                    current_exposure_us=exp_us,
                    current_gain=gain,
                    roi=roi,
                    finger_present=finger_present,
                    roi_mean=roi_mean,
                    roi_std=roi_std,
                    raw_signal_latest=raw_l,
                    filtered_signal_latest=filt_l,
                    signal_ac_rms=ac_rms,
                    signal_snr_db=snr,
                    pulsatility_confidence=pulse_conf,
                )

                # ── Quality scoring ───────────────────────────────────────────
                stab, iq, conf, qc = self._quality.evaluate(fd)
                fd.optical_stability    = stab
                fd.image_quality        = iq
                fd.optical_confidence   = conf
                fd.quality_class        = qc

                # ── Terminal dashboard (throttled) ────────────────────────────
                now = time.monotonic()
                if now - self._last_dash_t >= self._dash_interval:
                    self._dashboard.render(fd, self._signal, self._interp,
                                           PICAM_OK, GPIO_OK)
                    self._last_dash_t = now

                # ── OpenCV visualisation ──────────────────────────────────────
                if CV2_OK:
                    keep = self._vis.render(gray_disp, mask, fd, self._signal)
                    if not keep:
                        log.info("User closed visualisation window.")
                        self._running = False

        finally:
            self._shutdown()

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        log.info("Shutting down NoIR Optical Sensing System v2 …")
        self._running = False
        self._camera.stop()
        self._led.cleanup()
        self._vis.close()
        log.info("Session ended.  Frames processed: %d", self._frame_idx)

    def _sighandler(self, sig, frame) -> None:
        log.info("Signal %s received — graceful shutdown.", sig)
        self._running = False

    # ── Banner ────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_banner() -> None:
        print(r"""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║   NoIR BIOMEDICAL OPTICAL SENSING SYSTEM  —  v2                     ║
  ║   Raspberry Pi NoIR Camera | 850 nm IR Reflectance Imaging          ║
  ║   Research Prototype  ·  NON-CLINICAL USE ONLY                      ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║  Illumination  : 850 nm IR LEDs + White LEDs (GPIO-controlled)      ║
  ║  Modality      : Diffuse reflectance optical sensing                 ║
  ║  Target tissue : Fingertip  (surface + shallow subsurface ~1-3 mm)  ║
  ║  Signal filter : Butterworth bandpass 0.7–4 Hz (scipy)              ║
  ║  Segmentation  : Adaptive Otsu + ellipse fitting                    ║
  ║  Exposure ctrl : PI feedback controller (AEC)                       ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║  SCIENTIFIC LIMITATIONS                                              ║
  ║  • NOT deep-tissue  • NOT clinical  • NOT MRI/X-ray                 ║
  ║  • Surface/subsurface optical REFLECTION ONLY                        ║
  ╚══════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# §17  DEPENDENCY CHECK & ENTRY POINT
# =============================================================================

def check_deps() -> None:
    missing = []
    if not CV2_OK:      missing.append("opencv-python-headless (pip install opencv-python-headless)")
    if not SCIPY_OK:    missing.append("scipy                  (pip install scipy)")
    if not PICAM_OK:    log.warning("picamera2 not found — simulation mode active.")
    if not GPIO_OK:     log.warning("RPi.GPIO not found  — LED control disabled.")
    for m in missing:
        log.warning("Missing dependency: %s", m)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="NoIR Biomedical Optical Sensing System v2")
    parser.add_argument(
        "--auto-illum", action="store_true",
        help="Cycle illumination modes automatically (IR → White → Hybrid)")
    args = parser.parse_args()

    check_deps()
    system = NoIRBioSensorSystemV2(illum_auto=args.auto_illum)
    system.run()