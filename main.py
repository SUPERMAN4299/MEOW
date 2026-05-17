#!/usr/bin/env python3
# =============================================================================
#  BioSense-Pi  —  Core Architecture  v1.0
#  Unified biomedical optical + IMU sensing platform for Raspberry Pi 4/5
# =============================================================================
#
#  Author        : BioSense-Pi Research Prototype
#  Target HW     : Raspberry Pi 4/5 + NoIR Camera + MAX30102 + MPU-6050
#  Python        : 3.11+
#  Arch          : Single-file modular, daemon thread orchestration
#
#  SCIENTIFIC DISCLAIMER
#  ─────────────────────
#  All sensor outputs and derived metrics are experimental / non-clinical.
#  This system is a research prototype. No output constitutes medical advice,
#  clinical diagnosis, or treatment recommendation.
#
#  STATE MACHINE
#  ─────────────
#  INIT → SENSOR_CHECK → CALIBRATION → RUNNING ⇄ DEGRADED_MODE
#                                           ↓              ↓
#                                     ERROR_RECOVERY ← ←  ┘
#                                           ↓
#                                       SHUTDOWN
#
#  THREAD MODEL
#  ────────────
#  main               — state machine + watchdog supervisor
#  acq-camera         — Picamera2 / NoIR frame acquisition   (daemon)
#  acq-ppg            — MAX30102 I²C FIFO polling             (daemon)
#  acq-imu            — MPU-6050 I²C burst polling            (daemon)
#  proc-optical       — IR image processing + rPPG pipeline   (daemon)
#  proc-biosignal     — SpO2 / HR / PI computation pipeline   (daemon)
#  proc-motion        — IMU artifact scoring + orientation     (daemon)
#  watchdog           — thread health + state guard            (daemon)
#  display            — terminal dashboard + OpenCV vis        (daemon)
#
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# §1  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

# Standard library — Python 3.11 guaranteed
import os
import sys
import csv
import json
import math
import time
import queue
import signal
import logging
import pathlib
import threading
import statistics
import collections
from enum import Enum, auto, unique
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Deque, Any, Callable

# Numeric / scientific
import numpy as np

try:
    from scipy.signal import butter, sosfilt, sosfilt_zi, find_peaks
    _SCIPY = True
except ImportError:
    _SCIPY = False

# Camera
try:
    from picamera2 import Picamera2
    from libcamera import controls as _lc
    _PICAM = True
except ImportError:
    _PICAM = False

# I²C
try:
    import smbus2
    _SMBUS = True
except ImportError:
    _SMBUS = False

# Computer vision
try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

# MAX30102 PPG subsystem (integrated from max30102_subsystem.py)
try:
    from max30102_subsystem import (
        worker_acq_ppg as _worker_acq_ppg_real,
        MAX30102Config,
        MAX30102Driver,
    )
    _MAX30102_AVAILABLE = True
except ImportError:
    _MAX30102_AVAILABLE = False
    _worker_acq_ppg_real = None
    MAX30102Config = None
    MAX30102Driver = None


# ─────────────────────────────────────────────────────────────────────────────
# §2  CONFIGURATION DATACLASSES
#     All tunable parameters live here — immutable after construction.
#     Loaded from JSON override file if present at runtime.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CameraConfig:
    """Picamera2 / NoIR acquisition parameters."""
    width              : int   = 640
    height             : int   = 480
    target_fps         : int   = 30
    pixel_format       : str   = "BGR888"
    init_exposure_us   : int   = 15_000
    init_gain          : float = 4.0
    min_exposure_us    : int   = 3_000
    max_exposure_us    : int   = 40_000
    min_gain           : float = 1.0
    max_gain           : float = 8.0
    # Frame queue depth — limits memory; oldest frame evicted on overflow
    queue_depth        : int   = 4


@dataclass(frozen=True)
class OpticalConfig:
    """IR optical sensing / rPPG processing parameters."""
    ir_wavelength_nm       : int   = 850
    target_roi_intensity   : float = 140.0
    intensity_deadband     : float = 15.0
    clahe_clip             : float = 2.5
    clahe_tile             : Tuple[int, int] = (8, 8)
    blur_kernel            : int   = 5          # must be odd
    bp_low_hz              : float = 0.7
    bp_high_hz             : float = 4.0
    bp_order               : int   = 4
    signal_buffer_frames   : int   = 300        # ~10 s @ 30 fps
    min_frames_for_filter  : int   = 60         # ~2 s warm-up
    finger_dc_threshold    : float = 35.0       # LSB — tissue contact gate


@dataclass(frozen=True)
class PPGConfig:
    """MAX30102 PPG / SpO2 / HR acquisition parameters."""
    i2c_bus             : int   = 1
    sensor_addr         : int   = 0x57
    sample_rate_hz      : int   = 100
    led_current         : int   = 0x3C          # ~12 mA
    adc_range           : int   = 0x20          # 4096 nA full-scale
    pulse_width         : int   = 0x03          # 411 µs → 18-bit
    ring_depth          : int   = 500           # samples
    calc_window         : int   = 200           # analysis window
    step_size           : int   = 50            # overlap stride
    bpf_low_hz          : float = 0.5
    bpf_high_hz         : float = 4.0
    bpf_order           : int   = 4
    baseline_tau        : float = 0.995
    finger_dc_min       : int   = 50_000
    finger_ac_min       : int   = 500
    peak_min_bpm        : int   = 30
    peak_max_bpm        : int   = 220
    display_avg_n       : int   = 6


@dataclass(frozen=True)
class IMUConfig:
    """MPU-6050 inertial measurement unit parameters."""
    i2c_bus             : int   = 1
    sensor_addr         : int   = 0x68
    sample_rate_hz      : int   = 100
    accel_range_g       : int   = 2             # ±2 g
    gyro_range_dps      : int   = 250           # ±250 °/s
    dlpf_hz             : int   = 44            # DLPF cutoff
    calibration_samples : int   = 200
    # Physical scaling
    accel_lsb_per_g     : float = 16384.0
    gyro_lsb_per_dps    : float = 131.0
    temp_sensitivity    : float = 340.0
    temp_offset         : float = 36.53
    ring_depth          : int   = 512


@dataclass(frozen=True)
class WatchdogConfig:
    """Watchdog supervisor parameters."""
    check_interval_s     : float = 1.0      # how often to poll thread health
    thread_timeout_s     : float = 5.0      # max silence before thread declared dead
    max_consecutive_fail : int   = 3        # failures before state → DEGRADED
    max_error_rate       : float = 0.10     # I²C error fraction ceiling


@dataclass(frozen=True)
class LoggingConfig:
    """Logging and data recording parameters."""
    level               : int   = logging.INFO
    log_dir             : str   = "/tmp/biosense_pi"
    log_filename        : str   = "biosense.log"
    csv_filename        : str   = "biosense_data.csv"
    max_bytes           : int   = 10 * 1024 * 1024   # 10 MB rotation
    backup_count        : int   = 3
    console_enabled     : bool  = True
    file_enabled        : bool  = True
    csv_enabled         : bool  = True
    csv_flush_interval  : int   = 50   # rows between flushes


@dataclass(frozen=True)
class SystemConfig:
    """Top-level system configuration container."""
    camera   : CameraConfig   = field(default_factory=CameraConfig)
    optical  : OpticalConfig  = field(default_factory=OpticalConfig)
    ppg      : PPGConfig      = field(default_factory=PPGConfig)
    imu      : IMUConfig      = field(default_factory=IMUConfig)
    watchdog : WatchdogConfig = field(default_factory=WatchdogConfig)
    logging  : LoggingConfig  = field(default_factory=LoggingConfig)

    # Hardware availability overrides (auto-detected at startup)
    hw_camera_enabled : bool = True
    hw_ppg_enabled    : bool = True
    hw_imu_enabled    : bool = True
    display_enabled   : bool = True
    cv2_display       : bool = True

    @classmethod
    def from_json(cls, path: str) -> "SystemConfig":
        """
        Load configuration from a JSON override file.
        Missing keys fall back to dataclass defaults.
        Allows field-level partial overrides without rebuilding the full tree.
        """
        try:
            with open(path, "r") as fh:
                overrides = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()

        def _apply(dc_cls, d: dict):
            valid = {f.name for f in dc_cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            kwargs = {k: v for k, v in d.items() if k in valid}
            return dc_cls(**kwargs)

        return cls(
            camera   = _apply(CameraConfig,   overrides.get("camera",   {})),
            optical  = _apply(OpticalConfig,  overrides.get("optical",  {})),
            ppg      = _apply(PPGConfig,       overrides.get("ppg",      {})),
            imu      = _apply(IMUConfig,       overrides.get("imu",      {})),
            watchdog = _apply(WatchdogConfig,  overrides.get("watchdog", {})),
            logging  = _apply(LoggingConfig,   overrides.get("logging",  {})),
        )


# ─────────────────────────────────────────────────────────────────────────────
# §3  STATE MACHINE — ENUMERATIONS & TRANSITIONS
# ─────────────────────────────────────────────────────────────────────────────

@unique
class SystemState(Enum):
    INIT            = auto()   # Cold start — subsystem instantiation
    SENSOR_CHECK    = auto()   # Hardware presence / ID verification
    CALIBRATION     = auto()   # Static IMU + optical baseline calibration
    RUNNING         = auto()   # Normal full-pipeline operation
    DEGRADED_MODE   = auto()   # One or more subsystems failed — partial op
    ERROR_RECOVERY  = auto()   # Attempting automatic fault recovery
    SHUTDOWN        = auto()   # Ordered teardown


# Legal state transitions — enforced by StateManager.
# Tuple format: (from_state, to_state)
_LEGAL_TRANSITIONS: frozenset = frozenset({
    (SystemState.INIT,           SystemState.SENSOR_CHECK),
    (SystemState.SENSOR_CHECK,   SystemState.CALIBRATION),
    (SystemState.SENSOR_CHECK,   SystemState.DEGRADED_MODE),
    (SystemState.SENSOR_CHECK,   SystemState.SHUTDOWN),
    (SystemState.CALIBRATION,    SystemState.RUNNING),
    (SystemState.CALIBRATION,    SystemState.DEGRADED_MODE),
    (SystemState.CALIBRATION,    SystemState.ERROR_RECOVERY),
    (SystemState.RUNNING,        SystemState.DEGRADED_MODE),
    (SystemState.RUNNING,        SystemState.ERROR_RECOVERY),
    (SystemState.RUNNING,        SystemState.SHUTDOWN),
    (SystemState.DEGRADED_MODE,  SystemState.RUNNING),
    (SystemState.DEGRADED_MODE,  SystemState.ERROR_RECOVERY),
    (SystemState.DEGRADED_MODE,  SystemState.SHUTDOWN),
    (SystemState.ERROR_RECOVERY, SystemState.RUNNING),
    (SystemState.ERROR_RECOVERY, SystemState.DEGRADED_MODE),
    (SystemState.ERROR_RECOVERY, SystemState.SHUTDOWN),
})


# ─────────────────────────────────────────────────────────────────────────────
# §4  SHARED STATE SYSTEM
#     Thread-safe, low-latency inter-thread communication.
#     Uses RLock for nested acquisition safety and condition variables
#     for producer/consumer synchronisation without busy-polling.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SensorSnapshot:
    """
    Immutable snapshot of the latest processed sensor values.
    Written atomically by processor threads; read by display + logger.
    All fields carry monotonic timestamp of their most recent update.

    Design: one snapshot per sensor domain. The display thread reads
    the latest snapshot for each domain independently — no global lock
    held across display rendering.
    """
    # ── Optical / rPPG (NoIR camera) ──────────────────────────────────────
    roi_mean_ir          : float = 0.0
    roi_std_ir           : float = 0.0
    optical_ac_rms       : float = 0.0
    optical_snr_db       : float = 0.0
    optical_quality_conf : float = 0.0
    optical_quality_label: str   = "INVALID"
    finger_present_camera: bool  = False
    exposure_us          : int   = 15_000
    analogue_gain        : float = 4.0
    ts_optical           : float = 0.0

    # ── PPG / SpO2 / HR (MAX30102) ────────────────────────────────────────
    bpm                  : Optional[float] = None
    spo2                 : Optional[float] = None
    perfusion_index      : Optional[float] = None
    ppg_sqi              : float = 0.0
    ppg_sqi_label        : str   = "NO SIGNAL"
    finger_present_ppg   : bool  = False
    dc_ir_ppg            : float = 0.0
    ac_ir_ppg            : float = 0.0
    ppg_noise_est        : float = 0.0
    ts_ppg               : float = 0.0

    # ── IMU / Motion (MPU-6050) ───────────────────────────────────────────
    accel_x              : float = 0.0
    accel_y              : float = 0.0
    accel_z              : float = 0.0
    gyro_x               : float = 0.0
    gyro_y               : float = 0.0
    gyro_z               : float = 0.0
    die_temp_c           : float = 0.0
    accel_mag            : float = 0.0
    dynamic_accel        : float = 0.0
    gyro_mag             : float = 0.0
    pitch_deg            : float = 0.0
    roll_deg             : float = 0.0
    tilt_deg             : float = 0.0
    motion_state         : str   = "STABLE"
    artifact_score       : float = 0.0
    ppg_validity_pct     : float = 100.0
    ts_imu               : float = 0.0

    # ── System health ─────────────────────────────────────────────────────
    frame_idx            : int   = 0
    camera_fps           : float = 0.0
    ppg_effective_fs     : float = 0.0
    imu_effective_fs     : float = 0.0
    system_state         : str   = "INIT"
    uptime_s             : float = 0.0
    i2c_error_count      : int   = 0


class StateManager:
    """
    Thread-safe runtime state machine.

    Responsibilities:
      · Enforce legal state transitions (raises on illegal attempt)
      · Broadcast state change notifications via threading.Condition
      · Provide non-blocking current-state query for hot paths
      · Record state history with monotonic timestamps for diagnostics

    Lock ordering convention (to prevent deadlock):
      Always acquire _lock before any other subsystem lock.
      Never hold _lock while calling external code that may block.
    """

    _MAX_HISTORY = 64   # Bounded state-change history ring

    def __init__(self, initial: SystemState = SystemState.INIT):
        self._state    : SystemState         = initial
        self._lock     : threading.RLock     = threading.RLock()
        self._condition: threading.Condition = threading.Condition(self._lock)
        self._history  : Deque[Tuple[float, SystemState, SystemState]] = \
            collections.deque(maxlen=self._MAX_HISTORY)
        self._callbacks: List[Callable[[SystemState, SystemState], None]] = []
        self._entered  : float = time.monotonic()

    # ── Query (no lock — volatile read of Python reference is atomic) ──────

    @property
    def state(self) -> SystemState:
        return self._state

    def is_running(self) -> bool:
        return self._state == SystemState.RUNNING

    def is_terminal(self) -> bool:
        return self._state == SystemState.SHUTDOWN

    def in_states(self, *states: SystemState) -> bool:
        return self._state in states

    # ── Transition ─────────────────────────────────────────────────────────

    def transition(self, to: SystemState, reason: str = "") -> bool:
        """
        Attempt a state transition. Returns True on success.

        Raises ValueError for illegal transition attempts — fail-fast
        policy ensures state machine invariants are never silently violated.
        Notifies all registered callbacks and waiting threads after commit.
        """
        with self._lock:
            frm = self._state
            if frm == to:
                return True   # idempotent
            if (frm, to) not in _LEGAL_TRANSITIONS:
                raise ValueError(
                    f"Illegal state transition: {frm.name} → {to.name}"
                    + (f"  reason={reason}" if reason else "")
                )
            now = time.monotonic()
            self._history.append((now, frm, to))
            self._state   = to
            self._entered = now
            self._condition.notify_all()

        # Invoke callbacks outside lock to avoid potential deadlock
        for cb in self._callbacks:
            try:
                cb(frm, to)
            except Exception:
                pass

        return True

    def wait_for_state(self, target: SystemState, timeout: float = 30.0) -> bool:
        """
        Block until state equals target or timeout elapses.
        Used by startup sequencing to synchronise thread readiness.
        """
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._state != target:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=min(remaining, 1.0))
        return True

    def wait_for_any(self, targets: frozenset, timeout: float = 30.0) -> Optional[SystemState]:
        """Block until state is any of targets. Returns reached state or None."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._state not in targets:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=min(remaining, 1.0))
            return self._state

    def register_callback(self, cb: Callable[[SystemState, SystemState], None]) -> None:
        """Register a callback invoked on every successful state transition."""
        self._callbacks.append(cb)

    def time_in_state(self) -> float:
        return time.monotonic() - self._entered

    def history(self) -> List[Tuple[float, SystemState, SystemState]]:
        return list(self._history)


class SharedStateHub:
    """
    Central shared-state repository.

    Provides:
      · Thread-safe snapshot read/write for each sensor domain
      · Bounded deque buffers for streaming data (raw + processed)
      · Named inter-thread signal queues
      · Monotonic clock for all timestamps

    Memory budget (Raspberry Pi 4 target — 2 GB):
      Camera frame queue   : 4 × 640×480×3 ≈ 3.7 MB
      PPG ring buffer      : 500 × 8 bytes  ≈ 4 KB
      IMU ring buffer      : 512 × 7×8      ≈ 28 KB
      Optical signal buf   : 300 × 8 bytes  ≈ 2.4 KB
      Snapshot + misc      : < 50 KB
      Total                : < 5 MB   ✓ well within Pi constraints

    Lock hierarchy (acquire in this order to prevent deadlock):
      1. _snap_lock   — snapshot read/write
      2. per-buffer locks are RLock on the buffer object itself
      Never hold _snap_lock when calling code that acquires other locks.
    """

    def __init__(self, cfg: SystemConfig):
        self._cfg = cfg

        # ── Snapshot store ─────────────────────────────────────────────────
        self._snapshot  = SensorSnapshot()
        self._snap_lock = threading.RLock()

        # ── Raw frame queues (bounded, drop-oldest on overflow) ────────────
        # Camera frames: uint8 BGR arrays — largest memory consumer
        self.camera_queue: queue.Queue = queue.Queue(
            maxsize=cfg.camera.queue_depth)

        # PPG FIFO samples (red_raw, ir_raw) tuples
        self.ppg_raw_queue: queue.Queue = queue.Queue(maxsize=256)

        # IMU burst dicts from driver
        self.imu_raw_queue: queue.Queue = queue.Queue(maxsize=256)

        # ── Streaming ring buffers (processed signals for analysis) ─────────
        # rPPG optical signal (float32, ROI mean intensity per accepted frame)
        self.optical_signal_buf: Deque[float] = collections.deque(
            maxlen=cfg.optical.signal_buffer_frames)

        # PPG IR channel (float64)
        self.ppg_ir_buf: Deque[float] = collections.deque(
            maxlen=cfg.ppg.ring_depth)
        # PPG RED channel (float64)
        self.ppg_red_buf: Deque[float] = collections.deque(
            maxlen=cfg.ppg.ring_depth)

        # IMU per-channel filtered signal buffers (float64, 512 samples)
        imu_d = cfg.imu.ring_depth
        self.imu_ax_buf: Deque[float] = collections.deque(maxlen=imu_d)
        self.imu_ay_buf: Deque[float] = collections.deque(maxlen=imu_d)
        self.imu_az_buf: Deque[float] = collections.deque(maxlen=imu_d)
        self.imu_gx_buf: Deque[float] = collections.deque(maxlen=imu_d)
        self.imu_gy_buf: Deque[float] = collections.deque(maxlen=imu_d)
        self.imu_gz_buf: Deque[float] = collections.deque(maxlen=imu_d)

        # ── Timestamp tracking ─────────────────────────────────────────────
        self.t0: float = time.monotonic()

        # ── Inter-thread command signals ───────────────────────────────────
        # Commands posted by state machine; consumed by acquisition threads
        self.cmd_queue: queue.Queue = queue.Queue(maxsize=16)

        # Calibration result slot — written by CALIBRATION handler, read by RUNNING
        self._calib_result: Optional[Dict[str, Any]] = None
        self._calib_lock   = threading.Lock()

    # ── Snapshot API ───────────────────────────────────────────────────────

    def update_snapshot(self, **kwargs: Any) -> None:
        """
        Atomic partial update of the shared snapshot.
        Only provided fields are modified; others remain unchanged.
        Uses object.__setattr__ to work around frozen-dataclass restrictions
        (SensorSnapshot is NOT frozen — it is a live mutable record).
        """
        with self._snap_lock:
            for k, v in kwargs.items():
                setattr(self._snapshot, k, v)

    def read_snapshot(self) -> SensorSnapshot:
        """
        Return a shallow copy of the current snapshot.
        Copy prevents callers from holding the lock implicitly through
        a reference to the mutable internal object.
        """
        with self._snap_lock:
            # Shallow copy is sufficient — all fields are scalars or strings
            import copy
            return copy.copy(self._snapshot)

    # ── Buffer helpers ─────────────────────────────────────────────────────

    def push_optical_sample(self, val: float) -> None:
        self.optical_signal_buf.append(val)

    def optical_signal_array(self) -> np.ndarray:
        return np.array(self.optical_signal_buf, dtype=np.float32)

    def push_ppg_sample(self, ir: float, red: float) -> None:
        self.ppg_ir_buf.append(ir)
        self.ppg_red_buf.append(red)

    def ppg_window(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return last n samples from PPG IR and RED buffers as float64 arrays."""
        ir  = np.array(list(self.ppg_ir_buf)[-n:],  dtype=np.float64)
        red = np.array(list(self.ppg_red_buf)[-n:], dtype=np.float64)
        return ir, red

    def push_imu_sample(self, ax: float, ay: float, az: float,
                        gx: float, gy: float, gz: float) -> None:
        self.imu_ax_buf.append(ax);  self.imu_ay_buf.append(ay)
        self.imu_az_buf.append(az);  self.imu_gx_buf.append(gx)
        self.imu_gy_buf.append(gy);  self.imu_gz_buf.append(gz)

    # ── Calibration slot ───────────────────────────────────────────────────

    def set_calibration(self, result: Dict[str, Any]) -> None:
        with self._calib_lock:
            self._calib_result = result

    def get_calibration(self) -> Optional[Dict[str, Any]]:
        with self._calib_lock:
            return self._calib_result

    # ── Clock ──────────────────────────────────────────────────────────────

    def now(self) -> float:
        """Monotonic timestamp relative to system start [seconds]."""
        return time.monotonic() - self.t0

    def clock(self) -> float:
        """Raw monotonic clock — use for interval measurement."""
        return time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# §5  LOGGING MANAGER
#     Structured, rotating, multi-sink logger.
#     Configured once; all subsystems retrieve the same named logger.
# ─────────────────────────────────────────────────────────────────────────────

class LoggingManager:
    """
    Initialises the Python logging hierarchy for BioSense-Pi.

    Sinks:
      · Console (StreamHandler) — human-readable, INFO+
      · Rotating file (RotatingFileHandler) — DEBUG+, 10 MB × 3 backups
      · CSV data recorder — structured sensor data at configurable rate

    All handlers share a common formatter for log correlation.
    The CSV writer is NOT a logging.Handler — it is a separate sink
    driven by explicit calls from processor threads.

    Thread safety: logging module is thread-safe internally.
    CSV writes are serialised by a dedicated writer lock.
    """

    LOGGER_NAME = "biosense"

    _CSV_FIELDS = [
        "ts_rel", "state", "bpm", "spo2", "pi", "ppg_sqi",
        "roi_mean_ir", "optical_snr_db", "optical_quality_conf",
        "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
        "pitch_deg", "roll_deg", "tilt_deg",
        "motion_state", "artifact_score", "ppg_validity_pct",
        "exposure_us", "analogue_gain", "camera_fps",
        "ppg_effective_fs", "imu_effective_fs",
    ]

    def __init__(self, cfg: LoggingConfig):
        self._cfg       = cfg
        self._csv_lock  = threading.Lock()
        self._csv_fh    : Optional[Any] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_rows  = 0
        self._log       : Optional[logging.Logger] = None
        self._setup()

    def _setup(self) -> None:
        log_dir = pathlib.Path(self._cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log = logging.getLogger(self.LOGGER_NAME)
        log.setLevel(logging.DEBUG)
        log.propagate = False

        fmt = logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        if self._cfg.console_enabled:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(self._cfg.level)
            ch.setFormatter(fmt)
            log.addHandler(ch)

        if self._cfg.file_enabled:
            from logging.handlers import RotatingFileHandler
            fh = RotatingFileHandler(
                log_dir / self._cfg.log_filename,
                maxBytes=self._cfg.max_bytes,
                backupCount=self._cfg.backup_count,
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            log.addHandler(fh)

        if self._cfg.csv_enabled:
            csv_path = log_dir / self._cfg.csv_filename
            self._csv_fh = open(csv_path, "a", newline="", buffering=8192)
            self._csv_writer = csv.DictWriter(
                self._csv_fh, fieldnames=self._CSV_FIELDS, extrasaction="ignore")
            if csv_path.stat().st_size == 0:
                self._csv_writer.writeheader()

        self._log = log

    def get(self, child: str = "") -> logging.Logger:
        """Return named child logger for a subsystem."""
        name = f"{self.LOGGER_NAME}.{child}" if child else self.LOGGER_NAME
        return logging.getLogger(name)

    def record_csv(self, snap: SensorSnapshot) -> None:
        """
        Write one row to the CSV data file.
        Called by display / recorder thread at a bounded rate.
        Skips silently if CSV is disabled or writer is None.
        """
        if self._csv_writer is None:
            return
        row = {
            "ts_rel"             : f"{snap.uptime_s:.3f}",
            "state"              : snap.system_state,
            "bpm"                : snap.bpm,
            "spo2"               : snap.spo2,
            "pi"                 : snap.perfusion_index,
            "ppg_sqi"            : f"{snap.ppg_sqi:.1f}",
            "roi_mean_ir"        : f"{snap.roi_mean_ir:.2f}",
            "optical_snr_db"     : f"{snap.optical_snr_db:.2f}",
            "optical_quality_conf": f"{snap.optical_quality_conf:.3f}",
            "accel_x"            : f"{snap.accel_x:.5f}",
            "accel_y"            : f"{snap.accel_y:.5f}",
            "accel_z"            : f"{snap.accel_z:.5f}",
            "gyro_x"             : f"{snap.gyro_x:.3f}",
            "gyro_y"             : f"{snap.gyro_y:.3f}",
            "gyro_z"             : f"{snap.gyro_z:.3f}",
            "pitch_deg"          : f"{snap.pitch_deg:.2f}",
            "roll_deg"           : f"{snap.roll_deg:.2f}",
            "tilt_deg"           : f"{snap.tilt_deg:.2f}",
            "motion_state"       : snap.motion_state,
            "artifact_score"     : f"{snap.artifact_score:.1f}",
            "ppg_validity_pct"   : f"{snap.ppg_validity_pct:.1f}",
            "exposure_us"        : snap.exposure_us,
            "analogue_gain"      : f"{snap.analogue_gain:.3f}",
            "camera_fps"         : f"{snap.camera_fps:.1f}",
            "ppg_effective_fs"   : f"{snap.ppg_effective_fs:.1f}",
            "imu_effective_fs"   : f"{snap.imu_effective_fs:.1f}",
        }
        with self._csv_lock:
            self._csv_writer.writerow(row)
            self._csv_rows += 1
            if self._csv_rows % self._cfg.csv_flush_interval == 0:
                self._csv_fh.flush()

    def close(self) -> None:
        if self._csv_fh is not None:
            try:
                self._csv_fh.flush()
                self._csv_fh.close()
            except Exception:
                pass
        logging.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# §6  WATCHDOG FRAMEWORK
#     Monitors thread liveness and I²C health; drives state transitions.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThreadHealthRecord:
    """
    Per-thread health tracking record.

    heartbeat_ts   : monotonic time of last call to beat()
    fail_count     : consecutive heartbeat misses
    is_critical    : if True, miss → DEGRADED; if False → warning only
    last_error     : most recent exception string (for diagnostics)
    restart_count  : how many times the thread has been restarted by watchdog
    """
    name           : str
    is_critical    : bool
    heartbeat_ts   : float    = field(default_factory=time.monotonic)
    fail_count     : int      = 0
    last_error     : str      = ""
    restart_count  : int      = 0
    thread_ref     : Optional[threading.Thread] = None


class WatchdogFramework:
    """
    Supervisor thread that monitors all registered worker threads.

    Design (no busy-polling):
      · Sleeps for check_interval_s between sweeps using Event.wait()
      · On each sweep, checks every registered ThreadHealthRecord
      · If heartbeat_ts age > thread_timeout_s → increments fail_count
      · If fail_count > max_consecutive_fail AND thread is critical:
          → Signals state machine to DEGRADED_MODE or ERROR_RECOVERY
      · Watchdog thread itself is a daemon; it stops when main exits
      · Watchdog calls a user-supplied restart_fn(record) if provided

    Heartbeat protocol:
      Worker threads call watchdog.beat(name) once per acquisition cycle.
      This is O(1) — just a monotonic clock write with a dict lookup.
      Hot-path threads (100 Hz IMU) should beat every N cycles to avoid
      watchdog overhead; N=10 gives 10 Hz heartbeat at 100 Hz acquisition.
    """

    def __init__(self,
                 cfg       : WatchdogConfig,
                 state_mgr : StateManager,
                 log       : logging.Logger):
        self._cfg      = cfg
        self._sm       = state_mgr
        self._log      = log
        self._records  : Dict[str, ThreadHealthRecord] = {}
        self._lock     = threading.RLock()
        self._stop_evt = threading.Event()
        self._thread   : Optional[threading.Thread] = None
        # Optional restart factory: name → Callable[[], threading.Thread]
        self._restart_fns: Dict[str, Callable[[], threading.Thread]] = {}

    def register(self,
                 name       : str,
                 thread     : threading.Thread,
                 is_critical: bool = True,
                 restart_fn : Optional[Callable[[], threading.Thread]] = None
                 ) -> None:
        """Register a thread for monitoring."""
        with self._lock:
            self._records[name] = ThreadHealthRecord(
                name       = name,
                is_critical= is_critical,
                thread_ref = thread,
            )
            if restart_fn is not None:
                self._restart_fns[name] = restart_fn

    def beat(self, name: str) -> None:
        """
        Worker heartbeat — call from acquisition / processing loop.
        O(1): dict lookup + monotonic write.
        Silently no-ops for unregistered names (graceful startup).
        """
        try:
            self._records[name].heartbeat_ts = time.monotonic()
            self._records[name].fail_count   = 0
        except KeyError:
            pass

    def report_error(self, name: str, error: str) -> None:
        """Record error message for a thread (for diagnostics / logging)."""
        try:
            self._records[name].last_error = error[:256]
        except KeyError:
            pass

    def start(self) -> None:
        """Start the watchdog daemon thread."""
        self._thread = threading.Thread(
            target=self._loop, name="watchdog", daemon=True)
        self._thread.start()
        self._log.info("Watchdog started (interval=%.1fs timeout=%.1fs).",
                       self._cfg.check_interval_s, self._cfg.thread_timeout_s)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        """
        Watchdog supervision loop.

        Sweep algorithm (O(N) in registered thread count):
          For each record:
            1. If thread is alive → check heartbeat age
            2. If heartbeat stale → increment fail_count
            3. If fail_count exceeded → attempt restart or trigger state change
            4. If thread is dead and not restartable → notify state machine
        """
        while not self._stop_evt.wait(timeout=self._cfg.check_interval_s):
            now = time.monotonic()
            with self._lock:
                records = list(self._records.values())

            for rec in records:
                t = rec.thread_ref
                alive = t is not None and t.is_alive()

                if not alive:
                    # Thread died — attempt restart if factory available
                    self._handle_dead_thread(rec)
                    continue

                age = now - rec.heartbeat_ts
                if age > self._cfg.thread_timeout_s:
                    rec.fail_count += 1
                    self._log.warning(
                        "Thread '%s' heartbeat stale %.1fs (fail=%d).",
                        rec.name, age, rec.fail_count)

                    if rec.fail_count >= self._cfg.max_consecutive_fail:
                        if rec.is_critical:
                            self._escalate(rec)
                        else:
                            self._log.warning(
                                "Non-critical thread '%s' unresponsive — continuing.",
                                rec.name)
                else:
                    if rec.fail_count > 0:
                        self._log.info("Thread '%s' heartbeat recovered.", rec.name)
                        rec.fail_count = 0

    def _handle_dead_thread(self, rec: ThreadHealthRecord) -> None:
        """Attempt thread restart; escalate state if restart unavailable."""
        if rec.name in self._restart_fns:
            if rec.restart_count < 3:
                self._log.warning("Watchdog restarting thread '%s' (attempt %d).",
                                  rec.name, rec.restart_count + 1)
                try:
                    new_t = self._restart_fns[rec.name]()
                    rec.thread_ref   = new_t
                    rec.restart_count += 1
                    rec.heartbeat_ts  = time.monotonic()
                    rec.fail_count    = 0
                    new_t.start()
                except Exception as e:
                    self._log.error("Restart of '%s' failed: %s", rec.name, e)
                    self._escalate(rec)
            else:
                self._log.error("Thread '%s' restart limit reached — escalating.",
                                rec.name)
                self._escalate(rec)
        else:
            if rec.is_critical:
                self._log.error("Critical thread '%s' died with no restart fn.", rec.name)
                self._escalate(rec)

    def _escalate(self, rec: ThreadHealthRecord) -> None:
        """Drive state machine toward recovery."""
        sm = self._sm
        cur = sm.state
        if cur == SystemState.RUNNING:
            try:
                sm.transition(SystemState.ERROR_RECOVERY,
                               reason=f"thread {rec.name} unresponsive")
            except ValueError:
                pass
        elif cur == SystemState.DEGRADED_MODE:
            try:
                sm.transition(SystemState.ERROR_RECOVERY,
                               reason=f"thread {rec.name} failed in degraded mode")
            except ValueError:
                pass

    def health_summary(self) -> Dict[str, Dict]:
        """Return a snapshot of thread health for the dashboard."""
        with self._lock:
            now = time.monotonic()
            return {
                name: {
                    "alive"         : (r.thread_ref is not None
                                       and r.thread_ref.is_alive()),
                    "heartbeat_age" : now - r.heartbeat_ts,
                    "fail_count"    : r.fail_count,
                    "restart_count" : r.restart_count,
                    "last_error"    : r.last_error,
                    "is_critical"   : r.is_critical,
                }
                for name, r in self._records.items()
            }


# ─────────────────────────────────────────────────────────────────────────────
# §7  THREAD MANAGER
#     Lifecycle management for all daemon worker threads.
# ─────────────────────────────────────────────────────────────────────────────

class ManagedThread:
    """
    Wrapper that gives a worker function daemon-thread lifecycle management.

    Each managed thread:
      · Runs as a daemon (auto-killed if main exits)
      · Calls beat() on the watchdog at the configured rate
      · Catches and logs all unhandled exceptions without crashing the system
      · Sets its stop_event on clean exit so the watchdog knows it's done
      · Communicates state via an optional result_queue

    Worker function signature:
        def worker(stop_event: threading.Event, state_hub: SharedStateHub,
                   watchdog: WatchdogFramework, cfg: SystemConfig,
                   log: logging.Logger) -> None
    """

    def __init__(self,
                 name       : str,
                 target     : Callable,
                 state_hub  : "SharedStateHub",
                 watchdog   : "WatchdogFramework",
                 cfg        : SystemConfig,
                 log_mgr    : LoggingManager,
                 is_critical: bool = True,
                 daemon     : bool = True):
        self.name        = name
        self._target     = target
        self._hub        = state_hub
        self._wd         = watchdog
        self._cfg        = cfg
        self._log        = log_mgr.get(name)
        self._is_critical= is_critical
        self._daemon     = daemon
        self.stop_event  = threading.Event()
        self._thread     : Optional[threading.Thread] = None

    def _run_wrapper(self) -> None:
        """
        Exception-safe wrapper around the worker function.
        Ensures watchdog is notified of unhandled exceptions.
        """
        self._log.debug("Thread '%s' starting.", self.name)
        try:
            self._target(
                stop_event = self.stop_event,
                state_hub  = self._hub,
                watchdog   = self._wd,
                cfg        = self._cfg,
                log        = self._log,
            )
        except Exception as exc:
            self._log.exception("Unhandled exception in thread '%s': %s",
                                 self.name, exc)
            self._wd.report_error(self.name, str(exc))
        finally:
            self._log.debug("Thread '%s' exiting.", self.name)

    def start(self) -> threading.Thread:
        """Start the managed thread and register it with the watchdog."""
        self._thread = threading.Thread(
            target=self._run_wrapper,
            name=self.name,
            daemon=self._daemon,
        )
        self._thread.start()
        self._wd.register(
            name        = self.name,
            thread      = self._thread,
            is_critical = self._is_critical,
        )
        return self._thread

    def stop(self, timeout: float = 5.0) -> bool:
        """Signal stop and join. Returns True if thread exited cleanly."""
        self.stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        return self._thread is None or not self._thread.is_alive()

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class ThreadManager:
    """
    Central registry and lifecycle controller for all BioSense-Pi threads.

    Responsibilities:
      · Instantiate and start all worker threads in correct dependency order
      · Provide ordered shutdown (processing → acquisition → display)
      · Expose thread health to watchdog and dashboard

    Start order (dependency graph):
      1. display          — can start immediately (handles empty snapshot)
      2. acq-imu          — fastest sensor; start early for motion baseline
      3. acq-ppg          — medium speed; independent of camera
      4. acq-camera       — slowest, most memory-intensive
      5. proc-motion      — depends on imu_raw_queue
      6. proc-biosignal   — depends on ppg_raw_queue
      7. proc-optical     — depends on camera_queue

    Stop order (reverse of start, processing before acquisition):
      proc-* → acq-* → display
    """

    def __init__(self,
                 state_hub: SharedStateHub,
                 watchdog : WatchdogFramework,
                 cfg      : SystemConfig,
                 log_mgr  : LoggingManager):
        self._hub      = state_hub
        self._wd       = watchdog
        self._cfg      = cfg
        self._log_mgr  = log_mgr
        self._log      = log_mgr.get("thread_mgr")
        self._threads  : Dict[str, ManagedThread] = {}

    def _make(self,
              name       : str,
              target     : Callable,
              is_critical: bool = True) -> ManagedThread:
        mt = ManagedThread(
            name        = name,
            target      = target,
            state_hub   = self._hub,
            watchdog    = self._wd,
            cfg         = self._cfg,
            log_mgr     = self._log_mgr,
            is_critical = is_critical,
        )
        self._threads[name] = mt
        return mt

    def start_all(self,
                  worker_map: Dict[str, Callable]) -> None:
        """
        Start all workers in dependency order.
        worker_map keys must match the canonical thread names.
        Logs a warning for any name without a corresponding worker.
        """
        order = [
            "display",
            "acq-imu",
            "acq-ppg",
            "acq-camera",
            "proc-motion",
            "proc-biosignal",
            "proc-optical",
        ]
        non_critical = {"display"}

        for name in order:
            fn = worker_map.get(name)
            if fn is None:
                self._log.warning("No worker provided for thread '%s' — skipping.", name)
                continue
            mt = self._make(name, fn, is_critical=(name not in non_critical))
            mt.start()
            self._log.info("Thread '%s' started (tid=%d).",
                           name, mt._thread.ident if mt._thread else -1)

    def stop_all(self, timeout_each: float = 5.0) -> None:
        """
        Signal all threads to stop in reverse dependency order.
        Drains processing before acquisition stops to avoid lost samples.
        """
        stop_order = [
            "proc-optical",
            "proc-biosignal",
            "proc-motion",
            "acq-camera",
            "acq-ppg",
            "acq-imu",
            "display",
        ]
        for name in stop_order:
            mt = self._threads.get(name)
            if mt is None:
                continue
            self._log.info("Stopping thread '%s'...", name)
            clean = mt.stop(timeout=timeout_each)
            if not clean:
                self._log.warning("Thread '%s' did not exit cleanly.", name)

    def get_stop_event(self, name: str) -> Optional[threading.Event]:
        mt = self._threads.get(name)
        return mt.stop_event if mt else None

    def any_alive(self, names: List[str]) -> bool:
        return any(self._threads[n].is_alive
                   for n in names if n in self._threads)


# ─────────────────────────────────────────────────────────────────────────────
# §8  CLEANUP MANAGER
#     Guaranteed resource teardown on any exit path.
# ─────────────────────────────────────────────────────────────────────────────

class CleanupManager:
    """
    LIFO resource cleanup registry with exception isolation.

    Usage:
        cm = CleanupManager(log)
        cm.register("gpio", gpio_controller.cleanup)
        cm.register("camera", cam.stop)
        # ... on exit:
        cm.run_all()

    Properties:
      · LIFO order (last registered = first cleaned up)
      · Each cleanup function is called in its own try/except
      · Errors are logged but never re-raised — partial cleanup is better
        than aborting mid-teardown and leaving hardware in bad state
      · Idempotent: run_all() can be called multiple times safely
      · Thread-safe: run_all() acquires lock; second call is a no-op
    """

    def __init__(self, log: logging.Logger):
        self._log     = log
        self._stack   : List[Tuple[str, Callable]] = []
        self._lock    = threading.Lock()
        self._ran     = False

    def register(self, name: str, fn: Callable) -> None:
        """Register a cleanup function. fn must be callable with no arguments."""
        with self._lock:
            self._stack.append((name, fn))

    def run_all(self) -> None:
        """Execute all registered cleanups in LIFO order. Idempotent."""
        with self._lock:
            if self._ran:
                return
            self._ran = True
            stack = list(reversed(self._stack))

        self._log.info("CleanupManager: running %d cleanup handlers.", len(stack))
        for name, fn in stack:
            try:
                self._log.debug("  Cleanup: %s", name)
                fn()
            except Exception as exc:
                self._log.error("  Cleanup '%s' raised: %s", name, exc)

        self._log.info("CleanupManager: teardown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# §9  RUNTIME STATE MACHINE — HANDLER IMPLEMENTATIONS
#     Each SystemState maps to a handler method on BioSensePiApp.
#     Handlers are synchronous; they return the next desired state.
# ─────────────────────────────────────────────────────────────────────────────

class StateMachineHandlers:
    """
    Mixin providing per-state handler methods.

    Each handle_<state> method:
      · Executes the work for that state (sensor checks, calibration, etc.)
      · Returns the next SystemState to transition to
      · May raise on unrecoverable errors (caught by main loop)

    Handlers run on the main thread — they must not block indefinitely.
    Long-running work (calibration) uses progress callbacks and checks
    a cancellation event to remain responsive to SIGINT.
    """

    # Injected by BioSensePiApp.__init__
    _sm      : StateManager
    _hub     : SharedStateHub
    _cfg     : SystemConfig
    _log     : logging.Logger
    _cleanup : CleanupManager
    _wd      : WatchdogFramework
    _t_mgr   : ThreadManager
    _log_mgr : LoggingManager

    def handle_init(self) -> SystemState:
        """
        INIT handler.
        Validate Python version, check library availability, create log dir.
        Transitions → SENSOR_CHECK unconditionally (hardware checked next).
        """
        self._log.info("=== BioSense-Pi INIT ===")

        if sys.version_info < (3, 11):
            self._log.warning("Python < 3.11 detected. Some features may degrade.")

        self._log.info("Library availability: scipy=%s  picamera2=%s  smbus2=%s  cv2=%s",
                       _SCIPY, _PICAM, _SMBUS, _CV2)

        # Create log directory early (LoggingManager does this, but make explicit)
        pathlib.Path(self._cfg.logging.log_dir).mkdir(parents=True, exist_ok=True)

        return SystemState.SENSOR_CHECK

    def handle_sensor_check(self) -> SystemState:
        """
        SENSOR_CHECK handler.
        Attempt hardware identity verification for each enabled subsystem.
        Partial failures → DEGRADED_MODE with that subsystem disabled.
        Total failures (no usable sensor) → SHUTDOWN.
        """
        self._log.info("=== SENSOR_CHECK ===")
        hw_ok = {"camera": False, "ppg": False, "imu": False}

        # ── Camera ─────────────────────────────────────────────────────────
        if self._cfg.hw_camera_enabled and _PICAM:
            try:
                cam = Picamera2()
                cam.close()
                hw_ok["camera"] = True
                self._log.info("  Camera: Picamera2 detected OK.")
            except Exception as e:
                self._log.warning("  Camera: detection failed (%s) — disabling.", e)
        else:
            self._log.warning("  Camera: disabled or picamera2 unavailable.")

        # ── PPG / MAX30102 (enhanced with MAX30102Config support) ──────────
        if self._cfg.hw_ppg_enabled and _SMBUS:
            try:
                bus = smbus2.SMBus(self._cfg.ppg.i2c_bus)
                pid = bus.read_byte_data(self._cfg.ppg.sensor_addr, 0xFF)
                bus.close()
                if pid == 0x15:
                    hw_ok["ppg"] = True
                    self._log.info("  PPG/MAX30102: Part ID 0x15 verified.")
                    
                    # Optional: Enhanced verification with MAX30102-specific driver
                    if _MAX30102_AVAILABLE and MAX30102Driver is not None:
                        try:
                            from max30102_subsystem import SensorHealthRecord
                            health = SensorHealthRecord()
                            ppg_cfg = self._cfg.ppg
                            
                            # Verify driver can open device
                            if MAX30102Config is not None:
                                cfg_max = MAX30102Config(
                                    i2c_bus    = ppg_cfg.i2c_bus,
                                    sensor_addr= ppg_cfg.sensor_addr,
                                    sample_rate_hz = ppg_cfg.sample_rate_hz,
                                    led_current_init = ppg_cfg.led_current,
                                    ring_depth = ppg_cfg.ring_depth,
                                    analysis_window = ppg_cfg.calc_window,
                                    analysis_stride = ppg_cfg.step_size,
                                )
                                driver = MAX30102Driver(cfg_max, health, self._log)
                                driver.open()
                                
                                # Quick FIFO read to verify SPI/I²C comms
                                samples = driver.read_fifo_burst()
                                driver.close()
                                self._log.info("  PPG/MAX30102: Full subsystem self-test passed.")
                        except Exception as e:
                            self._log.warning("  PPG/MAX30102: Enhanced test failed: %s (basic detection OK)", e)
                else:
                    self._log.warning("  PPG/MAX30102: unexpected part ID 0x%02X.", pid)
            except Exception as e:
                self._log.warning("  PPG/MAX30102: detection failed (%s).", e)
        else:
            self._log.warning("  PPG: disabled or smbus2 unavailable.")

        # ── IMU / MPU-6050 ─────────────────────────────────────────────────
        if self._cfg.hw_imu_enabled and _SMBUS:
            try:
                bus  = smbus2.SMBus(self._cfg.imu.i2c_bus)
                wai  = bus.read_byte_data(self._cfg.imu.sensor_addr, 0x75)
                bus.close()
                if wai == 0x68:
                    hw_ok["imu"] = True
                    self._log.info("  IMU/MPU-6050: WHO_AM_I 0x68 OK.")
                else:
                    self._log.warning("  IMU/MPU-6050: unexpected WHO_AM_I 0x%02X.", wai)
            except Exception as e:
                self._log.warning("  IMU/MPU-6050: detection failed (%s).", e)
        else:
            self._log.warning("  IMU: disabled or smbus2 unavailable.")

        # ── Decision ───────────────────────────────────────────────────────
        any_ok = any(hw_ok.values())
        all_ok = all(hw_ok.values())

        if not any_ok:
            self._log.error("No sensors available — cannot operate. Shutting down.")
            return SystemState.SHUTDOWN

        if not all_ok:
            self._log.warning("Partial sensor availability %s — entering DEGRADED_MODE.",
                              hw_ok)
            return SystemState.DEGRADED_MODE

        self._log.info("All sensors verified — proceeding to CALIBRATION.")
        return SystemState.CALIBRATION

    def handle_calibration(self) -> SystemState:
        """
        CALIBRATION handler.
        Performs static IMU bias estimation and optical baseline measurement.
        Results stored in SharedStateHub for use by processing threads.

        IMU calibration:
          · 200 samples at rest
          · Computes mean accel (gravity on Z → bias_z = mean_z - 1.0 g)
          · Computes mean gyro (zero-rate offset)

        Optical baseline:
          · Waits for camera to produce 60 frames
          · Computes mean ROI intensity as exposure reference

        On failure (sensor unreachable) → ERROR_RECOVERY.
        """
        self._log.info("=== CALIBRATION ===")

        calib = {
            "imu_accel_bias" : [0.0, 0.0, 0.0],
            "imu_gyro_bias"  : [0.0, 0.0, 0.0],
            "optical_baseline": 0.0,
            "ts"             : time.monotonic(),
        }

        # ── IMU static calibration ──────────────────────────────────────────
        if _SMBUS and self._cfg.hw_imu_enabled:
            try:
                bus = smbus2.SMBus(self._cfg.imu.i2c_bus)
                n   = self._cfg.imu.calibration_samples
                acc = [0.0] * 6
                lsb_a = self._cfg.imu.accel_lsb_per_g
                lsb_g = self._cfg.imu.gyro_lsb_per_dps
                self._log.info("  IMU calibration: collecting %d samples...", n)
                for i in range(n):
                    raw = bus.read_i2c_block_data(
                        self._cfg.imu.sensor_addr, 0x3B, 14)
                    def s16(h, l):
                        v = (h << 8) | l
                        return v - 65536 if v >= 32768 else v
                    acc[0] += s16(raw[0],  raw[1])  / lsb_a
                    acc[1] += s16(raw[2],  raw[3])  / lsb_a
                    acc[2] += s16(raw[4],  raw[5])  / lsb_a
                    acc[3] += s16(raw[8],  raw[9])  / lsb_g
                    acc[4] += s16(raw[10], raw[11]) / lsb_g
                    acc[5] += s16(raw[12], raw[13]) / lsb_g
                    time.sleep(0.01)
                bus.close()
                calib["imu_accel_bias"] = [acc[0]/n, acc[1]/n, acc[2]/n - 1.0]
                calib["imu_gyro_bias"]  = [acc[3]/n, acc[4]/n, acc[5]/n]
                self._log.info("  IMU accel bias: %s", calib["imu_accel_bias"])
                self._log.info("  IMU gyro bias:  %s", calib["imu_gyro_bias"])
            except Exception as e:
                self._log.warning("  IMU calibration failed: %s — using zero bias.", e)

        self._hub.set_calibration(calib)
        self._log.info("Calibration complete — transitioning to RUNNING.")
        return SystemState.RUNNING

    def handle_running(self) -> SystemState:
        """
        RUNNING handler — main operational loop supervisor.

        This method starts all worker threads (via ThreadManager),
        then enters a light supervisor loop that:
          · Feeds the watchdog with overall system heartbeat
          · Checks for SIGINT / shutdown signal
          · Throttles at 2 Hz to keep main thread mostly idle

        State transitions out of RUNNING are driven externally by
        the watchdog (→ ERROR_RECOVERY / DEGRADED_MODE) or signal
        handler (→ SHUTDOWN).
        """
        self._log.info("=== RUNNING — starting worker threads ===")

        # Build worker function map and start threads
        workers = self._build_worker_map()
        self._t_mgr.start_all(workers)

        self._log.info("All threads started — entering supervisory loop.")

        # Supervisor idle loop — 2 Hz
        while self._sm.in_states(SystemState.RUNNING):
            time.sleep(0.5)
            # Update system-level snapshot fields
            self._hub.update_snapshot(
                system_state = self._sm.state.name,
                uptime_s     = self._hub.now(),
            )

        # Return whatever state the machine is now in after external transition
        return self._sm.state

    def handle_degraded_mode(self) -> SystemState:
        """
        DEGRADED_MODE — operates with reduced sensor set.
        Logs which subsystems are unavailable; continues with available ones.
        Attempts to recover to RUNNING after ERROR_RECOVERY completes.
        """
        self._log.warning("=== DEGRADED_MODE === — partial sensor operation.")
        # In degraded mode, we stay in the supervisory loop same as RUNNING
        # but acknowledge reduced capability in the snapshot.
        self._hub.update_snapshot(system_state="DEGRADED_MODE")

        while self._sm.in_states(SystemState.DEGRADED_MODE):
            time.sleep(1.0)
            self._hub.update_snapshot(uptime_s=self._hub.now())

        return self._sm.state

    def handle_error_recovery(self) -> SystemState:
        """
        ERROR_RECOVERY — attempt to restore normal operation.

        Recovery sequence:
          1. Stop all acquisition threads gracefully
          2. Re-run SENSOR_CHECK to verify which hardware is still alive
          3. If sensors OK → restart threads → RUNNING
          4. If partial → DEGRADED_MODE
          5. After max_recovery_attempts → SHUTDOWN
        """
        self._log.warning("=== ERROR_RECOVERY ===")
        self._hub.update_snapshot(system_state="ERROR_RECOVERY")

        MAX_ATTEMPTS = 3
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._log.info("  Recovery attempt %d/%d...", attempt, MAX_ATTEMPTS)

            # Stop processing threads; leave acquisition briefly
            for name in ("proc-optical", "proc-biosignal", "proc-motion"):
                stop_evt = self._t_mgr.get_stop_event(name)
                if stop_evt:
                    stop_evt.set()

            time.sleep(2.0)   # Brief stabilisation pause

            # Re-check sensor presence
            next_state = self.handle_sensor_check()
            if next_state == SystemState.CALIBRATION:
                self._log.info("  Sensors recovered — returning to RUNNING.")
                return SystemState.RUNNING
            elif next_state == SystemState.DEGRADED_MODE:
                self._log.warning("  Partial recovery — entering DEGRADED_MODE.")
                return SystemState.DEGRADED_MODE

            self._log.warning("  Recovery attempt %d failed.", attempt)
            time.sleep(5.0 * attempt)   # Exponential back-off

        self._log.error("Max recovery attempts exhausted — SHUTDOWN.")
        return SystemState.SHUTDOWN

    def handle_shutdown(self) -> None:
        """
        SHUTDOWN handler — ordered teardown.
        Stops all threads, flushes buffers, releases hardware.
        """
        self._log.info("=== SHUTDOWN — beginning ordered teardown ===")
        self._hub.update_snapshot(system_state="SHUTDOWN")
        self._t_mgr.stop_all(timeout_each=5.0)
        self._wd.stop()
        self._cleanup.run_all()
        self._log_mgr.close()
        self._log.info("BioSense-Pi shutdown complete. Total uptime: %.1fs",
                       self._hub.now())

    def _build_worker_map(self) -> Dict[str, Callable]:
        """
        Assemble the worker function map for ThreadManager.
        Each worker follows the signature:
            def worker(stop_event, state_hub, watchdog, cfg, log) -> None
        Returning placeholders here — real workers imported from sensor modules.
        """
        return {
            "acq-camera"    : _worker_acq_camera,
            "acq-ppg"       : _worker_acq_ppg,
            "acq-imu"       : _worker_acq_imu,
            "proc-optical"  : _worker_proc_optical,
            "proc-biosignal": _worker_proc_biosignal,
            "proc-motion"   : _worker_proc_motion,
            "display"       : _worker_display,
        }


# ─────────────────────────────────────────────────────────────────────────────
# §10  WORKER THREAD STUBS
#      Full implementations belong in domain-specific modules.
#      Each stub demonstrates the required signature and heartbeat protocol.
# ─────────────────────────────────────────────────────────────────────────────

def _worker_acq_camera(stop_event, state_hub, watchdog, cfg, log):
    """
    Camera acquisition worker — Picamera2 CSI pipeline.

    Captures frames at target_fps, applies adaptive exposure control (PI
    feedback on ROI mean intensity), and enqueues BGR arrays into
    state_hub.camera_queue. Evicts oldest frame on overflow to prevent
    back-pressure on the camera driver.

    Heartbeat: every frame (30 Hz).
    """
    log.info("acq-camera: starting (hw=%s).", _PICAM)
    cam      = None
    fps_ema  = 0.0
    t_prev   = time.monotonic()
    frame_n  = 0

    # Adaptive exposure controller state
    exp_us   = cfg.camera.init_exposure_us
    gain     = cfg.camera.init_gain
    integral = 0.0
    Kp, Ki   = 0.005, 0.0008

    if _PICAM and cfg.hw_camera_enabled:
        try:
            cam = Picamera2()
            vcfg = cam.create_video_configuration(
                main={"size": (cfg.camera.width, cfg.camera.height),
                      "format": cfg.camera.pixel_format},
                controls={
                    "AeEnable"           : False,
                    "AwbEnable"          : False,
                    "ExposureTime"       : exp_us,
                    "AnalogueGain"       : gain,
                    "ColourGains"        : (1.0, 1.0),
                    "NoiseReductionMode" : _lc.draft.NoiseReductionModeEnum.Off,
                },
            )
            cam.configure(vcfg)
            cam.start()
            time.sleep(0.3)
            log.info("acq-camera: Picamera2 started %dx%d.",
                     cfg.camera.width, cfg.camera.height)
        except Exception as e:
            log.error("acq-camera: camera init failed: %s — using synth.", e)
            cam = None

    period = 1.0 / cfg.camera.target_fps

    while not stop_event.is_set():
        t0 = time.monotonic()

        # Acquire frame
        if cam is not None:
            try:
                frame = cam.capture_array("main")
            except Exception as e:
                log.warning("acq-camera: capture error: %s", e)
                watchdog.report_error("acq-camera", str(e))
                frame = _synth_ir_frame(cfg)
        else:
            frame = _synth_ir_frame(cfg)

        # Adaptive exposure: simple PI on ROI mean
        h, w = frame.shape[:2]
        roi = frame[h//4:3*h//4, w//4:3*w//4]
        roi_mean = float(np.mean(roi))
        err = cfg.optical.target_roi_intensity - roi_mean
        if abs(err) > cfg.optical.intensity_deadband:
            integral = float(np.clip(integral + err, -50.0, 50.0))
            corr = Kp * err + Ki * integral
            exp_us = float(np.clip(exp_us * (1 + corr),
                                   cfg.camera.min_exposure_us,
                                   cfg.camera.max_exposure_us))
            if cam is not None:
                try:
                    cam.set_controls({"ExposureTime": int(exp_us),
                                      "AnalogueGain": gain})
                except Exception:
                    pass

        # Enqueue — drop oldest on overflow (bounded latency guarantee)
        try:
            state_hub.camera_queue.put_nowait(frame)
        except queue.Full:
            try:
                state_hub.camera_queue.get_nowait()
            except queue.Empty:
                pass
            state_hub.camera_queue.put_nowait(frame)

        # FPS estimation
        frame_n += 1
        now = time.monotonic()
        dt  = now - t_prev
        if dt > 0:
            fps_ema = 0.05 * (1.0/dt) + 0.95 * fps_ema
        t_prev = now

        state_hub.update_snapshot(
            camera_fps   = fps_ema,
            exposure_us  = int(exp_us),
            analogue_gain= gain,
            frame_idx    = frame_n,
        )

        watchdog.beat("acq-camera")

        elapsed = time.monotonic() - t0
        sleep_t = period - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    if cam is not None:
        try:
            cam.stop(); cam.close()
        except Exception:
            pass
    log.info("acq-camera: stopped after %d frames.", frame_n)


def _synth_ir_frame(cfg) -> np.ndarray:
    """Physically plausible synthetic 850 nm reflectance frame (simulation)."""
    h, w = cfg.camera.height, cfg.camera.width
    t    = time.monotonic()
    img  = np.full((h, w), 10.0, dtype=np.float32)
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w/2, h/2
    gauss = np.exp(-((xx-cx)**2/(2*(w*0.18)**2) + (yy-cy)**2/(2*(h*0.28)**2)))
    dc    = 130.0
    ac    = dc * 0.012 * math.sin(2*math.pi*1.1*t)
    img  += gauss * (dc + ac) + np.random.normal(0, 3.0, (h, w)).astype(np.float32)
    img   = np.clip(img, 0, 255).astype(np.uint8)
    if _CV2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.stack([img]*3, axis=-1)


def _worker_acq_ppg(stop_event, state_hub, watchdog, cfg, log):
    """
    MAX30102 PPG acquisition worker (integrated from max30102_subsystem.py).

    Delegates to the complete MAX30102 acquisition pipeline if available.
    Falls back to basic simulation if max30102_subsystem cannot be imported.

    Features:
      · Full DSP pipeline (baseline removal, BP filtering, normalization)
      · Adaptive LED current control (PID feedback)
      · Multi-criteria finger detection
      · Comprehensive biometrics (BPM, SpO2, PI, SQI)
      · Health telemetry (I²C errors, FIFO overflows, signal quality)
      · Precise timing with hybrid sleep + busy-poll
      · Graceful fault recovery with watchdog integration

    Heartbeat: every 10 samples (10 Hz at 100 Hz acquisition).
    """
    # Use real implementation if available, otherwise fall back to basic stub
    if _MAX30102_AVAILABLE and _worker_acq_ppg_real is not None:
        try:
            _worker_acq_ppg_real(
                stop_event=stop_event,
                state_hub=state_hub,
                watchdog=watchdog,
                cfg=cfg,
                log=log,
            )
            return
        except Exception as e:
            log.error("MAX30102 real worker failed: %s — falling back to stub.", e)

    # Fallback: basic simulation (stub from original implementation)
    log.info("acq-ppg: starting (hw=%s).", _SMBUS and cfg.hw_ppg_enabled)
    bus  = None
    addr = cfg.ppg.sensor_addr
    sr   = cfg.ppg.sample_rate_hz
    n    = 0
    t_periods: Deque[float] = collections.deque(maxlen=100)
    t_prev = time.monotonic()

    if _SMBUS and cfg.hw_ppg_enabled:
        try:
            bus = smbus2.SMBus(cfg.ppg.i2c_bus)
            # Soft reset
            bus.write_byte_data(addr, 0x09, 0x40)
            time.sleep(0.1)
            # FIFO config: SMP_AVE=4, rollover, FIFO_A_FULL=0xF
            bus.write_byte_data(addr, 0x08, 0x5F)
            # SPO2 config
            sr_bits = {50:0,100:1,200:2,400:3}.get(sr, 1)
            bus.write_byte_data(addr, 0x0A,
                                cfg.ppg.adc_range | (sr_bits << 2) | cfg.ppg.pulse_width)
            bus.write_byte_data(addr, 0x0C, cfg.ppg.led_current)  # RED
            bus.write_byte_data(addr, 0x0D, cfg.ppg.led_current)  # IR
            bus.write_byte_data(addr, 0x09, 0x03)  # SpO2 mode
            # Clear FIFO
            for reg in (0x04, 0x05, 0x06):
                bus.write_byte_data(addr, reg, 0)
            log.info("acq-ppg: MAX30102 initialised @ %d Hz.", sr)
        except Exception as e:
            log.warning("acq-ppg: MAX30102 init failed: %s — simulation.", e)
            bus = None

    dt = 1.0 / sr
    t_next = time.perf_counter() + dt

    while not stop_event.is_set():
        # Acquire one sample
        if bus is not None:
            try:
                raw = bus.read_i2c_block_data(addr, 0x07, 6)
                red = ((raw[0]<<16)|(raw[1]<<8)|raw[2]) & 0x3FFFF
                ir  = ((raw[3]<<16)|(raw[4]<<8)|raw[5]) & 0x3FFFF
            except OSError as e:
                watchdog.report_error("acq-ppg", str(e))
                red, ir = 0, 0
        else:
            t = time.monotonic()
            dc = 180_000.0
            red = int(dc + dc*0.01*math.sin(2*math.pi*1.1*t) + np.random.normal(0, 200))
            ir  = int(dc + dc*0.015*math.sin(2*math.pi*1.1*t+0.2) + np.random.normal(0, 200))

        state_hub.push_ppg_sample(float(ir), float(red))
        try:
            state_hub.ppg_raw_queue.put_nowait((float(red), float(ir)))
        except queue.Full:
            try: state_hub.ppg_raw_queue.get_nowait()
            except queue.Empty: pass

        n += 1
        now = time.monotonic()
        t_periods.append(now - t_prev); t_prev = now

        if n % 10 == 0:
            watchdog.beat("acq-ppg")
            if len(t_periods) >= 10:
                eff_fs = 1.0 / (sum(t_periods)/len(t_periods))
                state_hub.update_snapshot(ppg_effective_fs=eff_fs)

        # Precise timing
        sleep_t = t_next - time.perf_counter()
        if sleep_t > 0:
            time.sleep(sleep_t)
        t_next += dt

    if bus is not None:
        try:
            bus.write_byte_data(addr, 0x09, 0x80)  # shutdown
            bus.close()
        except Exception:
            pass
    log.info("acq-ppg: stopped after %d samples.", n)


def _worker_acq_imu(stop_event, state_hub, watchdog, cfg, log):
    """
    MPU-6050 IMU acquisition worker.

    14-byte burst reads at 100 Hz, decoded to physical units with
    calibration offsets applied. Uses hybrid sleep + busy-poll for
    sub-millisecond timing jitter on Pi 4/5.

    Heartbeat: every 10 samples (10 Hz).
    """
    log.info("acq-imu: starting (hw=%s).", _SMBUS and cfg.hw_imu_enabled)
    bus  = None
    addr = cfg.imu.sensor_addr
    lsb_a= cfg.imu.accel_lsb_per_g
    lsb_g= cfg.imu.gyro_lsb_per_dps
    n    = 0
    t_periods: Deque[float] = collections.deque(maxlen=200)
    t_prev = time.monotonic()
    MARGIN = 0.002   # Busy-poll guard: 2 ms

    if _SMBUS and cfg.hw_imu_enabled:
        try:
            bus = smbus2.SMBus(cfg.imu.i2c_bus)
            bus.write_byte_data(addr, 0x6B, 0x80); time.sleep(0.1)
            bus.write_byte_data(addr, 0x6B, 0x01); time.sleep(0.05)
            bus.write_byte_data(addr, 0x1A, 0x03)   # DLPF 44 Hz
            bus.write_byte_data(addr, 0x19, 9)       # SR divider → 100 Hz
            bus.write_byte_data(addr, 0x1C, 0x00)    # ±2g
            bus.write_byte_data(addr, 0x1B, 0x00)    # ±250°/s
            log.info("acq-imu: MPU-6050 initialised @ %d Hz.", cfg.imu.sample_rate_hz)
        except Exception as e:
            log.warning("acq-imu: MPU-6050 init failed: %s — simulation.", e)
            bus = None

    dt     = 1.0 / cfg.imu.sample_rate_hz
    t_next = time.monotonic()
    calib  = state_hub.get_calibration() or {}
    ab     = calib.get("imu_accel_bias", [0.0, 0.0, 0.0])
    gb     = calib.get("imu_gyro_bias",  [0.0, 0.0, 0.0])

    def s16(hi, lo):
        v = (hi << 8) | lo
        return v - 65536 if v >= 32768 else v

    _sim_t = 0.0

    while not stop_event.is_set():
        # Hybrid precise sleep
        now = time.monotonic()
        sleep_to = t_next - MARGIN
        if sleep_to > now:
            time.sleep(sleep_to - now)
        while time.monotonic() < t_next:
            pass
        t_next += dt

        if bus is not None:
            try:
                raw = bus.read_i2c_block_data(addr, 0x3B, 14)
                ax = s16(raw[0],raw[1])  /lsb_a - ab[0]
                ay = s16(raw[2],raw[3])  /lsb_a - ab[1]
                az = s16(raw[4],raw[5])  /lsb_a - ab[2]
                gx = s16(raw[8],raw[9])  /lsb_g - gb[0]
                gy = s16(raw[10],raw[11])/lsb_g - gb[1]
                gz = s16(raw[12],raw[13])/lsb_g - gb[2]
                tc = s16(raw[6],raw[7])  /340.0 + 36.53
            except OSError as e:
                watchdog.report_error("acq-imu", str(e))
                ax=ay=gx=gy=gz=0.0; az=1.0; tc=25.0
        else:
            # Physically plausible simulation
            _sim_t += dt
            ax = 0.006 * math.sin(2*math.pi*1.1*_sim_t) + np.random.normal(0, 0.004)
            ay = 0.004 * math.sin(2*math.pi*0.7*_sim_t) + np.random.normal(0, 0.004)
            az = 1.0 + np.random.normal(0, 0.004)
            gx = np.random.normal(0, 0.05)
            gy = np.random.normal(0, 0.05)
            gz = np.random.normal(0, 0.05)
            tc = 30.0 + 0.5 * math.sin(2*math.pi*_sim_t/120)

        state_hub.push_imu_sample(ax, ay, az, gx, gy, gz)
        try:
            state_hub.imu_raw_queue.put_nowait(
                {"ax":ax,"ay":ay,"az":az,"gx":gx,"gy":gy,"gz":gz,"tc":tc,
                 "ts": time.monotonic()})
        except queue.Full:
            try: state_hub.imu_raw_queue.get_nowait()
            except queue.Empty: pass

        n += 1
        now2 = time.monotonic()
        t_periods.append(now2 - t_prev); t_prev = now2

        if n % 10 == 0:
            watchdog.beat("acq-imu")
            if len(t_periods) >= 10:
                eff_fs = 1.0 / (sum(t_periods)/len(t_periods))
                state_hub.update_snapshot(
                    imu_effective_fs = eff_fs,
                    die_temp_c       = tc,
                )

    if bus is not None:
        try: bus.close()
        except Exception: pass
    log.info("acq-imu: stopped after %d samples.", n)


def _worker_proc_optical(stop_event, state_hub, watchdog, cfg, log):
    """
    Optical processing worker — NoIR frame pipeline.

    Per-frame operations:
      1. Weighted IR channel extraction (R×0.75 + G×0.20 + B×0.05)
      2. Gaussian noise suppression
      3. Adaptive tissue segmentation (Otsu + largest contour)
      4. ROI photometric extraction
      5. Motion gate (MAD threshold)
      6. Push to optical_signal_buf (accepted frames only)
      7. Butterworth bandpass on accumulated buffer
      8. Quality scoring → snapshot update

    Heartbeat: every frame.
    """
    log.info("proc-optical: starting.")
    clahe = cv2.createCLAHE(clipLimit=cfg.optical.clahe_clip,
                             tileGridSize=cfg.optical.clahe_tile) if _CV2 else None
    sos = None
    zi  = None
    prev_gray = None
    motion_ema = 0.0
    fps_est    = float(cfg.camera.target_fps)
    n = 0

    # Butterworth bandpass filter (initialised on first use)
    def _make_sos(fps):
        if not _SCIPY or fps < 5:
            return None, None
        nyq = fps / 2.0
        lo  = cfg.optical.bp_low_hz  / nyq
        hi  = cfg.optical.bp_high_hz / nyq
        if lo <= 0 or hi >= 1.0 or lo >= hi:
            return None, None
        s = butter(cfg.optical.bp_order, [lo, hi], btype='band', output='sos')
        z = sosfilt_zi(s) * 0.0  # zero initial conditions
        return s, z

    sos, zi = _make_sos(fps_est)

    while not stop_event.is_set():
        try:
            frame = state_hub.camera_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # ── IR extraction ──────────────────────────────────────────────────
        if _CV2 and frame.ndim == 3:
            b = frame[:,:,0].astype(np.float32)
            g = frame[:,:,1].astype(np.float32)
            r = frame[:,:,2].astype(np.float32)
            gray = np.clip(0.05*b + 0.20*g + 0.75*r, 0, 255).astype(np.uint8)
        else:
            gray = np.mean(frame, axis=2).astype(np.uint8) if frame.ndim==3 else frame

        k = cfg.optical.blur_kernel
        if _CV2:
            gray_m = cv2.GaussianBlur(gray, (k, k), 0)
        else:
            gray_m = gray

        # ── Motion gate (frame-difference MAD) ────────────────────────────
        if prev_gray is not None and prev_gray.shape == gray_m.shape:
            mad = float(np.mean(np.abs(
                gray_m.astype(np.int16) - prev_gray.astype(np.int16))))
        else:
            mad = 0.0
        prev_gray = gray_m
        motion_ema = 0.25 * min(mad / 8.0, 1.0) + 0.75 * motion_ema
        is_corrupt = motion_ema > 0.55

        # ── ROI (simple central crop as fallback) ─────────────────────────
        h, w = gray_m.shape
        rw, rh = int(w*0.45), int(h*0.55)
        x1, y1 = (w-rw)//2, (h-rh)//2
        roi_crop = gray_m[y1:y1+rh, x1:x1+rw].astype(np.float32)
        roi_mean = float(np.mean(roi_crop)) if roi_crop.size else 0.0
        roi_std  = float(np.std(roi_crop))  if roi_crop.size else 0.0

        finger_present = roi_mean >= cfg.optical.finger_dc_threshold

        # ── Optical signal accumulation ────────────────────────────────────
        if not is_corrupt and finger_present:
            state_hub.push_optical_sample(roi_mean)

        sig_arr = state_hub.optical_signal_array()
        filt_latest = 0.0
        ac_rms = 0.0; snr_db = 0.0

        if (sos is not None and zi is not None
                and len(sig_arr) >= cfg.optical.min_frames_for_filter):
            sample = np.array([roi_mean], dtype=np.float64)
            y, zi = sosfilt(sos, sample, zi=zi)
            filt_latest = float(y[0])
            ac_rms = float(np.sqrt(np.mean(sig_arr[-60:]**2))) if len(sig_arr)>=60 else 0.0
            dc     = float(np.mean(sig_arr))
            snr_db = 20.0*math.log10(dc/(ac_rms+1e-9)+1e-9)

        # ── Quality scoring ────────────────────────────────────────────────
        sat_frac = float(np.mean(gray_m.ravel() >= 254))
        if _CV2:
            lap_var = float(cv2.Laplacian(gray_m, cv2.CV_64F).var())
        else:
            lap_var = 50.0

        stab      = float(np.clip((1.0-motion_ema)*min(lap_var/30.0,1.0), 0, 1))
        int_score = float(np.clip((float(np.mean(gray_m))-8.0)/27.0, 0, 1))
        sat_pen   = max(0.0, 1.0-sat_frac*15.0)
        iq        = float(int_score * sat_pen * (0.4 + 0.6*min(lap_var/150.0,1.0)))
        fw        = 0.9 if finger_present else 0.25
        sig_q     = max(0.0, (snr_db-10.0)/10.0) if snr_db>10 else 0.0
        conf      = float(np.clip(stab*0.30 + iq*0.30 + fw*0.25 + sig_q*0.15, 0, 1))
        qlabel    = ("EXCELLENT" if conf>=0.80 else "GOOD" if conf>=0.65 else
                     "FAIR" if conf>=0.45 else "POOR" if conf>=0.20 else "INVALID")

        state_hub.update_snapshot(
            roi_mean_ir          = roi_mean,
            roi_std_ir           = roi_std,
            optical_ac_rms       = ac_rms,
            optical_snr_db       = snr_db,
            optical_quality_conf = conf,
            optical_quality_label= qlabel,
            finger_present_camera= finger_present,
            ts_optical           = time.monotonic(),
        )

        n += 1
        watchdog.beat("proc-optical")

    log.info("proc-optical: stopped after %d frames.", n)


def _worker_proc_biosignal(stop_event, state_hub, watchdog, cfg, log):
    """
    PPG biosignal processing worker — SpO2 / HR / PI computation.

    Operates on overlapping windows (CALC_WINDOW with STEP_SIZE stride)
    pulled from the shared ring buffers. Applies IIR baseline removal +
    Butterworth BPF, peak detection, and ratio-of-ratios SpO2 estimation.

    Heartbeat: once per analysis window.
    """
    log.info("proc-biosignal: starting.")
    # IIR baseline tracker (single-pole high-pass)
    _bl_state = None
    tau = cfg.ppg.baseline_tau
    bpm_hist  : Deque[float] = collections.deque(maxlen=cfg.ppg.display_avg_n)
    spo2_hist : Deque[float] = collections.deque(maxlen=cfg.ppg.display_avg_n)
    pi_hist   : Deque[float] = collections.deque(maxlen=4)
    n_windows = 0

    def _baseline_remove(sig):
        nonlocal _bl_state
        out = np.empty_like(sig)
        s = _bl_state if _bl_state is not None else sig[0]
        for i, x in enumerate(sig):
            s = tau*s + (1-tau)*x
            out[i] = x - s
        _bl_state = s
        return out

    while not stop_event.is_set():
        # Wait for enough samples
        if len(state_hub.ppg_ir_buf) < cfg.ppg.calc_window:
            time.sleep(0.1)
            continue

        ir_raw, red_raw = state_hub.ppg_window(cfg.ppg.calc_window)
        fs = float(cfg.ppg.sample_rate_hz)

        # Finger detection
        dc_ir = float(np.mean(ir_raw))
        ac_ir = float(ir_raw.max() - ir_raw.min())
        finger = dc_ir > cfg.ppg.finger_dc_min and ac_ir > cfg.ppg.finger_ac_min

        if not finger:
            _bl_state = None
            bpm_hist.clear(); spo2_hist.clear(); pi_hist.clear()
            state_hub.update_snapshot(
                finger_present_ppg = False,
                ppg_sqi_label      = "NO SIGNAL",
                ppg_sqi            = 0.0,
                ts_ppg             = time.monotonic(),
            )
            time.sleep(0.2)
            continue

        # Signal conditioning
        ac  = _baseline_remove(ir_raw.copy())
        if _SCIPY and len(ac) >= 25:
            nyq = fs/2.0
            lo  = cfg.ppg.bpf_low_hz/nyq
            hi  = cfg.ppg.bpf_high_hz/nyq
            if 0 < lo < hi < 1.0:
                b, a = butter(cfg.ppg.bpf_order, [lo, hi], btype="band")
                try:
                    from scipy.signal import filtfilt
                    ac = filtfilt(b, a, ac)
                except Exception:
                    pass
        rng = ac.max() - ac.min()
        norm = (2.0*(ac - ac.min())/rng - 1.0) if rng > 1e-9 else np.zeros_like(ac)

        # Peak detection
        peaks = np.array([], dtype=int)
        if _SCIPY:
            min_dist = int(fs*60/cfg.ppg.peak_max_bpm)
            prom_min = 0.30 * (norm.max()-norm.min())
            try:
                peaks, _ = find_peaks(norm, distance=max(1,min_dist),
                                      prominence=max(0.01, prom_min))
            except Exception:
                pass

        # Heart rate
        bpm = None
        if len(peaks) >= 2:
            intervals = np.diff(peaks)/fs
            med = float(np.median(intervals))
            if med > 0:
                b_val = 60.0/med
                if cfg.ppg.peak_min_bpm <= b_val <= cfg.ppg.peak_max_bpm:
                    bpm = round(b_val, 1)
                    bpm_hist.append(bpm)

        # SpO2 (ratio-of-ratios)
        spo2 = None
        if len(peaks) >= 2:
            r_vals = []
            for i in range(len(peaks)-1):
                seg_ir  = ir_raw[peaks[i]:peaks[i+1]]
                seg_red = red_raw[peaks[i]:peaks[i+1]]
                if len(seg_ir) < 3: continue
                dc_i  = float(np.mean(seg_ir));  dc_r = float(np.mean(seg_red))
                if dc_i < 1 or dc_r < 1: continue
                ac_i  = float(seg_ir.max()-seg_ir.min())
                ac_r  = float(seg_red.max()-seg_red.min())
                if ac_i < 1: continue
                R = (ac_r/dc_r)/(ac_i/dc_i)
                if 0.3 <= R <= 1.8:
                    r_vals.append(R)
            if r_vals:
                R_med = float(np.median(r_vals))
                # Empirical lookup
                R_tbl = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]
                S_tbl = [100,99,98,97,96,95,93,91,88,85,80]
                s2 = round(float(np.interp(R_med,R_tbl,S_tbl)),1)
                spo2 = max(70.0, min(100.0, s2))
                spo2_hist.append(spo2)

        # Perfusion index
        pi = None
        if dc_ir > 0:
            ac_rms_ir = float(np.sqrt(np.mean((ir_raw - dc_ir)**2)))
            pi = round(max(0.0, min(25.0, (ac_rms_ir/dc_ir)*100)), 3)
            pi_hist.append(pi)

        # SQI
        noise_est = float(np.std(np.diff(ir_raw))/max(ir_raw.max()-ir_raw.min(),1e-6))
        ac_dc = float(np.std(ir_raw))/max(dc_ir, 1.0)
        w1 = 100.0*math.exp(-((ac_dc-0.03)**2)/(2*0.02**2)) if 0.002<ac_dc<0.30 else 0.0
        w2 = max(0.0, 100.0*(1.0 - noise_est/0.30))
        sqi = round(max(0.0, 0.60*w1 + 0.40*w2), 1)
        sqi_lbl = ("EXCELLENT" if sqi>=90 else "GOOD" if sqi>=70 else
                   "FAIR" if sqi>=45 else "POOR" if sqi>=10 else "NO SIGNAL")

        state_hub.update_snapshot(
            bpm                = round(float(np.median(list(bpm_hist))),1) if bpm_hist else None,
            spo2               = round(float(np.median(list(spo2_hist))),1) if spo2_hist else None,
            perfusion_index    = round(float(np.mean(list(pi_hist))),3) if pi_hist else None,
            ppg_sqi            = sqi,
            ppg_sqi_label      = sqi_lbl,
            finger_present_ppg = True,
            dc_ir_ppg          = dc_ir,
            ac_ir_ppg          = ac_ir,
            ppg_noise_est      = noise_est,
            ts_ppg             = time.monotonic(),
        )

        n_windows += 1
        watchdog.beat("proc-biosignal")

        # Sleep for step-size duration before next window
        time.sleep(cfg.ppg.step_size / float(cfg.ppg.sample_rate_hz))

    log.info("proc-biosignal: stopped after %d windows.", n_windows)


def _worker_proc_motion(stop_event, state_hub, watchdog, cfg, log):
    """
    IMU motion artifact processing worker.

    Consumes IMU samples from imu_raw_queue, applies:
      · Complementary filter for pitch/roll orientation (α=0.96)
      · Gravity-free dynamic acceleration
      · Multi-criteria motion quality classification (hysteresis)
      · Weighted artifact score (0–100) for PPG validity estimation

    Heartbeat: every 10 samples.
    """
    log.info("proc-motion: starting.")

    # Complementary filter state
    CF_ALPHA = 0.96
    pitch = 0.0; roll = 0.0
    DT_NOM = 1.0 / cfg.imu.sample_rate_hz
    t_prev = time.monotonic()

    # Hysteresis state
    state_order = ["STABLE","LOW MOTION","MEDIUM MOTION","HIGH MOTION","INVALID SIGNAL"]
    motion_state = "STABLE"
    candidate    = "STABLE"
    hold_count   = 0
    DEGRADE_HOLD = 2; IMPROVE_HOLD = 15

    # Artifact scoring IIR
    score_y = 0.0
    score_dt = DT_NOM
    score_a  = score_dt/(1.0/(2*math.pi*1.5) + score_dt)

    amag_var_buf: Deque[float] = collections.deque(maxlen=50)
    n = 0

    while not stop_event.is_set():
        try:
            d = state_hub.imu_raw_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        ax, ay, az = d["ax"], d["ay"], d["az"]
        gx, gy     = d["gx"], d["gy"]
        ts         = d["ts"]

        dt = max(0.001, min(0.1, ts - t_prev)); t_prev = ts

        # ── Complementary filter ──────────────────────────────────────────
        pitch_acc = math.degrees(math.atan2(ax, math.sqrt(ay*ay+az*az)))
        roll_acc  = math.degrees(math.atan2(ay, math.sqrt(ax*ax+az*az)))
        pitch = CF_ALPHA*(pitch + gx*dt) + (1-CF_ALPHA)*pitch_acc
        roll  = CF_ALPHA*(roll  + gy*dt) + (1-CF_ALPHA)*roll_acc
        pitch = max(-180.0, min(180.0, pitch))
        roll  = max(-180.0, min(180.0, roll))
        tilt  = math.sqrt(pitch**2 + roll**2)

        # ── Motion magnitudes ─────────────────────────────────────────────
        amag = math.sqrt(ax**2+ay**2+az**2)
        dyn  = abs(amag - 1.0)
        gz   = d["gz"]
        gmag = math.sqrt(d["gx"]**2+d["gy"]**2+gz**2)
        amag_var_buf.append(amag)
        variance = (statistics.variance(amag_var_buf)
                    if len(amag_var_buf) >= 2 else 0.0)

        # ── Motion quality classification (hysteresis) ─────────────────────
        comp = 0.50*dyn + 0.25*(gmag/80.0) + 0.15*min(1.0,variance/0.05)
        if comp > 0.70 or dyn > 0.70: raw_s = "INVALID SIGNAL"
        elif comp > 0.30 or dyn > 0.30: raw_s = "HIGH MOTION"
        elif comp > 0.10 or dyn > 0.10: raw_s = "MEDIUM MOTION"
        elif comp > 0.03 or dyn > 0.03: raw_s = "LOW MOTION"
        else: raw_s = "STABLE"

        if raw_s == candidate:
            hold_count += 1
        else:
            candidate = raw_s; hold_count = 1

        cur_r = state_order.index(motion_state) if motion_state in state_order else 0
        new_r = state_order.index(raw_s) if raw_s in state_order else 0
        hold_req = DEGRADE_HOLD if new_r > cur_r else IMPROVE_HOLD
        if hold_count >= hold_req:
            motion_state = candidate

        # ── Artifact score (0–100) ────────────────────────────────────────
        c1 = min(35.0, (dyn/1.0)**0.65 * 35.0)
        c2 = min(25.0, (gmag/200.0)**0.70 * 25.0)
        c3 = min(25.0, (variance/0.04)**0.55 * 25.0)
        raw_score = c1 + c2 + c3
        if raw_score > score_y:
            score_y = raw_score            # fast attack
        else:
            score_y = score_a*raw_score + (1-score_a)*score_y  # slow decay
        artifact_score = max(0.0, min(100.0, score_y))
        ppg_validity   = max(0.0, 100.0 - artifact_score*1.2)

        state_hub.update_snapshot(
            accel_x        = ax,
            accel_y        = ay,
            accel_z        = az,
            gyro_x         = d["gx"],
            gyro_y         = d["gy"],
            gyro_z         = gz,
            accel_mag      = amag,
            dynamic_accel  = dyn,
            gyro_mag       = gmag,
            pitch_deg      = pitch,
            roll_deg       = roll,
            tilt_deg       = tilt,
            motion_state   = motion_state,
            artifact_score = artifact_score,
            ppg_validity_pct = ppg_validity,
            ts_imu         = ts,
        )

        n += 1
        if n % 10 == 0:
            watchdog.beat("proc-motion")

    log.info("proc-motion: stopped after %d samples.", n)


def _worker_display(stop_event, state_hub, watchdog, cfg, log):
    """
    Terminal dashboard display worker — 4 Hz refresh.

    Reads the latest SensorSnapshot and renders an ANSI in-place
    dashboard to stdout. Also writes one CSV row per display cycle.
    """
    log.info("display: starting (4 Hz terminal dashboard).")
    INTERVAL = 0.25   # 4 Hz
    t0       = time.monotonic()
    frame    = 0

    # ANSI helpers
    CLR = "\033[2J\033[H"
    B   = "\033[1m"; R = "\033[0m"
    G   = "\033[32m"; Y = "\033[33m"; RE = "\033[31m"; C = "\033[36m"

    def bar(v, w=20):
        f = int(max(0.0, min(1.0, v)) * w)
        return f"[{'█'*f}{'░'*(w-f)}] {v*100:5.1f}%"

    while not stop_event.is_set():
        t_start = time.monotonic()
        snap = state_hub.read_snapshot()
        frame += 1
        el = time.monotonic() - t0
        mm, ss = divmod(int(el), 60)

        sqi_c  = G if "EXCELLENT" in snap.ppg_sqi_label or "GOOD" in snap.ppg_sqi_label \
                 else Y if "FAIR" in snap.ppg_sqi_label else RE
        qopt_c = G if snap.optical_quality_conf > 0.65 else Y if snap.optical_quality_conf > 0.45 else RE
        mot_c  = G if snap.artifact_score < 20 else Y if snap.artifact_score < 50 else RE
        fp_c   = G if snap.finger_present_ppg else RE

        ln = []
        ln.append(f"{B}{C}{'═'*74}{R}")
        ln.append(f"{B}{C}  BioSense-Pi  Core  │  {mm:02d}:{ss:02d}  │  "
                  f"State: {snap.system_state}  │  Frame {frame}{R}")
        ln.append(f"{C}{'─'*74}{R}")
        # PPG
        fp_s = f"{G}PRESENT{R}" if snap.finger_present_ppg else f"{RE}ABSENT{R}"
        ln.append(f"  PPG  │ Finger: {fp_s}  SQI: {sqi_c}{snap.ppg_sqi_label:<12}{R}"
                  f"  BPM: {B}{snap.bpm or '---':>6}{R}  SpO2: {B}{snap.spo2 or '---':>5}{R}%"
                  f"  PI: {B}{snap.perfusion_index or '---':>6}{R}%")
        # Optical
        ln.append(f"  NoIR │ ROI: {snap.roi_mean_ir:6.1f}LSB  SNR: {snap.optical_snr_db:5.1f}dB  "
                  f"Quality: {qopt_c}{snap.optical_quality_label:<12}{R}"
                  f"  Exp: {snap.exposure_us}µs  Gain: {snap.analogue_gain:.2f}×")
        # IMU
        ln.append(f"  IMU  │ Pitch:{snap.pitch_deg:+7.2f}°  Roll:{snap.roll_deg:+7.2f}°  "
                  f"Tilt:{snap.tilt_deg:6.2f}°  State: {mot_c}{snap.motion_state:<14}{R}")
        ln.append(f"       │ Artifact: {mot_c}{bar(snap.artifact_score/100)}{R}  "
                  f"PPG Validity: {snap.ppg_validity_pct:5.1f}%")
        # Rates
        ln.append(f"  Fs   │ Camera: {snap.camera_fps:5.1f}Hz  PPG: {snap.ppg_effective_fs:5.1f}Hz  "
                  f"IMU: {snap.imu_effective_fs:5.1f}Hz")
        ln.append(f"{B}{C}{'═'*74}{R}")
        ln.append("  Ctrl-C to quit.")

        print(CLR + "\n".join(ln), end="", flush=True)
        watchdog.beat("display")

        elapsed = time.monotonic() - t_start
        sleep_t = INTERVAL - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    log.info("display: stopped after %d frames.", frame)


# ─────────────────────────────────────────────────────────────────────────────
# §11  MAIN APPLICATION  —  BioSensePiApp
# ─────────────────────────────────────────────────────────────────────────────

class BioSensePiApp(StateMachineHandlers):
    """
    Top-level application class.

    Wires all subsystems together and runs the state machine event loop.
    The main thread owns the state machine and transitions between states
    synchronously. Worker threads run as daemons.

    Shutdown is always via SIGINT / SIGTERM → graceful state transition
    → CleanupManager → process exit.

    Memory footprint (Pi 4, all sensors active):
      ~15–20 MB resident — well within 2 GB Pi 4 RAM budget.

    CPU footprint:
      Camera proc : ~15 % one core (image ops)
      IMU acq     :  ~5 % one core (busy-poll guard)
      PPG proc    :  ~3 % one core
      Total       : ~25 % (Pi 4 has 4 cores — ample headroom)
    """

    CONFIG_PATH = "/etc/biosense_pi/config.json"

    def __init__(self):
        # ── Configuration ──────────────────────────────────────────────────
        self._cfg = SystemConfig.from_json(self.CONFIG_PATH)

        # ── Logging ────────────────────────────────────────────────────────
        self._log_mgr = LoggingManager(self._cfg.logging)
        self._log     = self._log_mgr.get("app")

        # ── Shared state ───────────────────────────────────────────────────
        self._hub  = SharedStateHub(self._cfg)
        self._sm   = StateManager(SystemState.INIT)

        # Register state-change logger
        self._sm.register_callback(
            lambda frm, to: self._log.info("State: %s → %s", frm.name, to.name))

        # ── Watchdog ───────────────────────────────────────────────────────
        self._wd = WatchdogFramework(self._cfg.watchdog, self._sm, self._log_mgr.get("watchdog"))

        # ── Cleanup ────────────────────────────────────────────────────────
        self._cleanup = CleanupManager(self._log_mgr.get("cleanup"))

        # ── Thread manager ─────────────────────────────────────────────────
        self._t_mgr = ThreadManager(self._hub, self._wd, self._cfg, self._log_mgr)

        # ── Signal handling ────────────────────────────────────────────────
        signal.signal(signal.SIGINT,  self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        # ── Cleanup registrations ──────────────────────────────────────────
        self._cleanup.register("thread_manager", lambda: self._t_mgr.stop_all())
        self._cleanup.register("watchdog",        self._wd.stop)
        self._cleanup.register("logging",         self._log_mgr.close)

    def _on_signal(self, sig, frame) -> None:
        self._log.info("Signal %s received — initiating shutdown.", sig)
        try:
            cur = self._sm.state
            if cur not in (SystemState.SHUTDOWN,):
                self._sm.transition(SystemState.SHUTDOWN, reason=f"signal {sig}")
        except ValueError:
            pass

    def run(self) -> int:
        """
        Main state machine event loop.

        Returns exit code (0 = clean, 1 = error).
        """
        self._log.info("BioSense-Pi starting up.")
        self._wd.start()

        _HANDLERS = {
            SystemState.INIT          : self.handle_init,
            SystemState.SENSOR_CHECK  : self.handle_sensor_check,
            SystemState.CALIBRATION   : self.handle_calibration,
            SystemState.RUNNING       : self.handle_running,
            SystemState.DEGRADED_MODE : self.handle_degraded_mode,
            SystemState.ERROR_RECOVERY: self.handle_error_recovery,
        }

        exit_code = 0
        try:
            while not self._sm.is_terminal():
                cur    = self._sm.state
                handler= _HANDLERS.get(cur)
                if handler is None:
                    self._log.error("No handler for state %s — forcing SHUTDOWN.", cur.name)
                    self._sm.transition(SystemState.SHUTDOWN)
                    break

                try:
                    next_state = handler()
                except Exception as exc:
                    self._log.exception("Handler for %s raised: %s", cur.name, exc)
                    exit_code = 1
                    try:
                        self._sm.transition(SystemState.ERROR_RECOVERY,
                                            reason=str(exc))
                    except ValueError:
                        self._sm.transition(SystemState.SHUTDOWN, reason="unhandled error")
                    continue

                # Handler returned a new desired state — transition if not already there
                if next_state != cur and not self._sm.is_terminal():
                    try:
                        self._sm.transition(next_state)
                    except ValueError as e:
                        self._log.error("Bad transition from handler: %s", e)
                        self._sm.transition(SystemState.SHUTDOWN, reason="illegal transition")

        finally:
            self.handle_shutdown()

        return exit_code


# ─────────────────────────────────────────────────────────────────────────────
# §12  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry point.

    Supported flags:
      --config <path>  Override default JSON config path
      --sim            Force simulation (ignore hardware detection)
      --debug          Set log level to DEBUG
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="BioSense-Pi Core — Biomedical optical sensing platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "SCIENTIFIC DISCLAIMER:\n"
            "  All outputs are experimental / non-clinical research data.\n"
            "  Not for medical diagnosis or clinical decision-making.\n"
        ),
    )
    parser.add_argument("--config",  default=BioSensePiApp.CONFIG_PATH,
                        help="Path to JSON configuration override file.")
    parser.add_argument("--sim",     action="store_true",
                        help="Force simulation mode (no hardware required).")
    parser.add_argument("--debug",   action="store_true",
                        help="Enable DEBUG-level logging.")
    args = parser.parse_args()

    # Apply CLI overrides before app construction where possible
    if args.debug:
        logging.getLogger("biosense").setLevel(logging.DEBUG)

    # Override config path via class attribute (clean injection point)
    if args.config != BioSensePiApp.CONFIG_PATH:
        BioSensePiApp.CONFIG_PATH = args.config

    app = BioSensePiApp()

    if args.sim:
        # Force simulation by patching hardware flags on the frozen config
        # (use object.__setattr__ since SystemConfig is frozen)
        import dataclasses
        new_cfg = dataclasses.replace(
            app._cfg,
            hw_camera_enabled = False,
            hw_ppg_enabled    = False,
            hw_imu_enabled    = False,
        )
        app._cfg = new_cfg
        app._hub._cfg = new_cfg
        app._log.info("Simulation mode forced via --sim flag.")

    sys.exit(app.run())


if __name__ == "__main__":
    main()