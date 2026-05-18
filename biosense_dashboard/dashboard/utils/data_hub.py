"""
DataHub — Thread-safe shared state bus for BioSense-Pi dashboard.

All sensor acquisition threads write here; the Streamlit UI thread reads here.
Uses threading.RLock for snapshot consistency and collections.deque ring
buffers for each signal stream.

Architecture rationale:
  Streamlit runs the entire script on every rerun.  Session state holds a
  single DataHub instance; each rerun calls snapshot() which takes an
  instantaneous thread-safe copy of all live values.  Graph components call
  get_buffer() which also acquires the lock for a consistent array snapshot.
"""

from __future__ import annotations

import time
import threading
import collections
from dataclasses import dataclass, field
from typing import Optional, Deque, Dict, Any, List
import numpy as np


# ── Snapshot dataclass (immutable view returned to UI) ─────────────────────────

@dataclass
class SensorSnapshot:
    """Flat, immutable snapshot of all live sensor values for one UI frame."""
    ts: float = 0.0

    # ── MAX30102 PPG ──────────────────────────────────────────────────────
    bpm:              Optional[float] = None
    spo2:             Optional[float] = None
    perfusion_index:  Optional[float] = None
    ppg_sqi:          float = 0.0
    ppg_sqi_label:    str   = "NO SIGNAL"
    finger_present_ppg: bool = False
    dc_ir_ppg:        float = 0.0
    ac_ir_ppg:        float = 0.0
    ppg_effective_fs: float = 100.0
    ppg_noise_est:    float = 1.0

    # ── MPU-6050 IMU ──────────────────────────────────────────────────────
    accel_x:          float = 0.0
    accel_y:          float = 0.0
    accel_z:          float = 0.0
    gyro_x:           float = 0.0
    gyro_y:           float = 0.0
    gyro_z:           float = 0.0
    accel_mag:        float = 0.0
    dynamic_accel:    float = 0.0
    gyro_mag:         float = 0.0
    pitch_deg:        float = 0.0
    roll_deg:         float = 0.0
    tilt_deg:         float = 0.0
    motion_state:     str   = "STABLE"
    artifact_score:   float = 0.0
    ppg_validity_pct: float = 100.0
    imu_effective_fs: float = 100.0
    die_temp_c:       float = 25.0

    # Tremor
    tremor_detected:  bool  = False
    tremor_hz:        float = 0.0
    tremor_band:      str   = "NONE"
    tremor_snr_db:    float = 0.0

    # ── NoIR Camera / iPPG ────────────────────────────────────────────────
    roi_mean_ir:          float = 0.0
    optical_ac_rms:       float = 0.0
    optical_snr_db:       float = 0.0
    optical_quality_conf: float = 0.0
    optical_quality_label:str   = "INVALID"
    finger_present_camera:bool  = False
    exposure_us:          int   = 15_000
    analogue_gain:        float = 4.0
    camera_fps:           float = 0.0

    # ── System ────────────────────────────────────────────────────────────
    i2c_error_count:  int   = 0


# ── Ring buffer depths ─────────────────────────────────────────────────────────
_PPG_DEPTH     = 500   # 5 s @ 100 Hz  (displayed as scrolling waveform)
_IMU_DEPTH     = 300   # 3 s @ 100 Hz
_OPTICAL_DEPTH = 300   # 10 s @ 30 Hz


class DataHub:
    """
    Central data bus.  All acquisition threads write via update_snapshot(),
    push_ppg_sample(), push_imu_sample(), push_optical_sample().
    The UI thread reads via snapshot() and get_buffer().
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snap = SensorSnapshot(ts=time.monotonic())

        # ── Signal ring buffers ───────────────────────────────────────────
        self._ppg_ir  : Deque[float] = collections.deque(maxlen=_PPG_DEPTH)
        self._ppg_red : Deque[float] = collections.deque(maxlen=_PPG_DEPTH)
        self._ppg_ts  : Deque[float] = collections.deque(maxlen=_PPG_DEPTH)

        self._imu_ax  : Deque[float] = collections.deque(maxlen=_IMU_DEPTH)
        self._imu_ay  : Deque[float] = collections.deque(maxlen=_IMU_DEPTH)
        self._imu_az  : Deque[float] = collections.deque(maxlen=_IMU_DEPTH)
        self._imu_gx  : Deque[float] = collections.deque(maxlen=_IMU_DEPTH)
        self._imu_gy  : Deque[float] = collections.deque(maxlen=_IMU_DEPTH)
        self._imu_gz  : Deque[float] = collections.deque(maxlen=_IMU_DEPTH)
        self._imu_mag : Deque[float] = collections.deque(maxlen=_IMU_DEPTH)
        self._imu_ts  : Deque[float] = collections.deque(maxlen=_IMU_DEPTH)

        self._opt_raw : Deque[float] = collections.deque(maxlen=_OPTICAL_DEPTH)
        self._opt_ts  : Deque[float] = collections.deque(maxlen=_OPTICAL_DEPTH)

        # Camera frame (latest only — grabbed by camera panel)
        self._cam_frame = None           # numpy uint8 BGR or None
        self._cam_frame_ts: float = 0.0

        # ── Queues for raw data from subsystem workers ────────────────────
        # These mirror the interface expected by worker_acq_ppg / worker_acq_imu
        self.ppg_raw_queue = collections.deque(maxlen=256)  # (red, ir) tuples
        self.imu_raw_queue = collections.deque(maxlen=256)  # dict
        self.camera_queue  = collections.deque(maxlen=4)    # BGR frames

        # expose as queue.Queue-compatible objects for subsystem workers
        self.ppg_raw_queue  = _DequeQueue(maxsize=256)
        self.imu_raw_queue  = _DequeQueue(maxsize=256)
        self.camera_queue   = _DequeQueue(maxsize=4)

        # iPPG signal buffer (for OpticalSignalBuffer compatibility)
        self.optical_signal_buf = collections.deque(maxlen=_OPTICAL_DEPTH)

    # ── Write interface (called from acquisition threads) ──────────────────

    def update_snapshot(self, **kwargs) -> None:
        """Thread-safe update of one or more snapshot fields."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._snap, k):
                    object.__setattr__(self._snap, k, v)
            object.__setattr__(self._snap, "ts", time.monotonic())

    def push_ppg_sample(self, ir: float, red: float) -> None:
        ts = time.monotonic()
        with self._lock:
            self._ppg_ir.append(ir)
            self._ppg_red.append(red)
            self._ppg_ts.append(ts)

    def push_imu_sample(self, ax: float, ay: float, az: float,
                        gx: float, gy: float, gz: float) -> None:
        ts  = time.monotonic()
        mag = float(np.sqrt(ax**2 + ay**2 + az**2))
        with self._lock:
            self._imu_ax.append(ax);  self._imu_ay.append(ay);  self._imu_az.append(az)
            self._imu_gx.append(gx);  self._imu_gy.append(gy);  self._imu_gz.append(gz)
            self._imu_mag.append(mag); self._imu_ts.append(ts)

    def push_optical_sample(self, value: float) -> None:
        ts = time.monotonic()
        with self._lock:
            self._opt_raw.append(value)
            self._opt_ts.append(ts)
        self.optical_signal_buf.append(value)

    def push_camera_frame(self, frame) -> None:
        with self._lock:
            self._cam_frame    = frame
            self._cam_frame_ts = time.monotonic()

    # ── Read interface (called from UI thread) ─────────────────────────────

    def snapshot(self) -> SensorSnapshot:
        """Return a thread-safe copy of the current snapshot."""
        with self._lock:
            import copy
            return copy.copy(self._snap)

    def get_ppg_buffers(self) -> Dict[str, np.ndarray]:
        with self._lock:
            return {
                "ir" : np.array(self._ppg_ir,  dtype=np.float32),
                "red": np.array(self._ppg_red, dtype=np.float32),
                "ts" : np.array(self._ppg_ts,  dtype=np.float64),
            }

    def get_imu_buffers(self) -> Dict[str, np.ndarray]:
        with self._lock:
            return {
                "ax" : np.array(self._imu_ax,  dtype=np.float32),
                "ay" : np.array(self._imu_ay,  dtype=np.float32),
                "az" : np.array(self._imu_az,  dtype=np.float32),
                "gx" : np.array(self._imu_gx,  dtype=np.float32),
                "gy" : np.array(self._imu_gy,  dtype=np.float32),
                "gz" : np.array(self._imu_gz,  dtype=np.float32),
                "mag": np.array(self._imu_mag, dtype=np.float32),
                "ts" : np.array(self._imu_ts,  dtype=np.float64),
            }

    def get_optical_buffer(self) -> Dict[str, np.ndarray]:
        with self._lock:
            return {
                "raw": np.array(self._opt_raw, dtype=np.float32),
                "ts" : np.array(self._opt_ts,  dtype=np.float64),
            }

    def get_camera_frame(self):
        """Return latest camera frame (numpy BGR) or None."""
        with self._lock:
            return self._cam_frame

    def read_snapshot(self) -> SensorSnapshot:
        """Alias for snapshot() — compatibility with subsystem workers."""
        return self.snapshot()


# ── queue.Queue-compatible deque wrapper ───────────────────────────────────────

import queue as _queue

class _DequeQueue:
    """
    Thin wrapper that makes a deque behave like a queue.Queue (subset of API).
    Used so subsystem workers — which call put_nowait / get_nowait — work
    with the DataHub without holding real queue objects.
    """
    def __init__(self, maxsize: int = 256):
        self._dq = collections.deque(maxlen=maxsize)
        self._lock = threading.Lock()

    def put_nowait(self, item) -> None:
        with self._lock:
            self._dq.append(item)

    def get_nowait(self):
        with self._lock:
            if not self._dq:
                raise _queue.Empty
            return self._dq.popleft()

    def __len__(self) -> int:
        return len(self._dq)
