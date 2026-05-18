"""
SensorManager — Lifecycle manager for all BioSense-Pi acquisition threads.

Starts, monitors, and stops the three subsystem workers:
  · worker_acq_ppg    (MAX30102 — 100 Hz)
  · worker_acq_imu    (MPU-6050 — 100 Hz)
  · worker_acq_camera (NoIR     — 30 Hz)

Each worker is wrapped in a RestartableThread that respects the stop_event
contract defined in the subsystem files.  Status is exposed to the dashboard
via the status() method for the header and system panel.

Simulation mode:
  When the hardware subsystem packages (smbus2, picamera2) are unavailable
  — which is the expected state when running on a development machine —
  the workers automatically fall back to their built-in simulation mode.
  No code changes are required; the subsystem workers handle this internally.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

log = logging.getLogger("biosense.sensor_mgr")


@dataclass
class SubsystemStatus:
    """Runtime status of one acquisition subsystem."""
    name       : str
    running    : bool  = False
    hw_active  : bool  = False   # True = real hardware, False = simulation
    error_msg  : str   = ""
    last_beat  : float = field(default_factory=time.monotonic)
    restart_cnt: int   = 0


class _NullWatchdog:
    """No-op watchdog — subsystem workers call beat() / report_error() on this."""
    def beat(self, name: str) -> None:
        pass
    def report_error(self, name: str, msg: str) -> None:
        log.warning("[watchdog] %s: %s", name, msg)


class _NullConfig:
    """Minimal config passed to workers when no real SystemConfig is present."""
    hw_ppg_enabled    : bool = False   # force simulation on non-Pi
    hw_imu_enabled    : bool = False
    hw_camera_enabled : bool = False
    cv2_display       : bool = False   # never open CV2 window in dashboard mode

    class ppg:
        i2c_bus=1; sensor_addr=0x57; sample_rate_hz=100
        led_current=0x3C; ring_depth=600; calc_window=250
        step_size=50; bpf_low_hz=0.6; bpf_high_hz=4.0
        bpf_order=4; baseline_tau=0.995
        finger_dc_min=50000; finger_ac_min=300
        peak_min_bpm=30; peak_max_bpm=220; display_avg_n=6

    class imu:
        i2c_bus=1; sensor_addr=0x68; sample_rate_hz=100
        calibration_samples=200; ring_depth=512
        accel_lsb_per_g=16384.0; gyro_lsb_per_dps=131.0
        temp_sensitivity=340.0; temp_offset=36.53

    class camera:
        width=640; height=480; target_fps=30; pixel_format="BGR888"
        init_exposure_us=15000; init_gain=4.0
        min_exposure_us=3000; max_exposure_us=40000
        min_gain=1.0; max_gain=8.0

    class optical:
        target_roi_intensity=140.0; intensity_deadband=12.0
        clahe_clip=2.5; clahe_tile=(8, 8); blur_kernel=5
        bp_low_hz=0.65; bp_high_hz=4.0; bp_order=4
        signal_buffer_frames=300; min_frames_for_filter=60
        finger_dc_threshold=35.0


class SensorManager:
    """
    Manages all sensor acquisition threads with auto-restart on failure.
    """

    _RESTART_DELAY_S = 3.0

    def __init__(self, hub: Any) -> None:
        self._hub     = hub
        self._wd      = _NullWatchdog()
        self._cfg     = _NullConfig()

        self._stop_ppg    = threading.Event()
        self._stop_imu    = threading.Event()
        self._stop_camera = threading.Event()

        self._thread_ppg   : Optional[threading.Thread] = None
        self._thread_imu   : Optional[threading.Thread] = None
        self._thread_camera: Optional[threading.Thread] = None

        self._status: Dict[str, SubsystemStatus] = {
            "ppg"   : SubsystemStatus("MAX30102 PPG"),
            "imu"   : SubsystemStatus("MPU-6050 IMU"),
            "camera": SubsystemStatus("NoIR Camera"),
        }

        self._mgr_thread: Optional[threading.Thread] = None
        self._mgr_stop   = threading.Event()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start_all(self) -> None:
        """Start all acquisition threads and the watchdog manager."""
        self._start_ppg()
        self._start_imu()
        self._start_camera()

        self._mgr_thread = threading.Thread(
            target=self._manager_loop, daemon=True, name="biosense-mgr")
        self._mgr_thread.start()
        log.info("SensorManager: all subsystems launched.")

    def stop_all(self) -> None:
        """Signal all threads to stop and join with 5 s timeout."""
        self._mgr_stop.set()
        for ev in (self._stop_ppg, self._stop_imu, self._stop_camera):
            ev.set()
        for t in (self._thread_ppg, self._thread_imu, self._thread_camera,
                  self._mgr_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        log.info("SensorManager: stopped.")

    # ── Thread starters ────────────────────────────────────────────────────

    def _start_ppg(self) -> None:
        self._stop_ppg.clear()
        self._status["ppg"].running = True
        self._thread_ppg = threading.Thread(
            target=self._run_ppg, daemon=True, name="acq-ppg")
        self._thread_ppg.start()

    def _start_imu(self) -> None:
        self._stop_imu.clear()
        self._status["imu"].running = True
        self._thread_imu = threading.Thread(
            target=self._run_imu, daemon=True, name="acq-imu")
        self._thread_imu.start()

    def _start_camera(self) -> None:
        self._stop_camera.clear()
        self._status["camera"].running = True
        self._thread_camera = threading.Thread(
            target=self._run_camera, daemon=True, name="acq-camera")
        self._thread_camera.start()

    # ── Worker wrappers (each catches all exceptions for resilience) ───────

    def _run_ppg(self) -> None:
        try:
            from max30102_subsystem import worker_acq_ppg
            log.info("PPG worker: importing subsystem ✓")
            self._status["ppg"].hw_active = False
            worker_acq_ppg(
                stop_event   = self._stop_ppg,
                state_hub    = self._hub,
                watchdog     = self._wd,
                cfg          = self._cfg,
                log          = logging.getLogger("acq-ppg"),
            )
        except ImportError as e:
            log.warning("PPG worker: import failed (%s) — using internal sim.", e)
            self._run_ppg_sim()
        except Exception as e:
            log.error("PPG worker crashed: %s", e, exc_info=True)
            self._status["ppg"].error_msg = str(e)
        finally:
            self._status["ppg"].running = False

    def _run_imu(self) -> None:
        try:
            from mpu6050_subsystem import worker_acq_imu
            log.info("IMU worker: importing subsystem ✓")
            self._status["imu"].hw_active = False
            worker_acq_imu(
                stop_event   = self._stop_imu,
                state_hub    = self._hub,
                watchdog     = self._wd,
                cfg          = self._cfg,
                log          = logging.getLogger("acq-imu"),
            )
        except ImportError as e:
            log.warning("IMU worker: import failed (%s) — using internal sim.", e)
            self._run_imu_sim()
        except Exception as e:
            log.error("IMU worker crashed: %s", e, exc_info=True)
            self._status["imu"].error_msg = str(e)
        finally:
            self._status["imu"].running = False

    def _run_camera(self) -> None:
        try:
            from noir_camera_subsystem import worker_acq_camera
            log.info("Camera worker: importing subsystem ✓")
            self._status["camera"].hw_active = False
            worker_acq_camera(
                stop_event   = self._stop_camera,
                state_hub    = self._hub,
                watchdog     = self._wd,
                cfg          = self._cfg,
                log          = logging.getLogger("acq-camera"),
            )
        except ImportError as e:
            log.warning("Camera worker: import failed (%s) — using internal sim.", e)
            self._run_camera_sim()
        except Exception as e:
            log.error("Camera worker crashed: %s", e, exc_info=True)
            self._status["camera"].error_msg = str(e)
        finally:
            self._status["camera"].running = False

    # ── Internal simulators (fallback when subsystem files unavailable) ────

    def _run_ppg_sim(self) -> None:
        """Minimal synthetic PPG for dashboard-only demo mode."""
        import math, numpy as np
        log.info("PPG sim started.")
        t = 0.0; dt = 0.01; f_hr = 1.15
        dc_ir = 185_000.0; dc_red = 150_000.0
        while not self._stop_ppg.is_set():
            noise = float(np.random.normal(0, 300))
            ir  = dc_ir  * (1 + 0.015 * math.sin(2*math.pi*f_hr*t)) + noise
            red = dc_red * (1 + 0.012 * math.sin(2*math.pi*f_hr*t + 0.2)) + noise*0.9
            self._hub.push_ppg_sample(ir, red)
            # Fake biometrics
            if int(t * 2) % 10 == 0:
                self._hub.update_snapshot(
                    bpm=float(f_hr*60 + np.random.normal(0,0.3)),
                    spo2=float(97.0 + np.random.normal(0,0.2)),
                    perfusion_index=float(1.2 + np.random.normal(0,0.05)),
                    ppg_sqi=82.0, ppg_sqi_label="GOOD",
                    finger_present_ppg=True,
                    dc_ir_ppg=dc_ir, ac_ir_ppg=dc_ir*0.015,
                )
            t += dt
            time.sleep(dt)

    def _run_imu_sim(self) -> None:
        """Minimal synthetic IMU for dashboard-only demo mode."""
        import math, numpy as np
        log.info("IMU sim started.")
        t = 0.0; dt = 0.01; f_tr = 7.0
        while not self._stop_imu.is_set():
            noise = float(np.random.normal(0, 0.002))
            ax = 0.005 * math.sin(2*math.pi*f_tr*t) + noise
            ay = 0.003 * math.cos(2*math.pi*f_tr*t) + noise
            az = 1.0   + 0.002 * math.sin(2*math.pi*0.2*t) + noise*0.5
            gx = 0.5   * math.cos(2*math.pi*f_tr*t) + np.random.normal(0,0.04)
            gy = 0.4   * math.sin(2*math.pi*f_tr*t) + np.random.normal(0,0.04)
            gz =          np.random.normal(0, 0.03)
            self._hub.push_imu_sample(ax, ay, az, gx, gy, gz)
            self._hub.update_snapshot(
                accel_x=ax, accel_y=ay, accel_z=az,
                gyro_x=gx, gyro_y=gy, gyro_z=gz,
                accel_mag=float(math.sqrt(ax**2+ay**2+az**2)),
                dynamic_accel=abs(ax)+abs(ay),
                gyro_mag=float(math.sqrt(gx**2+gy**2+gz**2)),
                pitch_deg=float(2.0*math.sin(2*math.pi*0.3*t)),
                roll_deg=float(1.5*math.cos(2*math.pi*0.2*t)),
                motion_state="STABLE",
                ppg_validity_pct=95.0,
                die_temp_c=float(28.0 + 0.5*math.sin(2*math.pi*t/120)),
            )
            t += dt
            time.sleep(dt)

    def _run_camera_sim(self) -> None:
        """Minimal synthetic camera frames for dashboard-only demo mode."""
        import math, numpy as np
        log.info("Camera sim started.")
        t = 0.0; dt = 1.0/30.0; f_hr = 1.1
        H, W = 240, 320
        while not self._stop_camera.is_set():
            # Synthetic IR scene
            yy, xx = np.ogrid[:H, :W]
            cx, cy = W/2 + 5*math.sin(2*math.pi*0.07*t), H/2 + 3*math.sin(2*math.pi*0.05*t)
            gauss = np.exp(-(((xx-cx)/(W*0.18))**2 + ((yy-cy)/(H*0.28))**2)/2.0)
            ac = 0.015 * math.sin(2*math.pi*f_hr*t)
            dc = 130.0
            signal = dc * (1 + ac)
            gray = np.clip(10 + gauss * signal + np.random.normal(0,3,(H,W)), 0,255).astype(np.uint8)
            bgr = np.stack([
                (gray.astype(np.uint16)*70//255).astype(np.uint8),
                (gray.astype(np.uint16)*180//255).astype(np.uint8),
                gray,
            ], axis=2)
            self._hub.push_camera_frame(bgr)
            roi_mean = float(gray[H//4:3*H//4, W//4:3*W//4].mean())
            self._hub.push_optical_sample(roi_mean)
            self._hub.update_snapshot(
                roi_mean_ir=roi_mean,
                optical_snr_db=float(12.0 + 2*math.sin(2*math.pi*0.1*t)),
                optical_quality_conf=0.72,
                optical_quality_label="GOOD",
                finger_present_camera=True,
                camera_fps=30.0,
            )
            t += dt
            time.sleep(dt)

    # ── Manager loop: monitor + restart dead threads ───────────────────────

    def _manager_loop(self) -> None:
        """Checks every 5 s whether any worker has died and restarts it."""
        while not self._mgr_stop.is_set():
            time.sleep(5.0)
            if not self._stop_ppg.is_set() and (
                    self._thread_ppg is None or not self._thread_ppg.is_alive()):
                log.warning("PPG thread died — restarting.")
                self._status["ppg"].restart_cnt += 1
                self._start_ppg()
            if not self._stop_imu.is_set() and (
                    self._thread_imu is None or not self._thread_imu.is_alive()):
                log.warning("IMU thread died — restarting.")
                self._status["imu"].restart_cnt += 1
                self._start_imu()
            if not self._stop_camera.is_set() and (
                    self._thread_camera is None or not self._thread_camera.is_alive()):
                log.warning("Camera thread died — restarting.")
                self._status["camera"].restart_cnt += 1
                self._start_camera()

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, SubsystemStatus]:
        return dict(self._status)
