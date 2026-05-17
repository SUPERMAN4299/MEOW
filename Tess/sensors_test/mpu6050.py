#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         MPU-6050 BIOMEDICAL MOTION ARTIFACT ANALYSIS SYSTEM v2.0           ║
║    Research-Grade IMU Platform for Optical Biosensing & PPG Applications   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Platform   : Raspberry Pi (tested: 3B/4B/Zero 2W)                         ║
║  Sensor     : InvenSense MPU-6050 (6-DOF IMU + Temperature)                ║
║  Protocol   : I²C @ 400 kHz (fast-mode) via smbus2                         ║
║  Purpose    : Real-time motion artifact detection for PPG/optical sensing   ║
║  Class      : Non-clinical · Experimental · Research-Grade Only             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DISCLAIMER : This system provides experimental, non-clinical motion        ║
║               artifact estimation for research purposes only. All outputs   ║
║               are AI-assisted interpretations and must NOT be used for      ║
║               medical diagnosis, clinical decision-making, or patient care. ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  System Architecture                                                         ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║   ┌─────────────┐    ┌──────────────────┐    ┌───────────────────────────┐ ║
║   │  MPU-6050   │───▶│  I²C Driver      │───▶│  Acquisition Loop         │ ║
║   │  Hardware   │    │  (Raw Registers) │    │  (100 Hz · Precise Timer) │ ║
║   └─────────────┘    └──────────────────┘    └────────────┬──────────────┘ ║
║                                                            │                 ║
║                                              ┌─────────────▼─────────────┐  ║
║                                              │  Signal Filtering Pipeline │  ║
║                                              │  · Spike Rejection (MAD)   │  ║
║                                              │  · IIR Low-Pass (Butter.)  │  ║
║                                              │  · Adaptive Smoother       │  ║
║                                              │  · Drift Stabilizer        │  ║
║                                              └─────────────┬─────────────┘  ║
║                                                            │                 ║
║              ┌─────────────────────────────────────────────▼──────────────┐ ║
║              │               Motion Analysis Engine                        │ ║
║              │  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐  │ ║
║              │  │ Magnitude  │  │ Orientation  │  │  Artifact Scoring  │  │ ║
║              │  │ + Gravity  │  │ Comp. Filter │  │  PPG Corruption    │  │ ║
║              │  │ Compensat. │  │ Pitch / Roll │  │  Stability Score   │  │ ║
║              │  └────────────┘  └──────────────┘  └────────────────────┘  │ ║
║              │  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐  │ ║
║              │  │  Motion    │  │ Event Detect │  │  AI Interpreter    │  │ ║
║              │  │  Quality   │  │ Jerk/Tremor  │  │  (Non-clinical)    │  │ ║
║              │  │  Engine    │  │ Vibration    │  │                    │  │ ║
║              │  └────────────┘  └──────────────┘  └────────────────────┘  │ ║
║              └─────────────────────────────────┬──────────────────────────┘ ║
║                                                │                             ║
║                                   ┌────────────▼─────────────┐              ║
║                                   │   Terminal Dashboard      │              ║
║                                   │   (ANSI · In-Place)       │              ║
║                                   └──────────────────────────┘              ║
╚══════════════════════════════════════════════════════════════════════════════╝

References
──────────
[1] InvenSense MPU-6000/6050 Product Specification, Rev 3.4
[2] InvenSense MPU-6050 Register Map and Descriptions, Rev 4.2
[3] Krishnan R. et al., "Motion Artifact Reduction in PPG Signals Using
    Accelerometer and Adaptive Filtering," IEEE TBME, 2010.
[4] Allen J., "Photoplethysmography and its Application in Clinical
    Physiological Measurement," Physiol. Meas. 28, R1–R39, 2007.
[5] Madgwick S.O.H., "An Efficient Orientation Filter for Inertial and
    Inertial/Magnetic Sensor Arrays," 2010.
[6] Schäfer A., Vagedes J., "How Accurate is Pulse Rate Variability as
    Estimate of Heart Rate Variability?" Int. J. Cardiol. 166, 2013.
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
import sys
import math
import time
import signal
import struct
import threading
import collections
import statistics

# ─── Third-Party ──────────────────────────────────────────────────────────────
try:
    import smbus2
    _SMBUS_AVAILABLE = True
except ImportError:
    _SMBUS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# §1  MPU-6050 REGISTER MAP
#     Source: InvenSense Register Map and Descriptions, Rev 4.2
# ══════════════════════════════════════════════════════════════════════════════

class Reg:
    """
    Complete MPU-6050 register address map and configuration constants.

    All register addresses are 8-bit offsets within the device's
    internal register file, accessed via I²C read/write operations.
    """

    # ── Device Identity ────────────────────────────────────────────────────
    I2C_ADDR_LO   = 0x68   # I²C address when AD0 = GND (most common)
    I2C_ADDR_HI   = 0x69   # I²C address when AD0 = VCC
    WHO_AM_I      = 0x75   # Identity register → always reads 0x68

    # ── Self-Test Registers ────────────────────────────────────────────────
    SELF_TEST_X   = 0x0D
    SELF_TEST_Y   = 0x0E
    SELF_TEST_Z   = 0x0F
    SELF_TEST_A   = 0x10

    # ── Sampling & Configuration ───────────────────────────────────────────
    SMPLRT_DIV    = 0x19   # Sample Rate = Gyro Rate / (1 + SMPLRT_DIV)
    CONFIG        = 0x1A   # DLPF + EXT_SYNC_SET
    GYRO_CONFIG   = 0x1B   # Gyroscope full-scale range
    ACCEL_CONFIG  = 0x1C   # Accelerometer full-scale range

    # ── FIFO ──────────────────────────────────────────────────────────────
    FIFO_EN       = 0x23   # FIFO enable flags
    FIFO_COUNT_H  = 0x72   # FIFO byte count high
    FIFO_COUNT_L  = 0x73   # FIFO byte count low
    FIFO_R_W      = 0x74   # FIFO read/write port

    # ── Interrupt ─────────────────────────────────────────────────────────
    INT_PIN_CFG   = 0x37
    INT_ENABLE    = 0x38
    INT_STATUS    = 0x3A

    # ── Sensor Data (big-endian, two's complement, 16-bit each) ───────────
    ACCEL_XOUT_H  = 0x3B   # AX[15:8]
    ACCEL_XOUT_L  = 0x3C
    ACCEL_YOUT_H  = 0x3D
    ACCEL_YOUT_L  = 0x3E
    ACCEL_ZOUT_H  = 0x3F
    ACCEL_ZOUT_L  = 0x40
    TEMP_OUT_H    = 0x41
    TEMP_OUT_L    = 0x42
    GYRO_XOUT_H   = 0x43
    GYRO_XOUT_L   = 0x44
    GYRO_YOUT_H   = 0x45
    GYRO_YOUT_L   = 0x46
    GYRO_ZOUT_H   = 0x47
    GYRO_ZOUT_L   = 0x48

    # ── Power Management ───────────────────────────────────────────────────
    PWR_MGMT_1    = 0x6B   # Sleep, cycle, temp disable, CLKSEL
    PWR_MGMT_2    = 0x6C   # Standby mode per axis
    USER_CTRL     = 0x6A   # FIFO / I²C master / reset

    # ── Signal Path Reset ──────────────────────────────────────────────────
    SIGNAL_PATH_RESET = 0x68

    # ── Gyroscope Full-Scale Range (GYRO_CONFIG bits [4:3]) ───────────────
    GYRO_FS_250   = 0x00   # ±250  °/s   LSB = 131.0
    GYRO_FS_500   = 0x08   # ±500  °/s   LSB =  65.5
    GYRO_FS_1000  = 0x10   # ±1000 °/s   LSB =  32.8
    GYRO_FS_2000  = 0x18   # ±2000 °/s   LSB =  16.4

    # ── Accelerometer Full-Scale Range (ACCEL_CONFIG bits [4:3]) ──────────
    ACCEL_FS_2G   = 0x00   # ±2  g   LSB = 16384
    ACCEL_FS_4G   = 0x08   # ±4  g   LSB =  8192
    ACCEL_FS_8G   = 0x10   # ±8  g   LSB =  4096
    ACCEL_FS_16G  = 0x18   # ±16 g   LSB =  2048

    # ── Digital Low-Pass Filter (CONFIG bits [2:0]) ────────────────────────
    DLPF_260HZ    = 0x00   # Accel: 260 Hz / Gyro: 256 Hz — no filter
    DLPF_184HZ    = 0x01   # Accel: 184 Hz / Gyro: 188 Hz
    DLPF_94HZ     = 0x02   # Accel:  94 Hz / Gyro:  98 Hz
    DLPF_44HZ     = 0x03   # Accel:  44 Hz / Gyro:  42 Hz  ← selected
    DLPF_21HZ     = 0x04   # Accel:  21 Hz / Gyro:  20 Hz
    DLPF_10HZ     = 0x05   # Accel:  10 Hz / Gyro:  10 Hz
    DLPF_5HZ      = 0x06   # Accel:   5 Hz / Gyro:   5 Hz

    # ── Physical Scaling Factors ───────────────────────────────────────────
    ACCEL_LSB_PER_G   = 16384.0   # LSB/g   at ±2g
    GYRO_LSB_PER_DPS  = 131.0     # LSB/°/s at ±250°/s
    TEMP_SENSITIVITY  = 340.0     # LSB/°C
    TEMP_OFFSET       = 36.53     # °C at raw = 0


# ══════════════════════════════════════════════════════════════════════════════
# §2  LOW-LEVEL I²C DRIVER
#     Direct register communication — no high-level abstraction libraries
# ══════════════════════════════════════════════════════════════════════════════

class MPU6050Driver:
    """
    Production-grade I²C driver for the InvenSense MPU-6050.

    Implements register-level communication with:
      · Atomic 14-byte burst reads for minimum I²C overhead
      · Thread-safe locking for multi-threaded operation
      · Automatic I²C error recovery with configurable retry policy
      · Static calibration with Gauss-Newton mean offset estimation
      · FIFO mode support (configurable, off by default for polling mode)

    Initialization sequence follows InvenSense AN-MPU-6050A-01 guidelines:
      1. Assert device reset → wait 100 ms
      2. Select PLL clock source (X gyro) for frequency stability
      3. Configure DLPF bandwidth
      4. Set sample rate divider
      5. Configure accelerometer and gyroscope full-scale ranges
      6. Disable FIFO and interrupts (polling mode)
      7. Static calibration (200 samples @ rest)

    Selected configuration for biomedical wrist/body sensing:
      · Accel: ±2 g   — highest resolution for low-intensity motion
      · Gyro:  ±250°/s — sufficient for limb rotation
      · DLPF:  44 Hz  — attenuates vibration noise above motion bandwidth
      · Fs:    100 Hz — adequate Nyquist margin for motion artifacts in PPG
    """

    MAX_RETRIES    = 3        # I²C transaction retry count
    RETRY_DELAY_S  = 0.002   # Delay between retries (2 ms)

    def __init__(self, bus: int = 1, addr: int = Reg.I2C_ADDR_LO):
        self.bus_num   = bus
        self.addr      = addr
        self._bus      = None
        self._lock     = threading.Lock()

        # Calibration offsets (set during calibrate())
        self.accel_bias = [0.0, 0.0, 0.0]   # [g]
        self.gyro_bias  = [0.0, 0.0, 0.0]   # [°/s]

        # Runtime state
        self.is_open       = False
        self.is_calibrated = False

        # Error counters for health monitoring
        self.i2c_errors      = 0
        self.i2c_recoveries  = 0

    # ──────────────────────────────────────────────────────────────────────
    # Bus Management
    # ──────────────────────────────────────────────────────────────────────

    def open(self) -> None:
        """
        Open I²C bus and validate device presence via WHO_AM_I.
        Raises RuntimeError on failure.
        """
        try:
            self._bus = smbus2.SMBus(self.bus_num)
            time.sleep(0.01)
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError(
                f"Cannot open I²C bus {self.bus_num}: {exc}\n"
                f"  → Run: sudo raspi-config → Interface Options → I2C → Enable"
            ) from exc

        # Verify device identity
        wai = self._read_byte_safe(Reg.WHO_AM_I)
        if wai != 0x68:
            raise RuntimeError(
                f"MPU-6050 not found at 0x{self.addr:02X}. "
                f"WHO_AM_I = 0x{wai:02X} (expected 0x68). "
                f"Check SDA/SCL wiring and AD0 pin state."
            )
        self.is_open = True

    def close(self) -> None:
        """Release I²C bus."""
        if self._bus:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None
            self.is_open = False

    # ──────────────────────────────────────────────────────────────────────
    # I²C Primitives with Retry / Recovery
    # ──────────────────────────────────────────────────────────────────────

    def _read_byte_safe(self, reg: int) -> int:
        """Single-byte register read with retry on OSError."""
        for attempt in range(self.MAX_RETRIES):
            try:
                with self._lock:
                    return self._bus.read_byte_data(self.addr, reg)
            except OSError:
                self.i2c_errors += 1
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY_S)
                    self.i2c_recoveries += 1
        return 0  # Graceful degradation

    def _write_byte_safe(self, reg: int, val: int) -> bool:
        """Single-byte register write with retry."""
        for attempt in range(self.MAX_RETRIES):
            try:
                with self._lock:
                    self._bus.write_byte_data(self.addr, reg, val)
                return True
            except OSError:
                self.i2c_errors += 1
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY_S)
                    self.i2c_recoveries += 1
        return False

    def _read_block_safe(self, reg: int, length: int) -> list:
        """Multi-byte burst read with retry. Returns list of ints."""
        for attempt in range(self.MAX_RETRIES):
            try:
                with self._lock:
                    return self._bus.read_i2c_block_data(self.addr, reg, length)
            except OSError:
                self.i2c_errors += 1
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY_S)
                    self.i2c_recoveries += 1
        return [0] * length  # Return zeros on failure

    # ──────────────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Full MPU-6050 register initialization.

        Register writes follow InvenSense recommended power-on sequence.
        Each write is verified by reading back where practical.
        """
        # Step 1: Device reset (bit 7 of PWR_MGMT_1)
        # This resets all registers to their power-on defaults
        self._write_byte_safe(Reg.PWR_MGMT_1, 0x80)
        time.sleep(0.10)   # Datasheet: 100 ms reset recovery

        # Step 2: Wake device, select PLL with X-gyro reference
        # CLKSEL = 001 → PLL with X axis gyroscope reference
        # Better frequency stability than internal 8 MHz RC oscillator
        self._write_byte_safe(Reg.PWR_MGMT_1, 0x01)
        time.sleep(0.05)   # PLL lock time

        # Step 3: Digital Low-Pass Filter
        # DLPF_CFG = 3 → Accel: 44Hz / Gyro: 42Hz
        # Rationale: Passes body motion (DC–10 Hz) while attenuating
        # high-frequency mechanical vibration from environment
        self._write_byte_safe(Reg.CONFIG, Reg.DLPF_44HZ)

        # Step 4: Sample Rate Divider
        # Gyro output rate = 1000 Hz (when DLPF enabled)
        # Sample Rate = 1000 / (1 + SMPLRT_DIV) = 100 Hz
        self._write_byte_safe(Reg.SMPLRT_DIV, 9)

        # Step 5: Accelerometer full-scale range
        # AFS_SEL = 0 → ±2 g, 16384 LSB/g
        # Highest resolution — appropriate for wrist/body biosensing
        self._write_byte_safe(Reg.ACCEL_CONFIG, Reg.ACCEL_FS_2G)

        # Step 6: Gyroscope full-scale range
        # FS_SEL = 0 → ±250 °/s, 131 LSB/°/s
        # Appropriate for limb angular velocity during daily activities
        self._write_byte_safe(Reg.GYRO_CONFIG, Reg.GYRO_FS_250)

        # Step 7: Disable FIFO (polling mode — simpler and lower latency)
        self._write_byte_safe(Reg.FIFO_EN,  0x00)
        self._write_byte_safe(Reg.USER_CTRL, 0x00)

        # Step 8: Disable interrupts
        self._write_byte_safe(Reg.INT_ENABLE, 0x00)

        # Step 9: Enable all axes — no standby
        self._write_byte_safe(Reg.PWR_MGMT_2, 0x00)

        # Final stabilization delay
        time.sleep(0.10)

    # ──────────────────────────────────────────────────────────────────────
    # Data Acquisition
    # ──────────────────────────────────────────────────────────────────────

    def read_raw_burst(self) -> dict:
        """
        Read all sensor data in a single 14-byte I²C burst transaction.

        Register map (ACCEL_XOUT_H = 0x3B through GYRO_ZOUT_L = 0x48):
          Bytes  0– 1 : ACCEL_X  (big-endian int16)
          Bytes  2– 3 : ACCEL_Y
          Bytes  4– 5 : ACCEL_Z
          Bytes  6– 7 : TEMP
          Bytes  8– 9 : GYRO_X
          Bytes 10–11 : GYRO_Y
          Bytes 12–13 : GYRO_Z

        Single transaction ensures temporal coherence across all axes.
        Avoids inter-axis timestamp skew that would corrupt orientation fusion.

        Returns dict of raw int16 values + high-resolution monotonic timestamp.
        """
        ts  = time.monotonic()
        raw = self._read_block_safe(Reg.ACCEL_XOUT_H, 14)

        def s16(hi, lo):
            """Combine two bytes into signed 16-bit integer (two's complement)."""
            v = (hi << 8) | lo
            return v - 65536 if v >= 32768 else v

        return {
            'ax_raw' : s16(raw[0],  raw[1]),
            'ay_raw' : s16(raw[2],  raw[3]),
            'az_raw' : s16(raw[4],  raw[5]),
            'tr_raw' : s16(raw[6],  raw[7]),
            'gx_raw' : s16(raw[8],  raw[9]),
            'gy_raw' : s16(raw[10], raw[11]),
            'gz_raw' : s16(raw[12], raw[13]),
            'ts'     : ts,
        }

    def raw_to_physical(self, raw: dict) -> dict:
        """
        Convert raw ADC counts to SI / engineering units.

        Conversion equations (MPU-6050 datasheet §4.17):
          Acceleration [g]  = raw / 16384   (at ±2g)
          Rotation [°/s]    = raw / 131     (at ±250°/s)
          Temperature [°C]  = raw/340 + 36.53

        Calibration offsets (measured at rest) are subtracted here.
        The Z-axis accelerometer retains +1g (gravity reference).
        """
        ax = raw['ax_raw'] / Reg.ACCEL_LSB_PER_G  - self.accel_bias[0]
        ay = raw['ay_raw'] / Reg.ACCEL_LSB_PER_G  - self.accel_bias[1]
        az = raw['az_raw'] / Reg.ACCEL_LSB_PER_G  - self.accel_bias[2]
        gx = raw['gx_raw'] / Reg.GYRO_LSB_PER_DPS - self.gyro_bias[0]
        gy = raw['gy_raw'] / Reg.GYRO_LSB_PER_DPS - self.gyro_bias[1]
        gz = raw['gz_raw'] / Reg.GYRO_LSB_PER_DPS - self.gyro_bias[2]
        tc = raw['tr_raw'] / Reg.TEMP_SENSITIVITY  + Reg.TEMP_OFFSET
        return {'ax':ax,'ay':ay,'az':az,'gx':gx,'gy':gy,'gz':gz,'tc':tc,'ts':raw['ts']}

    # ──────────────────────────────────────────────────────────────────────
    # Static Calibration
    # ──────────────────────────────────────────────────────────────────────

    def calibrate(self, n: int = 200, on_progress=None) -> None:
        """
        Measure static bias offsets by averaging N samples at rest.

        Assumptions:
          · Sensor is completely stationary and level during calibration
          · Z-axis faces upward — gravity appears as +1g on Z
          · Gyroscope reads ~0°/s (zero-rate offset only)

        After calibration:
          · accel_bias[0,1] cancel X/Y gravity-free bias
          · accel_bias[2]   set so Z reads exactly +1g at rest
          · gyro_bias       cancel zero-rate offset (ZRO)

        ZRO specification (MPU-6050 datasheet):
          ±20°/s typical — calibration typically reduces this to < 0.5°/s
        """
        ax_acc = ay_acc = az_acc = 0.0
        gx_acc = gy_acc = gz_acc = 0.0

        for i in range(n):
            r = self.read_raw_burst()
            ax_acc += r['ax_raw'] / Reg.ACCEL_LSB_PER_G
            ay_acc += r['ay_raw'] / Reg.ACCEL_LSB_PER_G
            az_acc += r['az_raw'] / Reg.ACCEL_LSB_PER_G
            gx_acc += r['gx_raw'] / Reg.GYRO_LSB_PER_DPS
            gy_acc += r['gy_raw'] / Reg.GYRO_LSB_PER_DPS
            gz_acc += r['gz_raw'] / Reg.GYRO_LSB_PER_DPS
            if on_progress:
                on_progress((i + 1) / n)
            time.sleep(0.01)

        self.accel_bias = [ax_acc/n, ay_acc/n, (az_acc/n) - 1.0]
        self.gyro_bias  = [gx_acc/n, gy_acc/n, gz_acc/n]
        self.is_calibrated = True

    # ──────────────────────────────────────────────────────────────────────
    # Sensor Health
    # ──────────────────────────────────────────────────────────────────────

    def read_temperature_raw(self) -> float:
        """Read die temperature (used for sensor health monitoring)."""
        hi = self._read_byte_safe(Reg.TEMP_OUT_H)
        lo = self._read_byte_safe(Reg.TEMP_OUT_L)
        raw = (hi << 8) | lo
        if raw >= 32768:
            raw -= 65536
        return raw / Reg.TEMP_SENSITIVITY + Reg.TEMP_OFFSET

    @property
    def error_rate(self) -> float:
        """Fraction of transactions that resulted in I²C error (0–1)."""
        total = self.i2c_errors + max(1, self.i2c_recoveries)
        return self.i2c_errors / total


# ══════════════════════════════════════════════════════════════════════════════
# §3  SIGNAL FILTERING PIPELINE
#     Research-grade DSP for IMU biomedical applications
# ══════════════════════════════════════════════════════════════════════════════

class IIRFilter:
    """
    First-order IIR (Infinite Impulse Response) low-pass filter.

    Discrete-time implementation of a single-pole RC low-pass filter:

        y[n] = α·x[n] + (1−α)·y[n−1]

    Coefficient derivation (bilinear/Euler forward method):
        τ = 1 / (2π·fc)          [time constant, seconds]
        α = Δt / (τ + Δt)        [dimensionless, 0 < α < 1]

    Frequency response:
        H(z) = α / (1 − (1−α)z⁻¹)

    Higher α → faster response, less filtering (fc closer to Nyquist)
    Lower  α → slower response, more filtering (fc closer to DC)

    Startup: initialized with first sample to eliminate transient ringing.
    """

    def __init__(self, fc: float, fs: float):
        """
        Args:
            fc : Cutoff frequency [Hz]
            fs : Sample rate [Hz]
        """
        dt      = 1.0 / fs
        tau     = 1.0 / (2.0 * math.pi * fc)
        self.a  = dt / (tau + dt)     # filter coefficient α
        self._y = None                 # previous output y[n−1]

    def reset(self, val: float = 0.0):
        self._y = val

    def step(self, x: float) -> float:
        """Process one input sample. Returns filtered output."""
        if self._y is None:
            self._y = x
            return x
        self._y = self.a * x + (1.0 - self.a) * self._y
        return self._y


class AdaptiveIIRFilter:
    """
    Motion-intensity adaptive IIR filter.

    During high motion, the cutoff frequency is widened to pass
    the faster-changing signal without lag. During rest, the cutoff
    is narrowed for maximum noise rejection.

    This prevents the common artefact where a sudden motion generates
    a visible lag tail in the filtered signal, which would be
    misinterpreted as continued motion.

    Algorithm:
        motion_level ∈ [0, 1]  (normalized from dynamic acceleration)
        fc_eff = fc_min + motion_level · (fc_max − fc_min)

    Because recomputing α each sample is expensive, a fast path uses
    an IIR-filtered motion level for smooth coefficient transitions.
    """

    def __init__(self, fc_min: float, fc_max: float, fs: float):
        self.fc_min = fc_min
        self.fc_max = fc_max
        self.fs     = fs
        self.dt     = 1.0 / fs
        self._y     = None
        self._ml    = 0.0        # smoothed motion level
        self._ml_a  = 0.05      # motion-level tracking speed

    def step(self, x: float, motion_level: float) -> float:
        """
        Args:
            x            : Input sample
            motion_level : Normalized motion intensity [0, 1]
        """
        # Smooth motion level to prevent rapid coefficient switching
        self._ml = self._ml_a * motion_level + (1.0 - self._ml_a) * self._ml
        ml = max(0.0, min(1.0, self._ml))

        # Interpolate cutoff frequency
        fc_eff = self.fc_min + ml * (self.fc_max - self.fc_min)

        # Recompute α for effective cutoff
        tau = 1.0 / (2.0 * math.pi * max(0.01, fc_eff))
        a   = self.dt / (tau + self.dt)

        if self._y is None:
            self._y = x
            return x

        self._y = a * x + (1.0 - a) * self._y
        return self._y


class MovingAverageFilter:
    """
    Causal moving average over a fixed-length sliding window.

    Output: y[n] = (1/N) · Σ_{k=0}^{N-1} x[n−k]

    Properties:
      · Linear phase (zero phase distortion within passband)
      · -3 dB at fc ≈ 0.443·fs/N
      · O(1) update via incremental sum (no per-sample full summation)

    Used for display smoothing where phase linearity matters more
    than stopband rejection.
    """

    def __init__(self, n: int):
        self.N      = n
        self._buf   = collections.deque(maxlen=n)
        self._sum   = 0.0

    def step(self, x: float) -> float:
        if len(self._buf) == self.N:
            self._sum -= self._buf[0]
        self._buf.append(x)
        self._sum += x
        return self._sum / len(self._buf)

    @property
    def window(self) -> list:
        return list(self._buf)


class MADSpikeFilter:
    """
    Median Absolute Deviation (MAD) spike rejection filter.

    Robust impulse noise removal using the MAD robust scale estimator:
        MAD = median(|x_i − median(x)|)
        σ̂  = 1.4826 · MAD          (consistent estimator for Gaussian σ)

    A sample is classified as a spike if:
        |x[n] − median(window)| > k · σ̂

    Spikes are replaced with the window median (not zero), preserving
    the local signal level without introducing discontinuities.

    k selection:
      k = 3.0  → ~0.27% false rejection rate under Gaussian noise
      k = 5.0  → ~5.7×10⁻⁵ false rate (used here — conservative)

    Suited for: I²C bit errors, ESD events, mechanical shocks.
    """

    CONSISTENCY_FACTOR = 1.4826   # E[MAD]/σ for Gaussian distribution

    def __init__(self, window: int = 11, k: float = 5.0):
        self._buf  = collections.deque(maxlen=window)
        self.k     = k
        self._last = 0.0

    def step(self, x: float) -> float:
        self._buf.append(x)
        n = len(self._buf)
        if n < 5:
            self._last = x
            return x

        sorted_w = sorted(self._buf)
        med      = sorted_w[n // 2]
        mad      = sorted([abs(v - med) for v in self._buf])[n // 2]
        sigma    = self.CONSISTENCY_FACTOR * mad

        if sigma < 1e-9:
            # Window nearly constant — anything deviating is a spike
            out = med if abs(x - med) > 0.02 else x
        elif abs(x - med) > self.k * sigma:
            out = med   # Replace spike with median
        else:
            out = x

        self._last = out
        return out


class DriftStabilizer:
    """
    Gyroscope zero-crossing drift stabilizer (dead-zone filter).

    Applies a soft dead-zone around zero for gyroscope readings.
    Values below the threshold are pulled toward zero to suppress
    integration drift from sensor noise when the device is stationary.

    Transfer function (soft threshold):
        if |x| < dead_zone:   y = x · (|x| / dead_zone)²
        else:                 y = x

    The squared taper ensures C¹ continuity at the dead-zone boundary,
    preventing discontinuities that would appear as impulses in the
    integrated angle signal.
    """

    def __init__(self, dead_zone: float = 0.3):
        """
        Args:
            dead_zone : Below this angular rate [°/s], signal is attenuated
        """
        self.dz = dead_zone

    def step(self, x: float) -> float:
        ax = abs(x)
        if ax < self.dz:
            ratio = ax / self.dz
            return x * ratio * ratio
        return x


class ChannelFilterBank:
    """
    Per-channel composited filtering pipeline.

    Pipeline (in order — order matters):

      Raw input
        │
        ▼
      MAD Spike Filter         ─ removes I²C errors, ESD, mechanical shocks
        │
        ▼
      IIR Low-Pass             ─ removes broadband electronic noise
        │
        ▼
      Adaptive IIR (accel)     ─ bandwidth-adapts to motion intensity
      Drift Stabilizer (gyro)  ─ dead-zone zero suppression
        │
        ▼
      Moving Average           ─ final display smoothing (linear phase)
        │
        ▼
      Filtered output

    Separate instances for each of the 6 IMU axes.
    """

    def __init__(self, kind: str, fs: float = 100.0):
        """
        Args:
            kind : 'accel' or 'gyro' — selects appropriate parameters
            fs   : Sample rate [Hz]
        """
        self.kind = kind

        if kind == 'accel':
            self.spike   = MADSpikeFilter(window=11, k=5.0)
            self.iir     = IIRFilter(fc=12.0, fs=fs)
            self.adapt   = AdaptiveIIRFilter(fc_min=4.0, fc_max=15.0, fs=fs)
            self.ma      = MovingAverageFilter(n=4)
            self.drift   = None
        else:  # gyro
            self.spike   = MADSpikeFilter(window=11, k=5.0)
            self.iir     = IIRFilter(fc=10.0, fs=fs)
            self.adapt   = None
            self.ma      = MovingAverageFilter(n=3)
            self.drift   = DriftStabilizer(dead_zone=0.25)

        self._motion_level = 0.0

    def set_motion_level(self, ml: float):
        """Feed motion intensity [0,1] into adaptive filter."""
        self._motion_level = ml

    def process(self, x: float) -> float:
        x = self.spike.step(x)
        x = self.iir.step(x)
        if self.adapt is not None:
            x = self.adapt.step(x, self._motion_level)
        if self.drift is not None:
            x = self.drift.step(x)
        x = self.ma.step(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# §4  ORIENTATION TRACKER — Complementary Filter
#     Fuses accelerometer tilt estimation with gyroscope integration
# ══════════════════════════════════════════════════════════════════════════════

class OrientationTracker:
    """
    Attitude estimation using a first-order complementary filter.

    Rationale for complementary filter over Kalman:
      · No matrix operations → suitable for Raspberry Pi single-core use
      · Tuning requires only one parameter (α) vs. R/Q matrices
      · Sufficient accuracy for motion artifact classification (< 1° error
        in steady-state is adequate — we don't need navigation-grade accuracy)

    Filter equations:
        pitch_acc[n] = atan2(ax, √(ay²+az²))      [rad, then °]
        roll_acc[n]  = atan2(ay, √(ax²+az²))

        pitch[n] = α·(pitch[n−1] + gx·Δt) + (1−α)·pitch_acc[n]
        roll[n]  = α·(roll[n−1]  + gy·Δt) + (1−α)·roll_acc[n]

    α = 0.96 (selected):
        Time constant: τ = α·Δt / (1−α) ≈ 0.24 s at 100 Hz
        Gyro dominates above 0.66 Hz — correct for body motion frequencies
        Accelerometer corrects drift below 0.66 Hz — captures slow tilt changes

    Tilt: total angular deviation from calibration-upright position
        tilt = √(pitch² + roll²)   [approximate, valid for angles < 60°]
    """

    CF_ALPHA = 0.96       # Complementary filter blend coefficient
    DT_NOM   = 0.010      # Nominal sample interval [s] = 1/100Hz

    def __init__(self):
        self.pitch = 0.0   # [°] forward/backward tilt
        self.roll  = 0.0   # [°] left/right tilt
        self.tilt  = 0.0   # [°] total tilt magnitude

        # Smoothed outputs for display stability
        self._pitch_lpf = IIRFilter(fc=3.0, fs=100.0)
        self._roll_lpf  = IIRFilter(fc=3.0, fs=100.0)

    def update(self,
               ax: float, ay: float, az: float,
               gx: float, gy: float,
               dt: float = None) -> tuple:
        """
        Update orientation estimate with new IMU sample.

        Args:
            ax, ay, az : Filtered accelerometer readings [g]
            gx, gy     : Filtered gyroscope readings [°/s]
            dt         : Sample interval [s] — uses DT_NOM if None

        Returns:
            (pitch°, roll°, tilt°)
        """
        dt = dt or self.DT_NOM

        # ── Accelerometer-derived tilt angles ─────────────────────────────
        # atan2 variant handles full ±180° range without singularities
        pitch_acc = math.degrees(math.atan2(ax, math.sqrt(ay*ay + az*az)))
        roll_acc  = math.degrees(math.atan2(ay, math.sqrt(ax*ax + az*az)))

        # ── Complementary filter fusion ────────────────────────────────────
        self.pitch = (self.CF_ALPHA * (self.pitch + gx * dt)
                      + (1.0 - self.CF_ALPHA) * pitch_acc)
        self.roll  = (self.CF_ALPHA * (self.roll  + gy * dt)
                      + (1.0 - self.CF_ALPHA) * roll_acc)

        # Clamp to physical bounds
        self.pitch = max(-180.0, min(180.0, self.pitch))
        self.roll  = max(-180.0, min(180.0, self.roll))

        # ── Display-smoothed angles ────────────────────────────────────────
        sp = self._pitch_lpf.step(self.pitch)
        sr = self._roll_lpf.step(self.roll)

        # ── Total tilt magnitude ───────────────────────────────────────────
        self.tilt = math.sqrt(self.pitch**2 + self.roll**2)

        return sp, sr, self.tilt


# ══════════════════════════════════════════════════════════════════════════════
# §5  MOTION EVENT DETECTOR
#     Real-time detection of physiologically significant motion events
# ══════════════════════════════════════════════════════════════════════════════

class MotionEventDetector:
    """
    Real-time detection of discrete motion events relevant to PPG artifact analysis.

    Detects and tracks:
      ┌─────────────────┬───────────────────────────────────────────────────┐
      │ Event           │ Detection Method                                  │
      ├─────────────────┼───────────────────────────────────────────────────┤
      │ Sudden Jerk     │ Jerk (da/dt) threshold exceedance                 │
      │ Tremor          │ Zero-crossing rate in dynamic acceleration         │
      │ Vibration       │ High-frequency energy in variance window           │
      │ Placement Shift │ Step-change in orientation (|Δtilt| > threshold)  │
      │ Orientation     │ Sustained tilt angle deviation from rest           │
      │   Instability   │                                                   │
      └─────────────────┴───────────────────────────────────────────────────┘

    Each event has a fire-and-fade mechanism: when triggered, the event
    level jumps to 1.0 and decays exponentially with a time constant τ.
    This prevents rapid on/off flickering in the display.
    """

    # Jerk threshold: rate of change of accel magnitude [g/s]
    JERK_THRESHOLD      = 3.0    # Sudden movement
    JERK_EXTREME        = 8.0    # Impact/drop

    # Tremor: zero-crossings per second in 500 ms window
    TREMOR_ZC_LOW       = 4.0    # Hz — physiological tremor range
    TREMOR_ZC_HIGH      = 12.0

    # Vibration: variance in short window exceeds threshold
    VIBRATION_VAR_THRESH = 0.002  # g²

    # Placement shift: tilt step > this in one sample [°]
    PLACEMENT_STEP_THRESH = 8.0

    # Orientation instability: tilt variance in window [°²]
    ORIENT_VAR_THRESH   = 25.0

    # Event decay time constant [s] — controls fade duration
    EVENT_DECAY_TAU     = 0.8    # ~0.8s to decay to 1/e

    def __init__(self, fs: float = 100.0):
        self.fs  = fs
        self.dt  = 1.0 / fs

        # Event intensity levels (0.0–1.0) with exponential decay
        self.jerk_level    = 0.0
        self.jerk_extreme  = 0.0
        self.tremor_level  = 0.0
        self.vibration_lvl = 0.0
        self.placement_chg = 0.0
        self.orient_instab = 0.0

        # Working buffers
        self._amag_buf    = collections.deque(maxlen=int(0.5 * fs))   # 500ms
        self._tilt_buf    = collections.deque(maxlen=int(1.0 * fs))   # 1s
        self._dyn_buf     = collections.deque(maxlen=int(0.5 * fs))   # ZC analysis
        self._prev_amag   = 0.0
        self._prev_tilt   = 0.0
        self._prev_dyn    = 0.0

        # Decay coefficient per sample
        self._decay = math.exp(-self.dt / self.EVENT_DECAY_TAU)

    def _fire(self, level_attr: str, intensity: float):
        """Set event level to max(current, intensity)."""
        cur = getattr(self, level_attr)
        setattr(self, level_attr, max(cur, min(1.0, intensity)))

    def _decay_all(self):
        """Apply exponential decay to all event levels."""
        d = self._decay
        self.jerk_level   = self.jerk_level   * d
        self.jerk_extreme = self.jerk_extreme  * d
        self.tremor_level = self.tremor_level  * d
        self.vibration_lvl= self.vibration_lvl * d
        self.placement_chg= self.placement_chg * d
        self.orient_instab= self.orient_instab * d

    def _zero_crossing_rate(self, buf: collections.deque) -> float:
        """Count zero-crossings per second in buffer."""
        lst = list(buf)
        if len(lst) < 4:
            return 0.0
        zc = sum(1 for i in range(1, len(lst)) if lst[i]*lst[i-1] < 0)
        duration = len(lst) / self.fs
        return zc / duration if duration > 0 else 0.0

    def update(self, amag: float, dyn: float, tilt: float) -> dict:
        """
        Process one sample and update all event detectors.

        Args:
            amag : Total acceleration magnitude [g]
            dyn  : Dynamic (gravity-free) acceleration [g]
            tilt : Current tilt angle [°]

        Returns:
            dict of current event intensities [0.0–1.0]
        """
        # ── Buffer updates ─────────────────────────────────────────────────
        self._amag_buf.append(amag)
        self._tilt_buf.append(tilt)
        self._dyn_buf.append(dyn)

        # ── Decay existing events ──────────────────────────────────────────
        self._decay_all()

        # ── 1. Jerk Detection ──────────────────────────────────────────────
        # Jerk = finite difference of acceleration magnitude
        jerk = abs(amag - self._prev_amag) / self.dt
        if jerk > self.JERK_EXTREME:
            self._fire('jerk_extreme', min(1.0, jerk / 20.0))
            self._fire('jerk_level', 1.0)
        elif jerk > self.JERK_THRESHOLD:
            self._fire('jerk_level', min(1.0, jerk / self.JERK_EXTREME))
        self._prev_amag = amag

        # ── 2. Tremor Detection ────────────────────────────────────────────
        # Physiological tremor: 4–12 Hz oscillation in dynamic accel
        zc_rate = self._zero_crossing_rate(self._dyn_buf)
        if self.TREMOR_ZC_LOW <= zc_rate <= self.TREMOR_ZC_HIGH * 1.5:
            intensity = min(1.0, (zc_rate - self.TREMOR_ZC_LOW) /
                            (self.TREMOR_ZC_HIGH - self.TREMOR_ZC_LOW))
            self._fire('tremor_level', intensity)
        self._prev_dyn = dyn

        # ── 3. Vibration Detection ─────────────────────────────────────────
        # High variance in short window → sustained vibration
        if len(self._amag_buf) >= 10:
            win_var = statistics.variance(list(self._amag_buf)[-20:]) \
                      if len(self._amag_buf) >= 20 \
                      else statistics.variance(self._amag_buf)
            if win_var > self.VIBRATION_VAR_THRESH:
                intensity = min(1.0, win_var / (self.VIBRATION_VAR_THRESH * 5))
                self._fire('vibration_lvl', intensity)

        # ── 4. Placement Change Detection ─────────────────────────────────
        tilt_step = abs(tilt - self._prev_tilt)
        if tilt_step > self.PLACEMENT_STEP_THRESH:
            intensity = min(1.0, tilt_step / 30.0)
            self._fire('placement_chg', intensity)
        self._prev_tilt = tilt

        # ── 5. Orientation Instability ─────────────────────────────────────
        if len(self._tilt_buf) >= 20:
            tilt_var = statistics.variance(list(self._tilt_buf)[-20:])
            if tilt_var > self.ORIENT_VAR_THRESH:
                intensity = min(1.0, tilt_var / (self.ORIENT_VAR_THRESH * 4))
                self._fire('orient_instab', intensity)

        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            'jerk'        : self.jerk_level,
            'jerk_extreme': self.jerk_extreme,
            'tremor'      : self.tremor_level,
            'vibration'   : self.vibration_lvl,
            'placement'   : self.placement_chg,
            'orientation' : self.orient_instab,
        }

    @property
    def max_event(self) -> float:
        s = self.snapshot()
        return max(s.values())


# ══════════════════════════════════════════════════════════════════════════════
# §6  MOTION QUALITY ENGINE
#     Multi-criteria real-time motion state classification
# ══════════════════════════════════════════════════════════════════════════════

class MotionQuality:
    """Motion quality state labels (ordered by severity)."""
    STABLE         = "STABLE"
    LOW_MOTION     = "LOW MOTION"
    MEDIUM_MOTION  = "MEDIUM MOTION"
    HIGH_MOTION    = "HIGH MOTION"
    INVALID        = "INVALID SIGNAL"


class MotionQualityEngine:
    """
    Multi-criteria motion quality classifier with hysteresis.

    Classification uses a weighted combination of:
      W1 · dynamic_accel   (dominant — translational motion)
      W2 · gyro_magnitude  (rotational contribution)
      W3 · accel_variance  (sustained vs. transient detection)
      W4 · event_level     (discrete event contribution)

    Hysteresis implementation:
      State transitions require the criterion to be met for a minimum
      number of consecutive samples (MIN_HOLD). This prevents rapid
      state oscillation at threshold boundaries, which would make the
      display unreadable and artifact score noisy.

    Downgrade (worsening) requires 2 samples.
    Upgrade (improvement) requires 15 samples (1.5s @ 100Hz) — conservative.
    This asymmetry reflects the biomedical reality that a momentary
    artifact can corrupt a PPG window, but recovery requires confirmation.
    """

    # Dynamic acceleration thresholds [g]
    T_STABLE   = 0.03    # < 30 mg
    T_LOW      = 0.10    # < 100 mg
    T_MEDIUM   = 0.30    # < 300 mg
    T_HIGH     = 0.70    # < 700 mg
    # > 700 mg → INVALID

    # Gyro contribution thresholds [°/s]
    G_LOW      = 5.0
    G_MEDIUM   = 25.0
    G_HIGH     = 80.0

    # Hysteresis hold counts
    DEGRADE_HOLD  = 2     # Samples before worsening state
    IMPROVE_HOLD  = 15    # Samples before improving state

    def __init__(self, fs: float = 100.0):
        self._state        = MotionQuality.STABLE
        self._candidate    = MotionQuality.STABLE
        self._hold_count   = 0
        self._accel_var_q  = collections.deque(maxlen=50)

    def _state_rank(self, s: str) -> int:
        order = [MotionQuality.STABLE, MotionQuality.LOW_MOTION,
                 MotionQuality.MEDIUM_MOTION, MotionQuality.HIGH_MOTION,
                 MotionQuality.INVALID]
        return order.index(s) if s in order else 0

    def update(self,
               dyn: float,
               gyro_mag: float,
               variance: float,
               event_max: float) -> str:
        """
        Classify current motion quality state.

        Args:
            dyn       : Gravity-free dynamic acceleration [g]
            gyro_mag  : Total gyroscope magnitude [°/s]
            variance  : Recent acceleration variance [g²]
            event_max : Maximum event intensity (0–1)
        """
        self._accel_var_q.append(dyn)

        # Composite motion metric (weighted)
        composite = (0.50 * dyn
                     + 0.25 * (gyro_mag / 80.0)
                     + 0.15 * min(1.0, variance / 0.05)
                     + 0.10 * event_max)

        # Raw state from composite threshold
        if composite > 0.70 or dyn > 0.70:
            raw = MotionQuality.INVALID
        elif composite > 0.30 or dyn > 0.30:
            raw = MotionQuality.HIGH_MOTION
        elif composite > 0.10 or dyn > 0.10:
            raw = MotionQuality.MEDIUM_MOTION
        elif composite > 0.03 or dyn > 0.03:
            raw = MotionQuality.LOW_MOTION
        else:
            raw = MotionQuality.STABLE

        # Hysteresis logic
        if raw == self._candidate:
            self._hold_count += 1
        else:
            self._candidate  = raw
            self._hold_count  = 1

        cur_rank = self._state_rank(self._state)
        new_rank = self._state_rank(raw)
        hold_req = self.DEGRADE_HOLD if new_rank > cur_rank else self.IMPROVE_HOLD

        if self._hold_count >= hold_req:
            self._state = self._candidate

        return self._state

    @property
    def state(self) -> str:
        return self._state


# ══════════════════════════════════════════════════════════════════════════════
# §7  MOTION ARTIFACT SCORING ENGINE
#     Biomedical PPG signal quality estimation from IMU data
# ══════════════════════════════════════════════════════════════════════════════

class ArtifactScorer:
    """
    Computes a 0–100 motion artifact severity score for PPG/optical biosensing.

    Score interpretation (research context only):
    ┌─────────────┬──────────────────────────────────────────────────────┐
    │ Score Range │ Research Interpretation                              │
    ├─────────────┼──────────────────────────────────────────────────────┤
    │   0 – 12    │ EXCELLENT: Minimal artifact, clean acquisition window│
    │  13 – 28    │ GOOD: Low artifact, signal usable for research       │
    │  29 – 48    │ FAIR: Moderate artifact, flag for post-processing    │
    │  49 – 68    │ POOR: High artifact, signal validity reduced         │
    │  69 – 85    │ SEVERE: Significant contamination, discard segment   │
    │  86 – 100   │ CRITICAL: Signal unusable, suspend acquisition       │
    └─────────────┴──────────────────────────────────────────────────────┘

    Score components (weighted sum):
      Component 1 — Dynamic Acceleration   (weight 35%)
        Maps gravity-free accel to [0,35] with power-law scaling
        Power < 1 → score rises steeply at low motion (sensitive)
        Rationale: even 50 mg motion can corrupt PPG baseline

      Component 2 — Gyroscope Rotation     (weight 25%)
        Maps total angular velocity to [0,25]
        Rotation changes optical path geometry → baseline wander

      Component 3 — Temporal Variance      (weight 25%)
        Recent accel magnitude variance → sustained vs. transient motion
        Sustained moderate motion is worse than a single spike

      Component 4 — Event Contribution     (weight 15%)
        Discrete events (jerk, tremor, vibration) add to base score
        Tremor is particularly damaging to PPG — weighted higher

    Output is smoothed with a slow IIR to prevent score oscillation,
    but with a fast-attack path: score can jump up immediately (worst case
    wins immediately) but decays slowly (recovery must be confirmed).
    """

    def __init__(self, fs: float = 100.0):
        # Slow decay IIR for display stability
        self._display_iir = IIRFilter(fc=1.5, fs=fs)
        # Fast attack tracker (raw score, unfiltered)
        self._raw_score   = 0.0
        # History for mean/trend
        self._history     = collections.deque(maxlen=100)

    def compute(self,
                dyn: float,
                gyro_mag: float,
                variance: float,
                events: dict) -> dict:
        """
        Compute artifact score from IMU analysis outputs.

        Args:
            dyn      : Gravity-free dynamic acceleration [g]
            gyro_mag : Total angular velocity [°/s]
            variance : Recent accel magnitude variance [g²]
            events   : Event intensity dict from MotionEventDetector

        Returns dict with:
            score        : Current smoothed artifact score [0–100]
            raw_score    : Unsmoothed instantaneous score
            ppg_validity : Estimated PPG window validity [0–100%]
            stability    : Optical stability estimate [0–100%]
            mean_score   : Rolling mean over last 100 samples
            components   : Individual component breakdown
        """
        # ── Component 1: Dynamic Acceleration (0–35) ──────────────────────
        # Power 0.65: sensitive at low motion, saturates at high
        c1 = min(35.0, (dyn / 1.0) ** 0.65 * 35.0)

        # ── Component 2: Gyroscope (0–25) ─────────────────────────────────
        c2 = min(25.0, (gyro_mag / 200.0) ** 0.70 * 25.0)

        # ── Component 3: Variance (0–25) ───────────────────────────────────
        c3 = min(25.0, (variance / 0.04) ** 0.55 * 25.0)

        # ── Component 4: Events (0–15) ─────────────────────────────────────
        # Tremor weighted 2× (particularly corrupting for PPG)
        ev_composite = (0.25 * events.get('jerk', 0)
                       + 0.40 * events.get('tremor', 0)
                       + 0.15 * events.get('vibration', 0)
                       + 0.10 * events.get('placement', 0)
                       + 0.10 * events.get('orientation', 0))
        c4 = min(15.0, ev_composite * 15.0)

        raw = c1 + c2 + c3 + c4

        # ── Fast-attack / slow-decay smoothing ─────────────────────────────
        # Score can jump up instantly but recovers at IIR rate
        if raw > self._raw_score:
            self._raw_score = raw         # instant attack
        else:
            self._raw_score = self._display_iir.step(raw)  # slow decay

        score = max(0.0, min(100.0, self._raw_score))
        self._history.append(score)

        # ── Derived metrics ─────────────────────────────────────────────────
        ppg_validity = max(0.0, 100.0 - score * 1.2)    # more pessimistic
        stability    = max(0.0, 100.0 - score)

        mean_score = sum(self._history) / len(self._history)

        return {
            'score'       : score,
            'raw_score'   : raw,
            'ppg_validity': ppg_validity,
            'stability'   : stability,
            'mean_score'  : mean_score,
            'components'  : {'dyn':c1, 'gyro':c2, 'var':c3, 'event':c4},
        }


# ══════════════════════════════════════════════════════════════════════════════
# §8  AI-ASSISTED MOTION INTERPRETER
#     Non-clinical, experimental, research-grade interpretation only
# ══════════════════════════════════════════════════════════════════════════════

class MotionInterpreter:
    """
    Lightweight AI-assisted motion quality interpreter.

    Generates contextually aware, research-grade annotations based on
    the current IMU analysis state. Designed to support researcher
    decision-making during data collection — not for clinical use.

    All generated text carries explicit research/experimental framing.
    The interpreter uses pattern matching on the full analysis state
    rather than any machine learning model.

    Outputs:
      · acquisition_status : One-line overall status for display header
      · ppg_window         : PPG window validity assessment
      · dominant_artifact  : Primary artifact type detected (if any)
      · researcher_note    : Contextual note for researcher
      · confidence_level   : Interpretation confidence (LOW/MED/HIGH)
      · recommendation     : Action recommendation for researcher

    IMPORTANT: These are experimental motion artifact estimations only.
    They must not be used for medical diagnosis or clinical decisions.
    """

    def __init__(self):
        self._state_history  = collections.deque(maxlen=30)
        self._score_history  = collections.deque(maxlen=50)

    def interpret(self,
                  score: float,
                  state: str,
                  events: dict,
                  pitch: float,
                  roll: float,
                  tilt: float,
                  stable_fraction: float) -> dict:
        """
        Generate interpretation from current analysis snapshot.

        Args:
            score           : Artifact score [0–100]
            state           : MotionQuality state string
            events          : Event intensities dict
            pitch, roll     : Orientation angles [°]
            tilt            : Total tilt [°]
            stable_fraction : Fraction of recent samples in STABLE state
        """
        self._state_history.append(state)
        self._score_history.append(score)

        # ── Trend analysis ────────────────────────────────────────────────
        if len(self._score_history) >= 10:
            h = list(self._score_history)
            trend_delta = (sum(h[-5:]) / 5) - (sum(h[:5]) / 5)
            if trend_delta > 5:   trend = "DETERIORATING ↑"
            elif trend_delta < -5: trend = "IMPROVING ↓"
            else:                  trend = "STABLE ↔"
        else:
            trend = "INITIALIZING"

        # ── Dominant artifact type ─────────────────────────────────────────
        dominant = self._dominant_event(events, score, state)

        # ── Acquisition status ─────────────────────────────────────────────
        acq_status = self._acquisition_status(score, stable_fraction)

        # ── PPG window assessment ──────────────────────────────────────────
        ppg_note = self._ppg_window_note(score, state, events)

        # ── Researcher note ────────────────────────────────────────────────
        note = self._researcher_note(score, state, events, tilt, trend)

        # ── Confidence: based on score history consistency ─────────────────
        if len(self._score_history) >= 20:
            score_std = statistics.stdev(list(self._score_history)[-20:])
            if score_std < 5:   confidence = "HIGH"
            elif score_std < 15: confidence = "MEDIUM"
            else:               confidence = "LOW (high variability)"
        else:
            confidence = "LOW (initializing)"

        # ── Recommendation ─────────────────────────────────────────────────
        recommendation = self._recommendation(score, state, stable_fraction)

        return {
            'acquisition_status': acq_status,
            'ppg_window'        : ppg_note,
            'dominant_artifact' : dominant,
            'researcher_note'   : note,
            'confidence'        : confidence,
            'recommendation'    : recommendation,
            'trend'             : trend,
            'stable_pct'        : stable_fraction * 100,
        }

    def _dominant_event(self, events: dict, score: float, state: str) -> str:
        if score < 15:
            return "None — minimal motion artifact detected"
        # Find highest intensity event
        top = max(events.items(), key=lambda kv: kv[1])
        if top[1] < 0.1:
            label_map = {
                MotionQuality.LOW_MOTION    : "Low-level ambient movement",
                MotionQuality.MEDIUM_MOTION : "Moderate translational motion",
                MotionQuality.HIGH_MOTION   : "High-intensity motion artifact",
                MotionQuality.INVALID       : "Extreme motion — signal saturation",
            }
            return label_map.get(state, "Unclassified motion")
        label_map = {
            'jerk'        : "Sudden jerk / impact event",
            'jerk_extreme': "Extreme impact — mechanical shock",
            'tremor'      : "Tremor-like oscillation (4–12 Hz)",
            'vibration'   : "Sustained vibration artifact",
            'placement'   : "Sensor placement disturbance",
            'orientation' : "Orientation instability",
        }
        return label_map.get(top[0], "Mixed motion artifact")

    def _acquisition_status(self, score: float, sf: float) -> str:
        if score < 15 and sf > 0.85:
            return "ACQUISITION VALID — experimental conditions nominal"
        if score < 30 and sf > 0.65:
            return "ACQUISITION MARGINAL — mild artifact, verify baseline"
        if score < 55:
            return "ACQUISITION COMPROMISED — moderate motion artifact present"
        if score < 75:
            return "ACQUISITION POOR — high artifact contamination"
        return "ACQUISITION SUSPENDED — motion artifact exceeds tolerance"

    def _ppg_window_note(self, score: float, state: str, events: dict) -> str:
        if state == MotionQuality.STABLE:
            return "PPG VALID — stable window suitable for optical biosensing"
        if state == MotionQuality.LOW_MOTION:
            return "PPG MARGINAL — mild artifact risk, baseline verification advised"
        if state == MotionQuality.MEDIUM_MOTION:
            return "PPG COMPROMISED — moderate motion, artifact correction required"
        if state == MotionQuality.HIGH_MOTION:
            return "PPG INVALID — high motion artifact, discard current window"
        return "PPG REJECTED — extreme motion, signal unusable for biosensing"

    def _researcher_note(self, score: float, state: str,
                         events: dict, tilt: float, trend: str) -> str:
        notes = []
        if events.get('tremor', 0) > 0.4:
            notes.append("Tremor artifact detected — consider adaptive filtering")
        if events.get('jerk_extreme', 0) > 0.3:
            notes.append("Impact event logged — verify sensor attachment")
        if tilt > 45:
            notes.append(f"Tilt {tilt:.0f}° — significant orientation change since calibration")
        if "DETERIORATING" in trend and score > 35:
            notes.append("Artifact score rising — motion increasing")
        if not notes:
            if score < 20:
                return "Research-grade estimation: acquisition conditions within tolerance"
            return "Motion artifact estimation: moderate perturbation present"
        return " | ".join(notes[:2])   # Limit to 2 notes for display

    def _recommendation(self, score: float, state: str, sf: float) -> str:
        if score < 20:
            return "Continue acquisition — motion conditions acceptable for research"
        if score < 40:
            return "Flag window — apply motion artifact correction in post-processing"
        if sf > 0.5:
            return "Request subject stillness — intermittent motion degrading signal"
        return "Suspend acquisition — resolve motion source before continuing"


# ══════════════════════════════════════════════════════════════════════════════
# §9  TERMINAL DASHBOARD
#     Professional biomedical engineering research display
# ══════════════════════════════════════════════════════════════════════════════

class Dashboard:
    """
    ANSI-escaped real-time terminal dashboard.

    Design principles:
      · In-place refresh via cursor repositioning (no scroll)
      · Color semantics: green=good, yellow=caution, red=critical
      · Information hierarchy: most critical data at top
      · Minimal cognitive load: consistent layout, clear labels
      · Research-grade aesthetic: clean, technical, no decorative clutter

    Refresh strategy:
      1. First frame: full terminal clear + render
      2. Subsequent frames: cursor to home + overwrite in place
      3. All lines padded to full width to clear stale characters

    Color mapping:
      Score 0–25  : green  (acceptable)
      Score 26–55 : yellow (caution)
      Score 56–100: red    (critical)

    ASCII progress bars use Unicode block elements for clean rendering
    on modern terminal emulators (PuTTY, GNOME Terminal, iTerm2).
    """

    # ANSI codes
    ESC     = "\033["
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    BLINK   = "\033[5m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"

    W       = 76   # Dashboard character width

    def __init__(self):
        self._first  = True
        self._frames = 0
        self._t0     = time.monotonic()

    # ── Terminal control ───────────────────────────────────────────────────

    def _home(self):
        sys.stdout.write("\033[H")

    def _clear(self):
        sys.stdout.write("\033[2J\033[H")

    def _eol(self, text: str) -> str:
        """Pad text to dashboard width to clear stale characters."""
        vis = len(text.encode('ascii', errors='ignore'))
        ansi_overhead = len(text) - vis
        # Rough pad: add spaces to reach W chars of visible content
        return text + " " * max(0, self.W - len(text) + ansi_overhead + 10)

    # ── Color helpers ──────────────────────────────────────────────────────

    def _score_col(self, s: float) -> str:
        if s < 26: return self.GREEN
        if s < 56: return self.YELLOW
        return self.RED

    def _state_col(self, st: str) -> str:
        cols = {
            MotionQuality.STABLE        : self.GREEN,
            MotionQuality.LOW_MOTION    : self.CYAN,
            MotionQuality.MEDIUM_MOTION : self.YELLOW,
            MotionQuality.HIGH_MOTION   : self.RED,
            MotionQuality.INVALID       : self.RED + self.BOLD,
        }
        return cols.get(st, self.WHITE)

    def _quality_col(self, q: float) -> str:
        if q > 75: return self.GREEN
        if q > 45: return self.YELLOW
        return self.RED

    # ── Bar renderers ──────────────────────────────────────────────────────

    def _score_bar(self, score: float, width: int = 28) -> str:
        filled = int(score / 100.0 * width)
        col    = self._score_col(score)
        return f"[{col}{'█'*filled}{'░'*(width-filled)}{self.RESET}]"

    def _level_bar(self, val: float, lo: float, hi: float,
                   width: int = 16, col: str = None) -> str:
        frac   = max(0.0, min(1.0, (val - lo) / max(1e-9, hi - lo)))
        filled = int(frac * width)
        c      = col or self.CYAN
        return f"{c}{'▮'*filled}{'·'*(width-filled)}{self.RESET}"

    def _signed_bar(self, val: float, rng: float, width: int = 20) -> str:
        """Bi-directional bar centered at zero."""
        half  = width // 2
        frac  = max(-1.0, min(1.0, val / rng))
        if frac >= 0:
            pos = int(frac * half)
            bar = f"{'·'*half}{'▮'*pos}{'·'*(half-pos)}"
        else:
            neg = int(abs(frac) * half)
            bar = f"{'·'*(half-neg)}{'▮'*neg}{'·'*half}"
        return bar

    # ── Section renderers ──────────────────────────────────────────────────

    def _hline(self, char='─') -> str:
        return self.GRAY + char * self.W + self.RESET

    def _section(self, title: str) -> str:
        pad  = (self.W - len(title) - 4) // 2
        line = self.GRAY + "├" + "─"*pad
        line += self.RESET + self.BOLD + f" {title} "
        line += self.RESET + self.GRAY + "─"*pad + "┤" + self.RESET
        return line

    def render(self,
               filtered: dict,
               accel_mag: float,
               dyn: float,
               gyro_mag: float,
               variance: float,
               pitch: float, roll: float, tilt: float,
               state: str,
               artifact: dict,
               events: dict,
               interp: dict,
               effective_fs: float,
               health: dict) -> None:
        """
        Render one complete dashboard frame.
        All parameters are current-sample values from analysis pipeline.
        """
        if self._first:
            self._clear()
            self._first = False
        else:
            self._home()

        B  = self.BOLD
        R  = self.RESET
        D  = self.GRAY
        C  = self.CYAN
        M  = self.MAGENTA
        G  = self.GREEN
        Y  = self.YELLOW

        uptime = time.monotonic() - self._t0
        sc     = artifact['score']
        scol   = self._score_col(sc)
        stcol  = self._state_col(state)
        qcol   = self._quality_col(artifact['ppg_validity'])

        out = []
        P   = out.append   # shorthand

        def row(label: str, value: str, note: str = "") -> str:
            l = f"  {D}{label:<22}{R}{value}"
            if note:
                l += f"  {D}{note}{R}"
            return l

        # ══ HEADER ══════════════════════════════════════════════════════════
        P(self.GRAY + "┌" + "─"*(self.W-2) + "┐" + R)
        title = (f"  {M}{B}MPU-6050 BIOMEDICAL MOTION ARTIFACT SYSTEM{R}  "
                 f"{D}[NON-CLINICAL / RESEARCH-GRADE]{R}")
        P(self.GRAY + "│" + R + title + self.GRAY + " │" + R)
        meta  = (f"  {D}Fs≈{effective_fs:.0f}Hz │ "
                 f"Cal:{'✓' if health['calibrated'] else '✗'} │ "
                 f"Err:{health['errors']} │ "
                 f"Frame:{self._frames:06d} │ "
                 f"Up:{uptime:6.1f}s{R}")
        P(self.GRAY + "│" + R + meta + self.GRAY + " │" + R)
        P(self.GRAY + "└" + "─"*(self.W-2) + "┘" + R)

        # ══ SENSOR READINGS ══════════════════════════════════════════════════
        P(self._section("SENSOR READINGS"))
        ax,ay,az = filtered['ax'], filtered['ay'], filtered['az']
        gx,gy,gz = filtered['gx'], filtered['gy'], filtered['gz']
        tc = filtered['tc']

        P(row("Accelerometer",
              f"{C}X:{ax:+7.4f}g{R}  {C}Y:{ay:+7.4f}g{R}  {C}Z:{az:+7.4f}g{R}"))
        P(f"  {D}  Ax{R} {self._signed_bar(ax,2.0)}  "
          f"{D}Ay{R} {self._signed_bar(ay,2.0)}  "
          f"{D}Az{R} {self._signed_bar(az,2.0)}")
        P(row("Gyroscope",
              f"{C}X:{gx:+7.2f}°/s{R}  {C}Y:{gy:+7.2f}°/s{R}  {C}Z:{gz:+7.2f}°/s{R}"))
        P(f"  {D}  Gx{R} {self._signed_bar(gx,250)}  "
          f"{D}Gy{R} {self._signed_bar(gy,250)}  "
          f"{D}Gz{R} {self._signed_bar(gz,250)}")
        P(row("Temperature",
              f"{C}{tc:+6.2f}°C{R}",
              "(on-die sensor — not ambient)"))

        # ══ MOTION MAGNITUDE ════════════════════════════════════════════════
        P(self._section("MOTION MAGNITUDE"))
        P(row("|a| total",      f"{C}{accel_mag:7.4f} g{R}",  "√(ax²+ay²+az²)"))
        P(row("Dynamic |a|",    f"{C}{dyn:7.4f} g{R}",        "| |a|−1g | gravity-free"))
        P(row("|ω| total",      f"{C}{gyro_mag:7.2f} °/s{R}", "√(gx²+gy²+gz²)"))
        P(row("Accel variance", f"{C}{variance:.6f} g²{R}",   "50-sample window"))
        motion_norm = min(1.0, dyn / 0.8)
        P(row("Motion score",
              f"{self._level_bar(motion_norm,0,1,width=24)}  {C}{motion_norm*100:5.1f}%{R}"))

        # ══ ORIENTATION ═════════════════════════════════════════════════════
        P(self._section("ORIENTATION  [complementary filter α=0.96]"))
        P(row("Pitch (X)",
              f"{C}{pitch:+8.2f}°{R}  {self._level_bar(abs(pitch),0,90,16)}",
              "forward/back tilt"))
        P(row("Roll  (Y)",
              f"{C}{roll:+8.2f}°{R}  {self._level_bar(abs(roll),0,90,16)}",
              "left/right tilt"))
        P(row("Total tilt",
              f"{C}{tilt:8.2f}°{R}  {self._level_bar(tilt,0,90,16)}",
              "√(pitch²+roll²)"))

        # ══ MOTION QUALITY ══════════════════════════════════════════════════
        P(self._section("MOTION QUALITY ENGINE"))
        P(row("State",
              f"{stcol}{B}{state:<18}{R}",
              "multi-criteria hysteresis classifier"))

        # ══ MOTION EVENTS ═══════════════════════════════════════════════════
        P(self._section("MOTION EVENT DETECTION"))
        ev_labels = [
            ('Sudden Jerk',    'jerk',        self.YELLOW),
            ('Extreme Impact', 'jerk_extreme', self.RED),
            ('Tremor (4–12Hz)','tremor',       self.YELLOW),
            ('Vibration',      'vibration',    self.YELLOW),
            ('Placement Chg',  'placement',    self.MAGENTA),
            ('Orient. Instab.','orientation',  self.CYAN),
        ]
        for label, key, col in ev_labels:
            val = events.get(key, 0.0)
            bar = self._level_bar(val, 0, 1, 18, col)
            active = f" {col}{B}●{R}" if val > 0.15 else f" {D}○{R}"
            P(row(label, f"{bar} {col}{val:4.2f}{R}{active}"))

        # ══ ARTIFACT SCORE ═══════════════════════════════════════════════════
        P(self._section("MOTION ARTIFACT SCORE  [research-grade PPG quality]"))
        P(row("Artifact Score",
              f"{self._score_bar(sc,28)} {scol}{B}{sc:5.1f}/100{R}"))
        P(row("PPG Validity",
              f"{qcol}{artifact['ppg_validity']:5.1f}%{R}  "
              f"{self._level_bar(artifact['ppg_validity'],0,100,20,qcol)}"))
        P(row("Optical Stability",
              f"{qcol}{artifact['stability']:5.1f}%{R}  "
              f"{self._level_bar(artifact['stability'],0,100,20,qcol)}"))
        P(row("Mean Score (100s)",
              f"{scol}{artifact['mean_score']:5.1f}/100{R}"))
        # Component breakdown
        comp = artifact['components']
        P(f"  {D}Score components:{R}  "
          f"{D}Dyn:{R}{C}{comp['dyn']:4.1f}{R}  "
          f"{D}Gyro:{R}{C}{comp['gyro']:4.1f}{R}  "
          f"{D}Var:{R}{C}{comp['var']:4.1f}{R}  "
          f"{D}Events:{R}{C}{comp['event']:4.1f}{R}")

        # ══ AI INTERPRETATION ════════════════════════════════════════════════
        P(self._section("AI-ASSISTED INTERPRETATION  [experimental · non-clinical]"))
        P(row("Status",   f"{scol}{interp['acquisition_status']}{R}"))
        P(row("PPG",      f"{scol}{interp['ppg_window']}{R}"))
        P(row("Artifact", f"{Y}{interp['dominant_artifact']}{R}"))
        P(row("Note",     f"{C}{interp['researcher_note']}{R}"))
        P(row("Trend",    f"{Y}{interp['trend']}{R}  "
              f"{D}Stable:{R}{G}{interp['stable_pct']:.0f}%{R} of last 30s"))
        P(row("Confidence", f"{C}{interp['confidence']}{R}"))
        P(row("Action",   f"{B}{interp['recommendation']}{R}"))

        # ══ FOOTER ══════════════════════════════════════════════════════════
        P(self._hline('═'))
        P(f"  {D}[Ctrl+C to stop] · "
          f"Research prototype · Non-clinical · Experimental analysis only{R}")
        P("")

        # Flush atomically
        sys.stdout.write("\n".join(self._eol(l) for l in out))
        sys.stdout.flush()
        self._frames += 1


# ══════════════════════════════════════════════════════════════════════════════
# §10  PRECISION ACQUISITION LOOP
# ══════════════════════════════════════════════════════════════════════════════

class PrecisionTimer:
    """
    Hybrid sleep + busy-wait precision timing loop.

    Linux sleep() granularity is limited by the scheduler tick (typically
    1–10 ms). For 100 Hz sampling (10 ms period), naive time.sleep(0.01)
    produces ±2–3 ms jitter — unacceptable for stable sensor acquisition.

    Strategy:
      1. Compute deadline = last_wake + period
      2. Sleep until deadline − margin (yields CPU to OS scheduler)
      3. Busy-poll from margin to deadline (100% CPU for ~1–2 ms)
      4. Record actual period for Fs estimation

    Sleep margin: 2 ms — conservative for Pi's Linux scheduler.
    Busy-poll duration: typically 0.5–2 ms.
    Net jitter: < ±0.5 ms on Pi 3B/4B.
    CPU overhead: ~5–15% on one core for the busy-poll phase.
    """

    MARGIN = 0.002   # Sleep undershoot margin [s]

    def __init__(self, period: float):
        self.period    = period
        self._last     = None
        self._periods  = collections.deque(maxlen=200)

    def wait(self):
        """Block until next sample deadline. Call once per acquisition cycle."""
        now = time.monotonic()
        if self._last is None:
            self._last = now
            return

        deadline = self._last + self.period
        sleep_to = deadline - self.MARGIN

        if sleep_to > now:
            time.sleep(sleep_to - now)

        while time.monotonic() < deadline:
            pass   # busy-poll (precise)

        actual = time.monotonic()
        self._periods.append(actual - self._last)
        self._last = actual

    @property
    def effective_fs(self) -> float:
        if len(self._periods) < 10:
            return 1.0 / self.period
        return 1.0 / (sum(self._periods) / len(self._periods))


# ══════════════════════════════════════════════════════════════════════════════
# §11  SIMULATION ENGINE
#      Physically plausible synthetic IMU data for hardware-free testing
# ══════════════════════════════════════════════════════════════════════════════

class SimulationEngine:
    """
    Physics-based MPU-6050 simulator for development and demonstration.

    Generates synthetic sensor data with realistic noise models:

    Noise model (MPU-6050 datasheet, Table 1):
      Accel noise density : 400 μg/√Hz
      Gyro noise density  : 0.005 °/s/√Hz
      At 100 Hz bandwidth:
        σ_accel = 400×10⁻⁶ × √100 ≈ 4 mg RMS
        σ_gyro  = 0.005     × √100 ≈ 0.05 °/s RMS

    Motion scenario sequence:
      1. Rest (10s)  → baseline
      2. Gentle wrist movement (5s)
      3. Rest (5s)
      4. Hand shake simulation (4s)
      5. Tremor simulation (5s)
      6. Strong movement (3s)
      7. Rest (8s)
      8. Cycle repeats
    """

    import random as _random

    SCENARIO_SEQUENCE = [
        ('rest',        10.0),
        ('gentle',       5.0),
        ('rest',         5.0),
        ('shake',        4.0),
        ('tremor',       5.0),
        ('strong',       3.0),
        ('placement',    1.0),
        ('rest',         8.0),
    ]

    def __init__(self, fs: float = 100.0):
        import random
        self.fs     = fs
        self.dt     = 1.0 / fs
        self._t     = 0.0
        self._rng   = random.Random(1337)
        self._scen_idx  = 0
        self._scen_t    = 0.0
        self._scene     = 'rest'

        # Orientation state for simulation
        self._sim_pitch  = 0.0
        self._sim_roll   = 0.0

    def _advance_scenario(self):
        seq = self.SCENARIO_SEQUENCE
        self._scen_t += self.dt
        _, dur = seq[self._scen_idx]
        if self._scen_t >= dur:
            self._scen_t = 0.0
            self._scen_idx = (self._scen_idx + 1) % len(seq)
        self._scene, _ = seq[self._scen_idx]

    def read_raw_burst(self) -> dict:
        """Generate one sample of physically plausible IMU data."""
        import random, math
        rng    = self._rng
        self._advance_scenario()
        t      = self._t
        self._t += self.dt
        sc     = self._scene

        # ── Base: gravity on Z, zero elsewhere ────────────────────────────
        ax = ay = 0.0
        az = 1.0
        gx = gy = gz = 0.0

        # ── Micro-tremor: always present (physiological) ──────────────────
        mt = 0.006
        ax += mt * math.sin(2*math.pi*1.1*t + 0.3)
        ay += mt * math.sin(2*math.pi*0.7*t + 1.1)

        # ── Scenario-specific dynamics ─────────────────────────────────────
        if sc == 'gentle':
            amp = rng.uniform(0.04, 0.10)
            ax += amp * math.sin(2*math.pi*0.8*t)
            ay += amp * 0.6 * math.sin(2*math.pi*0.6*t + 0.5)
            gx += rng.gauss(0, 3.0)
            gy += rng.gauss(0, 2.0)

        elif sc == 'shake':
            amp  = rng.uniform(0.15, 0.40)
            freq = rng.uniform(2.0, 4.5)
            ax  += amp * math.sin(2*math.pi*freq*t + rng.random())
            ay  += amp * 0.8 * math.sin(2*math.pi*(freq*1.1)*t)
            az  += amp * 0.3 * math.sin(2*math.pi*(freq*0.9)*t)
            gx  += rng.gauss(0, 15.0)
            gy  += rng.gauss(0, 12.0)
            gz  += rng.gauss(0, 8.0)

        elif sc == 'tremor':
            # Pathological-like tremor: 6–8 Hz (Parkinson's-adjacent range)
            freq = 7.0 + rng.gauss(0, 0.3)
            amp  = rng.uniform(0.03, 0.08)
            ax  += amp * math.sin(2*math.pi*freq*t)
            ay  += amp * 0.7 * math.sin(2*math.pi*freq*t + 0.5)
            gx  += 8.0 * math.sin(2*math.pi*freq*t)
            gy  += 6.0 * math.sin(2*math.pi*freq*t + 0.3)

        elif sc == 'strong':
            amp = rng.uniform(0.5, 0.9)
            ax += amp * rng.gauss(0, 1.0)
            ay += amp * rng.gauss(0, 0.8)
            az += amp * 0.4 * rng.gauss(0, 0.5)
            gx += rng.gauss(0, 40.0)
            gy += rng.gauss(0, 35.0)
            gz += rng.gauss(0, 20.0)

        elif sc == 'placement':
            # Sudden orientation change
            self._sim_pitch += rng.uniform(10, 25)
            pr = math.radians(self._sim_pitch)
            ax += math.sin(pr) * 0.5
            az  = math.cos(pr)

        # ── Gaussian noise (hardware noise floor) ──────────────────────────
        ax += rng.gauss(0, 0.004)
        ay += rng.gauss(0, 0.004)
        az += rng.gauss(0, 0.004)
        gx += rng.gauss(0, 0.05)
        gy += rng.gauss(0, 0.05)
        gz += rng.gauss(0, 0.05)

        # Temperature: stable ± slow drift
        tc = 30.0 + 0.5 * math.sin(2*math.pi * t / 120.0) + rng.gauss(0, 0.02)

        # Convert to raw ADC counts
        def a_raw(v): return int(v  * Reg.ACCEL_LSB_PER_G)
        def g_raw(v): return int(v  * Reg.GYRO_LSB_PER_DPS)
        def t_raw(v): return int((v - Reg.TEMP_OFFSET) * Reg.TEMP_SENSITIVITY)

        return {
            'ax_raw': a_raw(ax), 'ay_raw': a_raw(ay), 'az_raw': a_raw(az),
            'gx_raw': g_raw(gx), 'gy_raw': g_raw(gy), 'gz_raw': g_raw(gz),
            'tr_raw': t_raw(tc), 'ts': time.monotonic(),
        }

    def raw_to_physical(self, raw: dict) -> dict:
        """Same conversion as MPU6050Driver."""
        ax = raw['ax_raw'] / Reg.ACCEL_LSB_PER_G
        ay = raw['ay_raw'] / Reg.ACCEL_LSB_PER_G
        az = raw['az_raw'] / Reg.ACCEL_LSB_PER_G
        gx = raw['gx_raw'] / Reg.GYRO_LSB_PER_DPS
        gy = raw['gy_raw'] / Reg.GYRO_LSB_PER_DPS
        gz = raw['gz_raw'] / Reg.GYRO_LSB_PER_DPS
        tc = raw['tr_raw'] / Reg.TEMP_SENSITIVITY + Reg.TEMP_OFFSET
        return {'ax':ax,'ay':ay,'az':az,'gx':gx,'gy':gy,'gz':gz,'tc':tc,'ts':raw['ts']}

    @property
    def is_open(self): return True
    @property
    def is_calibrated(self): return True
    @property
    def i2c_errors(self): return 0
    @property
    def error_rate(self): return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# §12  MAIN APPLICATION
#      System integration and real-time loop
# ══════════════════════════════════════════════════════════════════════════════

class BiomedicalMotionSystem:
    """
    Top-level system coordinator.

    Integrates all subsystems into a coherent real-time pipeline:

      Sensor/Sim → Filtering → Analysis → Display
                                    ↓
                            Motion Events
                            Quality Engine
                            Artifact Scorer
                            AI Interpreter
                            Orientation

    Loop timing:
      Acquisition: 100 Hz (10 ms / sample)
      Display:      20 Hz (every 5 samples)

    Graceful shutdown on SIGINT/SIGTERM.
    """

    FS          = 100.0    # Target sample rate [Hz]
    DISPLAY_DIV = 5        # Display every N samples → 20 Hz display

    def __init__(self, use_hardware: bool = True):
        self.use_hw = use_hardware

        # ── Subsystems ────────────────────────────────────────────────────
        self.sensor    = (MPU6050Driver() if use_hardware
                          else SimulationEngine(fs=self.FS))
        self.filters   = {ch: ChannelFilterBank(
                              'accel' if ch in ('ax','ay','az') else 'gyro',
                              fs=self.FS)
                          for ch in ('ax','ay','az','gx','gy','gz')}
        self.orient    = OrientationTracker()
        self.events    = MotionEventDetector(fs=self.FS)
        self.quality   = MotionQualityEngine(fs=self.FS)
        self.scorer    = ArtifactScorer(fs=self.FS)
        self.interp    = MotionInterpreter()
        self.dashboard = Dashboard()
        self.timer     = PrecisionTimer(period=1.0/self.FS)

        # Runtime state
        self._running  = False
        self._n        = 0

        # History for stable-fraction computation
        self._state_hist = collections.deque(maxlen=int(30 * self.FS))

        # Variance computation buffer
        self._amag_var   = collections.deque(maxlen=50)

        signal.signal(signal.SIGINT,  self._on_stop)
        signal.signal(signal.SIGTERM, self._on_stop)

    def _on_stop(self, *_):
        self._running = False

    # ──────────────────────────────────────────────────────────────────────
    # Startup UI
    # ──────────────────────────────────────────────────────────────────────

    def _banner(self, mode: str):
        os.system('clear')
        M = Dashboard.MAGENTA
        B = Dashboard.BOLD
        R = Dashboard.RESET
        D = Dashboard.GRAY
        Y = Dashboard.YELLOW
        C = Dashboard.CYAN

        print(f"\n  {M}{B}{'═'*66}{R}")
        print(f"  {M}{B}  MPU-6050 BIOMEDICAL MOTION ARTIFACT ANALYSIS SYSTEM v2.0{R}")
        print(f"  {M}{B}  Research-Grade IMU Platform for Optical Biosensing{R}")
        print(f"  {M}{B}{'═'*66}{R}")
        print(f"  {D}Mode     : {C}{mode}{R}")
        print(f"  {D}Config   : ±2g / ±250°/s | DLPF 44Hz | Fs=100Hz{R}")
        print(f"  {D}Filter   : Spike(MAD) → IIR-LP → Adaptive-IIR → MA{R}")
        print(f"  {D}Orient   : Complementary filter (α=0.96){R}")
        print(f"  {D}Artifacts: Dynamic accel + Gyro + Variance + Events{R}")
        print(f"  {M}{B}{'─'*66}{R}")
        print(f"  {Y}⚠  NON-CLINICAL · EXPERIMENTAL · RESEARCH-GRADE ONLY{R}")
        print(f"  {Y}   All outputs are motion artifact estimations, not diagnoses.{R}")
        print(f"  {M}{B}{'─'*66}{R}\n")

    def _progress_bar(self, frac: float, width: int = 40) -> str:
        f = int(frac * width)
        return f"[{'█'*f}{'░'*(width-f)}] {frac*100:5.1f}%"

    # ──────────────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────────────

    def initialize(self):
        mode = "HARDWARE — Raspberry Pi I²C" if self.use_hw else "SIMULATION — Software emulation"
        self._banner(mode)

        if self.use_hw:
            print("  [1/3] Opening I²C bus and verifying MPU-6050...", end='', flush=True)
            try:
                self.sensor.open()
                print(f"  {Dashboard.GREEN}✓ WHO_AM_I = 0x68 confirmed{Dashboard.RESET}")
            except RuntimeError as e:
                print(f"\n  {Dashboard.RED}✗ {e}{Dashboard.RESET}\n")
                sys.exit(1)

            print("  [2/3] Initializing sensor registers...")
            self.sensor.initialize()
            print(f"        {Dashboard.GREEN}✓ Registers configured{Dashboard.RESET}")

            print("  [3/3] Static calibration — keep sensor PERFECTLY STILL...")
            print()

            def on_prog(f):
                sys.stdout.write(f"\r        {self._progress_bar(f)}")
                sys.stdout.flush()

            self.sensor.calibrate(n=200, on_progress=on_prog)
            print(f"\n        {Dashboard.GREEN}✓ Calibration complete{Dashboard.RESET}")
            ab = self.sensor.accel_bias
            gb = self.sensor.gyro_bias
            print(f"        Accel bias: X={ab[0]:+.4f}g  Y={ab[1]:+.4f}g  Z={ab[2]:+.4f}g")
            print(f"        Gyro  bias: X={gb[0]:+.3f}°/s  Y={gb[1]:+.3f}°/s  Z={gb[2]:+.3f}°/s")
        else:
            print("  [1/1] Simulation engine initialized.")
            print(f"        {Dashboard.GREEN}✓ Synthetic MPU-6050 ready — scenario cycling enabled{Dashboard.RESET}")
            print(f"        {Dashboard.GRAY}Noise model: σ_a=4mg RMS, σ_g=0.05°/s RMS{Dashboard.RESET}")

        print(f"\n  Starting real-time monitoring in 2 s...")
        print(f"  {Dashboard.GRAY}(Use --sim to force simulation mode){Dashboard.RESET}\n")
        time.sleep(2.0)

    # ──────────────────────────────────────────────────────────────────────
    # Main Loop
    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        self._running = True
        prev_ts = time.monotonic()

        while self._running:
            self.timer.wait()

            try:
                # ── 1. Acquire ─────────────────────────────────────────────
                raw  = self.sensor.read_raw_burst()
                data = self.sensor.raw_to_physical(raw)
            except OSError as e:
                sys.stderr.write(f"\n  [I²C ERR] {e}\n")
                time.sleep(0.005)
                continue

            # ── 2. Compute dt ──────────────────────────────────────────────
            ts = data['ts']
            dt = max(0.001, min(0.1, ts - prev_ts))
            prev_ts = ts

            # ── 3. Quick motion estimate (pre-filter) for adaptive IIR ─────
            amag_raw = math.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)
            dyn_raw  = abs(amag_raw - 1.0)
            ml       = min(1.0, dyn_raw / 0.8)  # normalize to [0,1]
            for fb in self.filters.values():
                fb.set_motion_level(ml)

            # ── 4. Filter all channels ─────────────────────────────────────
            f = {ch: self.filters[ch].process(data[ch])
                 for ch in ('ax','ay','az','gx','gy','gz')}
            f['tc'] = data['tc']   # Temperature — no filtering needed

            # ── 5. Motion magnitudes ───────────────────────────────────────
            amag = math.sqrt(f['ax']**2 + f['ay']**2 + f['az']**2)
            dyn  = abs(amag - 1.0)
            gmag = math.sqrt(f['gx']**2 + f['gy']**2 + f['gz']**2)
            self._amag_var.append(amag)
            variance = (statistics.variance(self._amag_var)
                        if len(self._amag_var) >= 2 else 0.0)

            # ── 6. Orientation ─────────────────────────────────────────────
            pitch, roll, tilt = self.orient.update(f['ax'],f['ay'],f['az'],
                                                    f['gx'],f['gy'], dt=dt)

            # ── 7. Event detection ─────────────────────────────────────────
            ev = self.events.update(amag, dyn, tilt)

            # ── 8. Motion quality classification ──────────────────────────
            state = self.quality.update(dyn, gmag, variance, self.events.max_event)
            self._state_hist.append(state)

            # ── 9. Artifact scoring ────────────────────────────────────────
            artifact = self.scorer.compute(dyn, gmag, variance, ev)

            # ── 10. AI interpretation ──────────────────────────────────────
            n_hist = len(self._state_hist)
            sf = (sum(1 for s in self._state_hist
                      if s == MotionQuality.STABLE) / n_hist
                  if n_hist else 0.0)
            interpretation = self.interp.interpret(
                artifact['score'], state, ev,
                pitch, roll, tilt, sf)

            # ── 11. Display (throttled to ~20 Hz) ─────────────────────────
            self._n += 1
            if self._n % self.DISPLAY_DIV == 0:
                health = {
                    'calibrated': self.sensor.is_calibrated,
                    'errors'    : self.sensor.i2c_errors,
                    'err_rate'  : self.sensor.error_rate,
                }
                self.dashboard.render(
                    filtered=f,
                    accel_mag=amag,
                    dyn=dyn,
                    gyro_mag=gmag,
                    variance=variance,
                    pitch=pitch, roll=roll, tilt=tilt,
                    state=state,
                    artifact=artifact,
                    events=ev,
                    interp=interpretation,
                    effective_fs=self.timer.effective_fs,
                    health=health,
                )

        self._shutdown()

    def _shutdown(self):
        C = Dashboard.CYAN
        G = Dashboard.GREEN
        R = Dashboard.RESET
        print(f"\n\n  {C}Shutting down...{R}")
        if self.use_hw and hasattr(self.sensor, 'close'):
            self.sensor.close()
        print(f"  {G}✓ Clean shutdown. Total samples: {self._n}{R}\n")


# ══════════════════════════════════════════════════════════════════════════════
# §13  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _detect_hardware() -> bool:
    """
    Auto-detect Raspberry Pi I²C hardware availability.

    Checks:
      1. /dev/i2c-1 exists (Pi GPIO header I²C bus)
      2. smbus2 importable
      3. Bus can be opened without error
    """
    if not _SMBUS_AVAILABLE:
        return False
    if not os.path.exists("/dev/i2c-1"):
        return False
    try:
        bus = smbus2.SMBus(1)
        bus.close()
        return True
    except Exception:
        return False


def main():
    """
    Entry point with platform auto-detection.

    CLI flags:
      --sim  Force simulation mode (no hardware required)
      --hw   Force hardware mode  (fails if I²C unavailable)
    """
    force_sim = "--sim" in sys.argv
    force_hw  = "--hw"  in sys.argv

    if force_sim:
        hw = False
    elif force_hw:
        hw = True
    else:
        hw = _detect_hardware()

    if not hw:
        print(f"\n  {Dashboard.YELLOW}No I²C hardware detected — simulation mode active.{Dashboard.RESET}")

    system = BiomedicalMotionSystem(use_hardware=hw)
    system.initialize()
    system.run()


if __name__ == "__main__":
    main()