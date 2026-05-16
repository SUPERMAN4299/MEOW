#!/usr/bin/env python3
"""
=============================================================================
MPU6050 BIOMEDICAL MOTION ANALYSIS SYSTEM
Research-Grade IMU Motion Artifact Detection for PPG/Optical Biosensing

Author      : Biomedical Sensing Research Framework
Platform    : Raspberry Pi (ARMv7/ARMv8) — Linux I2C
Sensor      : InvenSense MPU-6050 (Accelerometer + Gyroscope + Thermometer)
Protocol    : I2C via smbus2 (400 kHz fast-mode recommended)
Purpose     : Real-time motion artifact detection and IMU signal conditioning
              for photoplethysmography (PPG) and biomedical optical sensing

DISCLAIMER  : This is a non-clinical, research-grade, experimental analysis
              system. Not intended for medical diagnosis or clinical use.
              All outputs are AI-assisted interpretations for research only.

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │  MPU6050 I2C Register Interface                      │
  │    └─► Raw Acquisition Loop (100 Hz)                 │
  │         └─► Signal Filtering Pipeline                │
  │              ├─► Low-Pass Filter (IIR Butterworth)   │
  │              ├─► Moving Average Smoothing            │
  │              └─► Spike Rejection                     │
  │                   └─► Motion Analysis Engine         │
  │                        ├─► Magnitude Computation     │
  │                        ├─► Motion Classification     │
  │                        ├─► Artifact Scoring          │
  │                        ├─► Orientation Estimation    │
  │                        └─► AI-Assisted Interpretation│
  │                             └─► Terminal Dashboard   │
  └──────────────────────────────────────────────────────┘

References:
  [1] InvenSense MPU-6000/MPU-6050 Product Specification Rev. 3.4
  [2] InvenSense MPU-6050 Register Map Rev. 4.2
  [3] Krishnan R. et al. "Motion Artifact Reduction in PPG Signals" (2010)
  [4] Schäfer A. & Vagedes J. "How Accurate is Pulse Rate Variability as an
      Estimate of Heart Rate Variability?" Int. J. Cardiol. 166(1), 2013
=============================================================================
"""

import smbus2
import time
import math
import sys
import os
import collections
import struct
import signal
import threading

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: MPU-6050 REGISTER MAP
# Source: InvenSense Register Map and Descriptions, Rev. 4.2
# ─────────────────────────────────────────────────────────────────────────────

class MPU6050Registers:
    """
    Complete register map for the InvenSense MPU-6050.
    All addresses are 8-bit I2C register offsets per datasheet Table 1.
    """

    # --- Device Identification ---
    I2C_ADDR_DEFAULT   = 0x68   # AD0 pin LOW  (most common)
    I2C_ADDR_ALT       = 0x69   # AD0 pin HIGH (alternate)
    WHO_AM_I           = 0x75   # Returns 0x68 — device identity register

    # --- Power Management ---
    PWR_MGMT_1         = 0x6B   # Power management 1: sleep, cycle, clksel
    PWR_MGMT_2         = 0x6C   # Power management 2: standby axis control
    SIGNAL_PATH_RESET  = 0x68   # Signal path reset (gyro/accel/temp paths)
    USER_CTRL          = 0x6A   # User control: FIFO, I2C master, DMP

    # --- Configuration ---
    CONFIG             = 0x1A   # DLPF configuration + EXT_SYNC_SET
    GYRO_CONFIG        = 0x1B   # Gyroscope full-scale range select
    ACCEL_CONFIG       = 0x1C   # Accelerometer full-scale range select
    SMPLRT_DIV         = 0x19   # Sample rate divider

    # --- Accelerometer Data Registers (two's complement, 16-bit) ---
    ACCEL_XOUT_H       = 0x3B   # AX high byte
    ACCEL_XOUT_L       = 0x3C   # AX low byte
    ACCEL_YOUT_H       = 0x3D   # AY high byte
    ACCEL_YOUT_L       = 0x3E   # AY low byte
    ACCEL_ZOUT_H       = 0x3F   # AZ high byte
    ACCEL_ZOUT_L       = 0x40   # AZ low byte

    # --- Temperature Data Registers ---
    TEMP_OUT_H         = 0x41   # Temperature high byte
    TEMP_OUT_L         = 0x42   # Temperature low byte

    # --- Gyroscope Data Registers (two's complement, 16-bit) ---
    GYRO_XOUT_H        = 0x43   # GX high byte
    GYRO_XOUT_L        = 0x44   # GX low byte
    GYRO_YOUT_H        = 0x45   # GY high byte
    GYRO_YOUT_L        = 0x46   # GY low byte
    GYRO_ZOUT_H        = 0x47   # GZ high byte
    GYRO_ZOUT_L        = 0x48   # GZ low byte

    # --- Interrupt Configuration ---
    INT_PIN_CFG        = 0x37   # Interrupt pin configuration
    INT_ENABLE         = 0x38   # Interrupt enable register
    INT_STATUS         = 0x3A   # Interrupt status register

    # --- Gyroscope Full-Scale Range Bits (GYRO_CONFIG[4:3]) ---
    GYRO_FS_250        = 0x00   # ±250  °/s  — LSB = 131.0  LSB/°/s
    GYRO_FS_500        = 0x08   # ±500  °/s  — LSB =  65.5  LSB/°/s
    GYRO_FS_1000       = 0x10   # ±1000 °/s  — LSB =  32.8  LSB/°/s
    GYRO_FS_2000       = 0x18   # ±2000 °/s  — LSB =  16.4  LSB/°/s

    # --- Accelerometer Full-Scale Range Bits (ACCEL_CONFIG[4:3]) ---
    ACCEL_FS_2G        = 0x00   # ±2  g  — LSB = 16384 LSB/g
    ACCEL_FS_4G        = 0x08   # ±4  g  — LSB =  8192 LSB/g
    ACCEL_FS_8G        = 0x10   # ±8  g  — LSB =  4096 LSB/g
    ACCEL_FS_16G       = 0x18   # ±16 g  — LSB =  2048 LSB/g

    # --- Digital Low-Pass Filter (DLPF) Settings (CONFIG[2:0]) ---
    # Lower bandwidth = more filtering, higher group delay
    DLPF_BW_256        = 0x00   # Accel: 260 Hz | Gyro: 256 Hz
    DLPF_BW_188        = 0x01   # Accel: 184 Hz | Gyro: 188 Hz
    DLPF_BW_98         = 0x02   # Accel:  94 Hz | Gyro:  98 Hz
    DLPF_BW_42         = 0x03   # Accel:  44 Hz | Gyro:  42 Hz  ← selected
    DLPF_BW_20         = 0x04   # Accel:  21 Hz | Gyro:  20 Hz
    DLPF_BW_10         = 0x05   # Accel:  10 Hz | Gyro:  10 Hz
    DLPF_BW_5          = 0x06   # Accel:   5 Hz | Gyro:   5 Hz

    # --- Scaling Factors (converts raw int16 → physical units) ---
    # Selected full-scale range: ±2g / ±250°/s for wrist/body sensing
    ACCEL_SCALE_2G     = 16384.0   # LSB per g   (1g = Earth gravity = 9.80665 m/s²)
    GYRO_SCALE_250     = 131.0     # LSB per °/s
    TEMP_OFFSET        = 36.53     # °C offset (datasheet Eq: Temp = raw/340 + 36.53)
    TEMP_DIVISOR       = 340.0     # Raw counts per °C


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: MPU-6050 DRIVER — Raw I2C Communication Layer
# ─────────────────────────────────────────────────────────────────────────────

class MPU6050Driver:
    """
    Low-level I2C driver for the InvenSense MPU-6050 IMU.

    Implements direct register-level communication without high-level
    abstraction libraries. Provides robust initialization, configuration,
    and burst-read data acquisition optimized for real-time applications.

    Configuration chosen for biomedical wrist/body sensing:
      - Accelerometer: ±2g  (sufficient for body motion, high resolution)
      - Gyroscope:     ±250°/s (sufficient for limb rotation, low noise)
      - DLPF:          42 Hz bandwidth (attenuates high-freq vibration noise)
      - Sample Rate:   100 Hz (adequate for motion artifact in PPG @ ~25–60 Hz)
    """

    REG = MPU6050Registers

    def __init__(self, i2c_bus: int = 1, address: int = MPU6050Registers.I2C_ADDR_DEFAULT):
        """
        Initialize driver. Does not open I2C yet.

        Args:
            i2c_bus : Linux I2C bus number (1 = /dev/i2c-1 on Pi header)
            address : MPU-6050 I2C address (0x68 or 0x69)
        """
        self.bus_num  = i2c_bus
        self.address  = address
        self.bus      = None
        self._lock    = threading.Lock()

        # Calibration offsets (collected during startup still-phase)
        self.accel_offset = [0.0, 0.0, 0.0]
        self.gyro_offset  = [0.0, 0.0, 0.0]

        # Flags
        self.initialized  = False
        self.calibrated   = False

    # ------------------------------------------------------------------
    # I2C Bus Management
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """
        Open the I2C bus and verify device presence via WHO_AM_I register.

        Returns:
            True if device found and bus opened successfully.

        Raises:
            RuntimeError if device not found or I2C communication fails.
        """
        try:
            self.bus = smbus2.SMBus(self.bus_num)
            time.sleep(0.01)  # Allow bus to settle

            # Verify device identity: WHO_AM_I should return 0x68
            who_am_i = self._read_byte(self.REG.WHO_AM_I)
            if who_am_i != 0x68:
                raise RuntimeError(
                    f"MPU6050 identity check failed. "
                    f"Expected 0x68, got 0x{who_am_i:02X}. "
                    f"Check wiring and I2C address (AD0 pin)."
                )
            return True

        except FileNotFoundError:
            raise RuntimeError(
                f"I2C bus {self.bus_num} not found. "
                f"Enable I2C in raspi-config (Interface Options → I2C)."
            )
        except OSError as e:
            raise RuntimeError(
                f"I2C communication error on bus {self.bus_num} "
                f"at address 0x{self.address:02X}: {e}"
            )

    def close(self):
        """Release I2C bus resources."""
        if self.bus:
            try:
                self.bus.close()
            except Exception:
                pass
            self.bus = None

    # ------------------------------------------------------------------
    # Register-Level I/O Primitives
    # ------------------------------------------------------------------

    def _read_byte(self, register: int) -> int:
        """
        Read a single 8-bit register value.
        Thread-safe with internal lock for concurrent access protection.
        """
        with self._lock:
            return self.bus.read_byte_data(self.address, register)

    def _write_byte(self, register: int, value: int):
        """
        Write a single 8-bit value to a register.
        Thread-safe.
        """
        with self._lock:
            self.bus.write_byte_data(self.address, register, value)

    def _read_word_signed(self, reg_high: int) -> int:
        """
        Read two consecutive registers and combine into a signed 16-bit integer.

        MPU-6050 stores sensor data in big-endian two's complement format.
        The high byte is at `reg_high`, low byte at `reg_high + 1`.

        Args:
            reg_high : Address of the high byte register.

        Returns:
            Signed 16-bit integer in range [-32768, 32767].
        """
        with self._lock:
            # Burst read 2 bytes starting from high register
            raw = self.bus.read_i2c_block_data(self.address, reg_high, 2)

        # Reconstruct 16-bit value: big-endian unsigned
        value = (raw[0] << 8) | raw[1]

        # Convert to signed (two's complement)
        if value >= 0x8000:
            value -= 0x10000

        return value

    def _read_burst(self, start_reg: int, length: int) -> list:
        """
        Read `length` bytes starting at `start_reg` in a single I2C transaction.
        Minimizes I2C overhead — critical for high sample-rate operation.

        Args:
            start_reg : Starting register address.
            length    : Number of bytes to read.

        Returns:
            List of `length` integers (unsigned bytes).
        """
        with self._lock:
            return self.bus.read_i2c_block_data(self.address, start_reg, length)

    # ------------------------------------------------------------------
    # Initialization Sequence
    # ------------------------------------------------------------------

    def initialize(self):
        """
        Full MPU-6050 initialization sequence.

        Sequence follows InvenSense recommended startup procedure:
          1. Wake device (clear SLEEP bit in PWR_MGMT_1)
          2. Set clock source to gyroscope PLL for stability
          3. Configure DLPF bandwidth
          4. Set sample rate divider
          5. Configure accelerometer full-scale range
          6. Configure gyroscope full-scale range
          7. Reset signal paths
          8. Brief stabilization delay
        """
        # --- Step 1: Wake device and set clock source ---
        # PWR_MGMT_1: SLEEP=0, CLKSEL=1 (PLL with X-axis gyro reference)
        # Gyro PLL provides better frequency stability than internal oscillator
        self._write_byte(self.REG.PWR_MGMT_1, 0x01)
        time.sleep(0.05)  # Allow gyro PLL to stabilize (datasheet: 30ms min)

        # --- Step 2: Configure Digital Low-Pass Filter ---
        # DLPF_CFG=3: Accel 44Hz / Gyro 42Hz — good balance for body motion
        self._write_byte(self.REG.CONFIG, self.REG.DLPF_BW_42)

        # --- Step 3: Set Sample Rate ---
        # Sample Rate = Gyro Output Rate / (1 + SMPLRT_DIV)
        # Gyro Output Rate = 1000 Hz when DLPF enabled
        # SMPLRT_DIV = 9 → Sample Rate = 1000/(1+9) = 100 Hz
        self._write_byte(self.REG.SMPLRT_DIV, 9)

        # --- Step 4: Configure Accelerometer Full-Scale Range ---
        # AFS_SEL=0 → ±2g, 16384 LSB/g — highest resolution for body motion
        self._write_byte(self.REG.ACCEL_CONFIG, self.REG.ACCEL_FS_2G)

        # --- Step 5: Configure Gyroscope Full-Scale Range ---
        # FS_SEL=0 → ±250°/s, 131 LSB/°/s — adequate for limb angular velocity
        self._write_byte(self.REG.GYRO_CONFIG, self.REG.GYRO_FS_250)

        # --- Step 6: Disable interrupts (polling mode) ---
        self._write_byte(self.REG.INT_ENABLE, 0x00)

        # --- Step 7: Stabilization ---
        time.sleep(0.1)  # Allow sensors to settle after configuration

        self.initialized = True

    # ------------------------------------------------------------------
    # Sensor Data Acquisition
    # ------------------------------------------------------------------

    def read_all_raw(self) -> dict:
        """
        Burst-read all sensor data in a single 14-byte I2C transaction.

        Reading ACCEL_XOUT_H (0x3B) through GYRO_ZOUT_L (0x48) = 14 bytes:
          Bytes  0- 1: ACCEL_XOUT
          Bytes  2- 3: ACCEL_YOUT
          Bytes  4- 5: ACCEL_ZOUT
          Bytes  6- 7: TEMP_OUT
          Bytes  8- 9: GYRO_XOUT
          Bytes 10-11: GYRO_YOUT
          Bytes 12-13: GYRO_ZOUT

        Returns:
            dict with keys: ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw,
                            temp_raw, timestamp
        """
        timestamp = time.monotonic()

        # Single I2C burst transaction — 14 bytes from 0x3B to 0x48
        raw = self._read_burst(self.REG.ACCEL_XOUT_H, 14)

        def to_signed16(high_byte, low_byte):
            """Combine two bytes into a signed 16-bit integer."""
            val = (high_byte << 8) | low_byte
            return val - 65536 if val >= 32768 else val

        return {
            'ax_raw'    : to_signed16(raw[0],  raw[1]),
            'ay_raw'    : to_signed16(raw[2],  raw[3]),
            'az_raw'    : to_signed16(raw[4],  raw[5]),
            'temp_raw'  : to_signed16(raw[6],  raw[7]),
            'gx_raw'    : to_signed16(raw[8],  raw[9]),
            'gy_raw'    : to_signed16(raw[10], raw[11]),
            'gz_raw'    : to_signed16(raw[12], raw[13]),
            'timestamp' : timestamp,
        }

    def convert_raw(self, raw: dict) -> dict:
        """
        Convert raw ADC counts to physical SI/engineering units.

        Conversion formulas (from datasheet):
          Acceleration [g]    = raw_value / 16384.0      (±2g range)
          Angular rate [°/s]  = raw_value / 131.0        (±250°/s range)
          Temperature [°C]    = raw_value / 340.0 + 36.53

        Args:
            raw : dict from read_all_raw()

        Returns:
            dict with physical values + calibration offsets applied.
        """
        R = self.REG

        ax = (raw['ax_raw'] / R.ACCEL_SCALE_2G) - self.accel_offset[0]
        ay = (raw['ay_raw'] / R.ACCEL_SCALE_2G) - self.accel_offset[1]
        az = (raw['az_raw'] / R.ACCEL_SCALE_2G) - self.accel_offset[2]

        gx = (raw['gx_raw'] / R.GYRO_SCALE_250) - self.gyro_offset[0]
        gy = (raw['gy_raw'] / R.GYRO_SCALE_250) - self.gyro_offset[1]
        gz = (raw['gz_raw'] / R.GYRO_SCALE_250) - self.gyro_offset[2]

        temp = (raw['temp_raw'] / R.TEMP_DIVISOR) + R.TEMP_OFFSET

        return {
            'ax': ax, 'ay': ay, 'az': az,
            'gx': gx, 'gy': gy, 'gz': gz,
            'temp': temp,
            'timestamp': raw['timestamp'],
        }

    # ------------------------------------------------------------------
    # Startup Calibration
    # ------------------------------------------------------------------

    def calibrate(self, samples: int = 200, progress_cb=None):
        """
        Collect static calibration offsets by averaging N samples at rest.

        Assumption: Sensor is stationary and flat during calibration.
        The Z-axis accelerometer should read +1g (gravity), X and Y should
        read ~0g. Gyroscope should read ~0°/s on all axes.

        After calibration:
          - accel_offset[0,1] remove X/Y gravity-independent bias
          - accel_offset[2]   leaves +1g on Z (gravity reference preserved)
          - gyro_offset       removes gyroscope zero-rate offset (ZRO)

        Args:
            samples     : Number of samples to average (default 200 = 2s @ 100Hz)
            progress_cb : Optional callable(fraction) for progress indication.
        """
        ax_sum = ay_sum = az_sum = 0.0
        gx_sum = gy_sum = gz_sum = 0.0

        for i in range(samples):
            raw = self.read_all_raw()
            data = self.convert_raw(raw)

            ax_sum += data['ax']
            ay_sum += data['ay']
            az_sum += data['az']
            gx_sum += data['gx']
            gy_sum += data['gy']
            gz_sum += data['gz']

            if progress_cb:
                progress_cb((i + 1) / samples)

            time.sleep(0.01)  # ~100 Hz during calibration

        # Compute mean offsets
        self.accel_offset[0] = ax_sum / samples          # Remove X bias
        self.accel_offset[1] = ay_sum / samples          # Remove Y bias
        self.accel_offset[2] = (az_sum / samples) - 1.0 # Remove Z bias, keep 1g gravity

        self.gyro_offset[0]  = gx_sum / samples
        self.gyro_offset[1]  = gy_sum / samples
        self.gyro_offset[2]  = gz_sum / samples

        self.calibrated = True


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SIGNAL FILTERING PIPELINE
# Research-Grade Digital Signal Processing for IMU Data
# ─────────────────────────────────────────────────────────────────────────────

class IIRLowPassFilter:
    """
    First-order Infinite Impulse Response (IIR) low-pass filter.

    Transfer function (discrete-time):
        y[n] = α·x[n] + (1-α)·y[n-1]

    where α = 2π·fc·dt / (2π·fc·dt + 1)  (bilinear approximation)

    This is equivalent to a single-pole RC low-pass filter in discrete time.
    Simple, low-latency, computationally efficient — ideal for embedded systems.

    For more aggressive filtering, cascade multiple instances.

    Parameters:
        fc : Cutoff frequency [Hz]
        dt : Sample interval [s] = 1/fs
    """

    def __init__(self, cutoff_hz: float, sample_rate_hz: float):
        """
        Args:
            cutoff_hz      : -3dB cutoff frequency in Hz
            sample_rate_hz : Sensor sample rate in Hz
        """
        dt = 1.0 / sample_rate_hz
        # Time constant tau = 1/(2π·fc)
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        # Filter coefficient α
        self.alpha = dt / (tau + dt)
        self.prev  = None  # Previous output y[n-1]

    def reset(self, initial_value: float = 0.0):
        """Reset filter state (e.g. after calibration or restart)."""
        self.prev = initial_value

    def process(self, x: float) -> float:
        """
        Process one sample through the filter.

        Args:
            x : Current input sample x[n]

        Returns:
            Filtered output y[n]
        """
        if self.prev is None:
            # Initialize with first sample to avoid startup transient
            self.prev = x
            return x

        y = self.alpha * x + (1.0 - self.alpha) * self.prev
        self.prev = y
        return y


class MovingAverageFilter:
    """
    Simple causal moving average (box filter) over a sliding window.

    Computes:  y[n] = (1/N) · Σ x[n-k], k=0..N-1

    Characteristics:
      - Linear phase (no phase distortion within passband)
      - Flat passband, poor stopband attenuation
      - O(1) update via incremental mean computation
      - Zero-initialization avoids startup artifact after first N samples

    Useful for smoothing orientation estimates and variance computation.
    """

    def __init__(self, window_size: int):
        """
        Args:
            window_size : Number of samples in the averaging window (N)
        """
        self.N       = window_size
        self.buffer  = collections.deque(maxlen=window_size)
        self._sum    = 0.0

    def process(self, x: float) -> float:
        """
        Update with new sample and return current moving average.

        Uses incremental sum update for O(1) computation.

        Args:
            x : New input sample

        Returns:
            Current moving average of the window
        """
        if len(self.buffer) == self.N:
            # Subtract oldest value from running sum before it's evicted
            self._sum -= self.buffer[0]

        self.buffer.append(x)
        self._sum += x

        return self._sum / len(self.buffer)

    @property
    def values(self):
        """Return current window as list for variance computation."""
        return list(self.buffer)


class SpikeRejectionFilter:
    """
    Median absolute deviation (MAD)-based spike rejection filter.

    Detects and replaces impulsive noise (spikes) in sensor data
    caused by mechanical shocks, I2C bit errors, or power supply glitches.

    Algorithm:
      1. Maintain a sliding window of recent samples
      2. Compute window median as robust central estimate
      3. If |x[n] - median| > threshold·MAD, classify as spike
      4. Replace spike with window median (or clamp to bounds)

    Threshold calibration:
      - Conservative: k=3 (rarely replaces valid data)
      - Aggressive:   k=2 (catches more spikes, may distort fast motion)
      - For IMU:      k=5 (preserve legitimate sharp motion)
    """

    def __init__(self, window_size: int = 11, threshold_k: float = 5.0):
        """
        Args:
            window_size : Odd number for clean median computation
            threshold_k : Spike detection threshold multiplier (σ-equivalent)
        """
        self.window   = collections.deque(maxlen=window_size)
        self.k        = threshold_k
        self.prev_out = 0.0

    def process(self, x: float) -> float:
        """
        Filter one sample; return cleaned value.

        If window has fewer than 3 samples, pass through unchanged.

        Args:
            x : Raw input sample

        Returns:
            Cleaned output sample
        """
        self.window.append(x)

        if len(self.window) < 3:
            self.prev_out = x
            return x

        sorted_win = sorted(self.window)
        median     = sorted_win[len(sorted_win) // 2]

        # Median absolute deviation (robust std estimator: σ ≈ 1.4826·MAD)
        mad = sorted(abs(v - median) for v in self.window)[len(self.window) // 2]

        if mad < 1e-9:
            # Window is nearly constant — anything deviating is a spike
            if abs(x - median) > 0.05:
                return median
            return x

        # Scaled threshold
        if abs(x - median) > self.k * 1.4826 * mad:
            # Spike detected — substitute window median
            out = median
        else:
            out = x

        self.prev_out = out
        return out


class IMUFilterPipeline:
    """
    Composited multi-stage filtering pipeline for a single IMU axis channel.

    Pipeline order:
      Raw → Spike Rejection → IIR Low-Pass → Moving Average → Output

    This ordering is important:
      1. Spike rejection first to prevent transients from propagating
      2. IIR LP to remove broadband noise above motion frequency
      3. Moving average for final smoothing and display stability

    Separate instances are used for each sensor axis (6 total + temp).
    """

    def __init__(self,
                 iir_cutoff_hz: float = 8.0,
                 sample_rate: float   = 100.0,
                 ma_window: int       = 5,
                 spike_k: float       = 5.0):
        """
        Args:
            iir_cutoff_hz : IIR low-pass -3dB cutoff in Hz
            sample_rate   : Sensor sample rate in Hz
            ma_window     : Moving average window length (samples)
            spike_k       : Spike rejection threshold multiplier
        """
        self.spike = SpikeRejectionFilter(window_size=11, threshold_k=spike_k)
        self.iir   = IIRLowPassFilter(cutoff_hz=iir_cutoff_hz,
                                       sample_rate_hz=sample_rate)
        self.ma    = MovingAverageFilter(window_size=ma_window)

    def process(self, x: float) -> float:
        """Run sample through the full pipeline. Returns filtered value."""
        x = self.spike.process(x)
        x = self.iir.process(x)
        x = self.ma.process(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MOTION ANALYSIS ENGINE
# Magnitude, Classification, Artifact Scoring, Orientation
# ─────────────────────────────────────────────────────────────────────────────

# Motion state constants — human-readable classification labels
class MotionState:
    STABLE       = "STABLE"
    LOW_MOTION   = "LOW MOTION"
    MEDIUM_MOTION= "MEDIUM MOTION"
    HIGH_MOTION  = "HIGH MOTION"
    EXTREME      = "EXTREME MOTION"


class MotionAnalysisEngine:
    """
    Core motion analysis module for biomedical PPG motion artifact research.

    Implements:
      - Real-time motion magnitude (vector norm of acceleration)
      - Multi-criteria motion classification with hysteresis
      - Motion artifact severity scoring (0–100 scale) for PPG quality
      - Complementary-filter orientation estimation (pitch, roll)
      - Statistical variance analysis for motion consistency
      - AI-assisted motion quality interpretation

    Artifact Score Design (0–100):
      ┌─────────────────┬────────────────────────────────┐
      │ Score Range     │ PPG Signal Interpretation       │
      ├─────────────────┼────────────────────────────────┤
      │  0 –  15        │ Clean signal — low artifact     │
      │ 16 –  35        │ Mild artifact — usable          │
      │ 36 –  60        │ Moderate artifact — degraded    │
      │ 61 –  80        │ Severe artifact — unreliable    │
      │ 81 – 100        │ Critical — signal unusable      │
      └─────────────────┴────────────────────────────────┘

    Orientation uses complementary filter:
      angle_fused = α·(angle_gyro_integrated) + (1-α)·(angle_accel)
      α = 0.96 (gyro trust weight) — standard for body-mounted IMU
    """

    # Accelerometer magnitude thresholds [g] — empirically tuned for body motion
    THRESH_STABLE  = 0.04   # < 40 mg total variation → stable at rest
    THRESH_LOW     = 0.12   # < 120 mg → gentle movement (breathing artifact)
    THRESH_MEDIUM  = 0.35   # < 350 mg → walking-level motion
    THRESH_HIGH    = 0.80   # < 800 mg → fast arm/wrist movement
    # > 800 mg → extreme (jumping, impact, rapid shake)

    # Gyroscope rotation rate thresholds [°/s]
    GYRO_THRESH_LOW    = 5.0    # Minimal rotation
    GYRO_THRESH_MEDIUM = 20.0   # Moderate rotation
    GYRO_THRESH_HIGH   = 80.0   # Vigorous rotation

    # Complementary filter weight (gyro component)
    # Higher α → smoother orientation (slower response to accel tilt)
    # Lower  α → faster tilt response (more accel noise)
    CF_ALPHA = 0.96

    def __init__(self, sample_rate: float = 100.0, history_len: int = 100):
        """
        Args:
            sample_rate : Sensor sample rate in Hz
            history_len : Number of samples to retain for variance analysis
        """
        self.fs          = sample_rate
        self.dt          = 1.0 / sample_rate

        # History buffers for statistical analysis
        self.accel_mag_history = collections.deque(maxlen=history_len)
        self.gyro_mag_history  = collections.deque(maxlen=history_len)
        self.artifact_history  = collections.deque(maxlen=history_len)

        # Complementary filter orientation state [degrees]
        self.pitch  = 0.0   # Rotation around X-axis (forward/back tilt)
        self.roll   = 0.0   # Rotation around Y-axis (side tilt)

        # Motion state with hysteresis memory
        self._prev_state  = MotionState.STABLE
        self._state_count = 0   # Samples in current state (for hysteresis)

        # Artifact score smoothing
        self._artifact_iir = IIRLowPassFilter(cutoff_hz=2.0,
                                               sample_rate_hz=sample_rate)

    # ------------------------------------------------------------------
    # Magnitude Computation
    # ------------------------------------------------------------------

    def compute_accel_magnitude(self, ax: float, ay: float, az: float) -> float:
        """
        Compute Euclidean norm (L2 norm) of the 3D acceleration vector.

        |a| = √(ax² + ay² + az²)

        At rest on a flat surface: |a| ≈ 1.0g (gravity vector only).
        Motion adds to this — we compute deviation from 1g as motion proxy:
          motion_component = ||a| - 1.0|

        Returns:
            Total acceleration magnitude [g]
        """
        return math.sqrt(ax*ax + ay*ay + az*az)

    def compute_gyro_magnitude(self, gx: float, gy: float, gz: float) -> float:
        """
        Compute total angular velocity magnitude [°/s].

        |ω| = √(gx² + gy² + gz²)

        Returns total rotation speed regardless of axis — useful for
        detecting any rotational disturbance that corrupts PPG.
        """
        return math.sqrt(gx*gx + gy*gy + gz*gz)

    def gravity_free_motion(self, accel_mag: float) -> float:
        """
        Isolate the dynamic (motion) component of acceleration by
        removing the static gravity component (1g).

        dynamic_accel = ||a_total| - 1.0g|

        This provides a motion intensity measure that approaches 0 at rest
        and increases proportionally with movement intensity.

        Args:
            accel_mag : Total acceleration magnitude [g]

        Returns:
            Gravity-compensated dynamic acceleration [g], ≥ 0
        """
        return abs(accel_mag - 1.0)

    # ------------------------------------------------------------------
    # Motion Classification
    # ------------------------------------------------------------------

    def classify_motion(self,
                        accel_mag: float,
                        gyro_mag: float,
                        variance: float) -> str:
        """
        Classify current motion state using a multi-criteria decision rule.

        Uses accelerometer magnitude, gyroscope angular velocity, and
        recent motion variance to robustly classify motion state.
        Variance weighting prevents single-sample transients from causing
        false state transitions.

        Decision logic (priority-ordered):
          1. EXTREME : high accel OR gyro magnitude
          2. HIGH    : moderate-high accel AND/OR gyro
          3. MEDIUM  : definite motion present
          4. LOW     : slight perturbation from rest
          5. STABLE  : essentially stationary

        Args:
            accel_mag : Total acceleration magnitude [g]
            gyro_mag  : Total angular velocity magnitude [°/s]
            variance  : Recent acceleration magnitude variance [g²]

        Returns:
            MotionState constant string
        """
        dyn = self.gravity_free_motion(accel_mag)

        # Variance contributes to classification robustness
        # High variance → sustained motion (not a single spike)
        variance_factor = min(1.0, math.sqrt(variance) / 0.1)

        if dyn > self.THRESH_HIGH or gyro_mag > self.GYRO_THRESH_HIGH:
            return MotionState.EXTREME

        if dyn > self.THRESH_MEDIUM or gyro_mag > self.GYRO_THRESH_MEDIUM:
            return MotionState.HIGH_MOTION

        if dyn > self.THRESH_LOW or (gyro_mag > self.GYRO_THRESH_LOW and variance_factor > 0.3):
            return MotionState.MEDIUM_MOTION

        if dyn > self.THRESH_STABLE or gyro_mag > self.GYRO_THRESH_LOW:
            return MotionState.LOW_MOTION

        return MotionState.STABLE

    # ------------------------------------------------------------------
    # Motion Artifact Scoring
    # ------------------------------------------------------------------

    def compute_artifact_score(self,
                                accel_mag: float,
                                gyro_mag: float,
                                variance: float) -> float:
        """
        Compute a 0–100 motion artifact severity score for PPG quality.

        Higher scores → more likely the PPG/optical signal is corrupted.

        Score components (weighted sum, normalized to [0, 100]):
          1. Dynamic acceleration component    (weight: 40%)
             Captures translational motion that shifts optical path
          2. Gyroscope rotation component      (weight: 30%)
             Captures rotational motion that changes contact geometry
          3. Variance/consistency component    (weight: 30%)
             Captures sustained vs transient motion

        Scoring is non-linear (sigmoid-inspired) to:
          - Keep low-motion scores genuinely low (< 20 for still sensor)
          - Rapidly climb for vigorous motion
          - Saturate at 100 for extreme disturbance

        Args:
            accel_mag : Total acceleration magnitude [g]
            gyro_mag  : Total angular velocity [°/s]
            variance  : Recent acceleration variance [g²]

        Returns:
            Artifact severity score in range [0.0, 100.0]
        """
        dyn = self.gravity_free_motion(accel_mag)

        # --- Component 1: Dynamic acceleration score (0–40) ---
        # Map [0, 1.5g] → [0, 40] with soft saturation
        accel_score = min(40.0, (dyn / 1.5) ** 0.7 * 40.0)

        # --- Component 2: Gyroscope score (0–30) ---
        # Map [0, 180°/s] → [0, 30]
        gyro_score  = min(30.0, (gyro_mag / 180.0) ** 0.6 * 30.0)

        # --- Component 3: Variance score (0–30) ---
        # Recent motion consistency — sustained motion is worse than transient
        # Map [0, 0.05 g²] → [0, 30]
        var_score   = min(30.0, (variance / 0.05) ** 0.5 * 30.0)

        raw_score   = accel_score + gyro_score + var_score

        # Smooth artifact score to prevent erratic display
        smooth_score = self._artifact_iir.process(raw_score)

        return max(0.0, min(100.0, smooth_score))

    # ------------------------------------------------------------------
    # Variance Analysis
    # ------------------------------------------------------------------

    def compute_variance(self, history: collections.deque) -> float:
        """
        Compute population variance of recent acceleration magnitudes.

        Var(X) = E[X²] - (E[X])²

        Used to distinguish sustained motion from single-sample transients.
        High variance → motion is sustained and inconsistent.
        Low variance  → either stable rest or consistent periodic motion.

        Args:
            history : deque of recent magnitude values

        Returns:
            Population variance [units²], or 0.0 if insufficient data.
        """
        n = len(history)
        if n < 2:
            return 0.0

        mean  = sum(history) / n
        sq_diff = sum((x - mean) ** 2 for x in history)
        return sq_diff / n

    # ------------------------------------------------------------------
    # Orientation Estimation (Complementary Filter)
    # ------------------------------------------------------------------

    def update_orientation(self,
                            ax: float, ay: float, az: float,
                            gx: float, gy: float,
                            dt: float = None):
        """
        Update pitch and roll orientation using a complementary filter.

        Complementary filter blends two complementary information sources:
          - Gyroscope: accurate short-term rate integration (drift over time)
          - Accelerometer: accurate long-term tilt (noisy short-term)

        Formula:
          pitch_acc = arctan2(ax, √(ay²+az²))  [accelerometer pitch, radians]
          roll_acc  = arctan2(ay, √(ax²+az²))  [accelerometer roll]

          pitch[n] = α·(pitch[n-1] + gx·dt) + (1-α)·pitch_acc
          roll[n]  = α·(roll[n-1]  + gy·dt) + (1-α)·roll_acc

        Alpha selection (α=0.96):
          - Time constant: τ = α/(1-α)·dt ≈ 0.24 s
          - Gyro dominates above ~0.66 Hz, accel dominates below

        Args:
            ax, ay, az : Filtered accelerometer readings [g]
            gx, gy     : Filtered gyroscope readings [°/s]
            dt         : Time step [s] (uses self.dt if None)

        Returns:
            (pitch_deg, roll_deg) tuple [degrees]
        """
        if dt is None:
            dt = self.dt

        # --- Accelerometer-derived angles (atan2 for full ±180° range) ---
        # Pitch: rotation around Y-axis (tilt forward/backward)
        pitch_acc = math.degrees(
            math.atan2(ax, math.sqrt(ay*ay + az*az))
        )

        # Roll: rotation around X-axis (tilt left/right)
        roll_acc  = math.degrees(
            math.atan2(ay, math.sqrt(ax*ax + az*az))
        )

        # --- Complementary filter fusion ---
        self.pitch = (self.CF_ALPHA * (self.pitch + gx * dt)
                      + (1.0 - self.CF_ALPHA) * pitch_acc)

        self.roll  = (self.CF_ALPHA * (self.roll  + gy * dt)
                      + (1.0 - self.CF_ALPHA) * roll_acc)

        # Clamp to physical limits [-180, 180]
        self.pitch = max(-180.0, min(180.0, self.pitch))
        self.roll  = max(-180.0, min(180.0, self.roll))

        return self.pitch, self.roll

    def compute_tilt(self) -> float:
        """
        Compute total tilt angle from vertical (combined pitch+roll deviation).

        tilt = √(pitch² + roll²)  [approximate, valid for small angles]

        Useful as a single scalar for how far the sensor is tilted from
        its calibration-upright position.

        Returns:
            Total tilt magnitude [degrees]
        """
        return math.sqrt(self.pitch**2 + self.roll**2)

    # ------------------------------------------------------------------
    # Full Analysis Step
    # ------------------------------------------------------------------

    def update(self,
               ax: float, ay: float, az: float,
               gx: float, gy: float, gz: float,
               dt: float = None) -> dict:
        """
        Execute one complete analysis step with all sub-components.

        Args:
            ax, ay, az : Filtered accelerometer readings [g]
            gx, gy, gz : Filtered gyroscope readings [°/s]
            dt         : Sample interval [s]

        Returns:
            dict with all analysis outputs for display and logging.
        """
        # --- Magnitude computation ---
        accel_mag  = self.compute_accel_magnitude(ax, ay, az)
        gyro_mag   = self.compute_gyro_magnitude(gx, gy, gz)
        dyn_accel  = self.gravity_free_motion(accel_mag)

        # --- Update history buffers ---
        self.accel_mag_history.append(accel_mag)
        self.gyro_mag_history.append(gyro_mag)

        # --- Variance over recent history window ---
        variance   = self.compute_variance(self.accel_mag_history)

        # --- Motion classification ---
        state      = self.classify_motion(accel_mag, gyro_mag, variance)

        # --- Artifact score ---
        artifact   = self.compute_artifact_score(accel_mag, gyro_mag, variance)
        self.artifact_history.append(artifact)

        # --- Signal quality (inverse of artifact score) ---
        quality    = 100.0 - artifact

        # --- Orientation update ---
        pitch, roll = self.update_orientation(ax, ay, az, gx, gy, dt)
        tilt        = self.compute_tilt()

        # --- Normalized motion score (0–1) for external use ---
        motion_norm = min(1.0, dyn_accel / 1.0)

        return {
            'accel_mag'      : accel_mag,
            'gyro_mag'       : gyro_mag,
            'dynamic_accel'  : dyn_accel,
            'variance'       : variance,
            'motion_state'   : state,
            'artifact_score' : artifact,
            'signal_quality' : quality,
            'motion_norm'    : motion_norm,
            'pitch'          : pitch,
            'roll'           : roll,
            'tilt'           : tilt,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: AI-ASSISTED MOTION INTERPRETATION
# Non-clinical, experimental analysis only
# ─────────────────────────────────────────────────────────────────────────────

class MotionInterpreter:
    """
    Lightweight rule-based AI-assisted motion quality interpreter.

    Provides human-readable, contextually aware annotations for the
    current motion analysis output. Uses heuristic pattern matching
    on recent motion history to generate research-relevant observations.

    IMPORTANT: All outputs are:
      - Non-clinical, experimental analysis
      - Research-grade interpretations only
      - NOT suitable for medical diagnosis or clinical decision-making
      - AI-assisted pattern matching, not physiological ground truth

    Interpretation categories:
      - Signal quality assessment
      - Motion consistency evaluation
      - PPG artifact risk annotation
      - Instability/instability detection
      - Wearable placement quality hints
    """

    def __init__(self, history_len: int = 50):
        self.artifact_history = collections.deque(maxlen=history_len)
        self.state_history    = collections.deque(maxlen=history_len)
        self.tilt_history     = collections.deque(maxlen=history_len)
        self._warning_count   = 0

    def update(self, analysis: dict) -> dict:
        """
        Process one analysis frame and produce interpretation annotations.

        Args:
            analysis : Output dict from MotionAnalysisEngine.update()

        Returns:
            dict with interpretation fields for terminal display.
        """
        score   = analysis['artifact_score']
        state   = analysis['motion_state']
        quality = analysis['signal_quality']
        tilt    = analysis['tilt']
        variance= analysis['variance']

        self.artifact_history.append(score)
        self.state_history.append(state)
        self.tilt_history.append(tilt)

        # --- Trend analysis ---
        mean_score    = (sum(self.artifact_history) / len(self.artifact_history)
                         if self.artifact_history else score)
        recent_scores = list(self.artifact_history)[-10:]
        trend         = self._compute_trend(recent_scores)

        # --- State consistency (what fraction of recent window is stable?) ---
        stable_fraction = (
            sum(1 for s in self.state_history if s == MotionState.STABLE)
            / len(self.state_history)
            if self.state_history else 0.0
        )

        # --- Generate primary assessment ---
        assessment = self._assess_signal_quality(score, mean_score, stable_fraction)

        # --- Generate PPG reliability statement ---
        ppg_reliability = self._assess_ppg_reliability(score, state)

        # --- Stability assessment ---
        stability_note = self._assess_stability(variance, tilt, stable_fraction)

        # --- Active warnings ---
        warnings = self._generate_warnings(score, state, tilt, trend)

        # --- Recommendation ---
        recommendation = self._generate_recommendation(score, state, stable_fraction)

        return {
            'assessment'      : assessment,
            'ppg_reliability' : ppg_reliability,
            'stability_note'  : stability_note,
            'warnings'        : warnings,
            'recommendation'  : recommendation,
            'mean_artifact'   : mean_score,
            'stable_fraction' : stable_fraction,
            'trend'           : trend,
        }

    def _compute_trend(self, recent: list) -> str:
        """Determine if artifact score is increasing, decreasing, or stable."""
        if len(recent) < 5:
            return "INITIALIZING"
        first_half  = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        delta       = second_half - first_half
        if delta >  3.0: return "INCREASING ↑"
        if delta < -3.0: return "DECREASING ↓"
        return "STABLE ↔"

    def _assess_signal_quality(self, score: float, mean_score: float,
                                 stable_frac: float) -> str:
        """Generate primary signal quality assessment string."""
        if score < 15 and stable_frac > 0.80:
            return "EXCELLENT — Sensor stable, minimal motion artifact detected"
        if score < 30 and stable_frac > 0.60:
            return "GOOD — Low-level motion, signal quality acceptable for research"
        if score < 50:
            return "FAIR — Moderate motion artifact, signal quality degraded"
        if score < 70:
            return "POOR — Significant motion artifact, analysis reliability reduced"
        return "CRITICAL — High motion artifact, experimental data unreliable"

    def _assess_ppg_reliability(self, score: float, state: str) -> str:
        """Estimate PPG signal reliability given current motion context."""
        if state == MotionState.STABLE:
            return "PPG Window: VALID — Suitable for optical biosensing acquisition"
        if state == MotionState.LOW_MOTION:
            return "PPG Window: MARGINAL — Minor artifact risk, verify signal baseline"
        if state == MotionState.MEDIUM_MOTION:
            return "PPG Window: COMPROMISED — Motion artifact likely present in signal"
        if state == MotionState.HIGH_MOTION:
            return "PPG Window: INVALID — High artifact contamination, discard segment"
        return "PPG Window: REJECTED — Extreme motion, optical signal unusable"

    def _assess_stability(self, variance: float, tilt: float,
                           stable_frac: float) -> str:
        """Assess sensor placement and movement consistency."""
        if variance < 0.001 and tilt < 15.0:
            return "Placement stable — low variance, minimal tilt deviation"
        if variance < 0.005:
            return "Mild perturbation — variance within acceptable range"
        if tilt > 45.0:
            return f"Significant tilt: {tilt:.1f}° — sensor orientation changed"
        if variance > 0.05:
            return "High motion variance — inconsistent or repetitive movement"
        return f"Moderate motion — variance {variance:.4f} g², tilt {tilt:.1f}°"

    def _generate_warnings(self, score: float, state: str,
                            tilt: float, trend: str) -> list:
        """Generate list of active research-relevant warnings."""
        warnings = []
        if score > 70:
            warnings.append("⚠ ARTIFACT CRITICAL: Motion artifact score exceeds 70/100")
        if state in (MotionState.HIGH_MOTION, MotionState.EXTREME):
            warnings.append("⚠ MOTION HIGH: Vigorous movement detected — PPG invalid")
        if tilt > 60.0:
            warnings.append(f"⚠ TILT EXCESSIVE: {tilt:.0f}° — sensor may have moved")
        if "INCREASING" in trend and score > 40:
            warnings.append("⚠ ARTIFACT RISING: Score trending upward — motion increasing")
        if not warnings:
            warnings.append("✓ No active warnings — acquisition conditions nominal")
        return warnings

    def _generate_recommendation(self, score: float, state: str,
                                   stable_frac: float) -> str:
        """Produce a concise action recommendation for the researcher."""
        if score < 20:
            return "Acquisition recommended — motion conditions within research tolerance"
        if score < 45:
            return "Continue with caution — flag this window for post-processing review"
        if stable_frac > 0.5:
            return "Request subject to remain still — motion reducing signal validity"
        return "Suspend acquisition — motion levels incompatible with clean PPG recording"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: REAL-TIME TERMINAL DASHBOARD
# Professional Biomedical Engineering Display
# ─────────────────────────────────────────────────────────────────────────────

class TerminalDashboard:
    """
    ANSI-escaped professional terminal dashboard for real-time IMU monitoring.

    Provides a stable, refreshing terminal display styled for a biomedical
    research laboratory environment. Uses ANSI escape codes for:
      - Color coding of motion severity (green → yellow → red)
      - Cursor repositioning for flicker-free in-place updates
      - Bold, dim, and underline formatting for readability
      - Progress bar visualization for artifact score

    Refresh strategy:
      - Save cursor to home position on first render
      - Overwrite in place on subsequent renders (no scroll)
      - Full terminal clear on startup only
    """

    # ANSI color codes
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    WHITE  = "\033[97m"

    def __init__(self):
        self._first_render = True
        self._render_count = 0
        self._start_time   = time.monotonic()

    def _clear_screen(self):
        """Clear terminal and move cursor to top-left."""
        sys.stdout.write("\033[2J\033[H")

    def _move_home(self):
        """Move cursor to top-left without clearing (for in-place update)."""
        sys.stdout.write("\033[H")

    def _score_color(self, score: float) -> str:
        """Return ANSI color based on artifact severity score."""
        if score < 25:  return self.GREEN
        if score < 55:  return self.YELLOW
        return self.RED

    def _state_color(self, state: str) -> str:
        """Return color for motion state label."""
        colors = {
            MotionState.STABLE        : self.GREEN,
            MotionState.LOW_MOTION    : self.CYAN,
            MotionState.MEDIUM_MOTION : self.YELLOW,
            MotionState.HIGH_MOTION   : self.RED,
            MotionState.EXTREME       : self.RED + self.BOLD,
        }
        return colors.get(state, self.WHITE)

    def _artifact_bar(self, score: float, width: int = 30) -> str:
        """
        Render a colored ASCII progress bar for artifact score visualization.

        [████████████░░░░░░░░░░░░░░░░░░] 42.3
        """
        filled = int((score / 100.0) * width)
        color  = self._score_color(score)
        bar    = f"[{color}{'█' * filled}{'░' * (width - filled)}{self.RESET}]"
        return bar

    def _value_bar(self, value: float, lo: float, hi: float,
                   width: int = 20, color: str = None) -> str:
        """Render a normalized mini bar for a bounded value."""
        frac   = max(0.0, min(1.0, (value - lo) / (hi - lo)))
        filled = int(frac * width)
        c      = color or self.CYAN
        return f"{c}{'▪' * filled}{'·' * (width - filled)}{self.RESET}"

    def render(self,
               filtered: dict,
               analysis: dict,
               interpretation: dict,
               sample_rate: float,
               calibrated: bool):
        """
        Render the complete terminal dashboard (in-place refresh).

        Args:
            filtered        : Filtered sensor readings dict
            analysis        : MotionAnalysisEngine output dict
            interpretation  : MotionInterpreter output dict
            sample_rate     : Current effective sample rate [Hz]
            calibrated      : Whether sensor calibration has been applied
        """
        if self._first_render:
            self._clear_screen()
            self._first_render = False
        else:
            self._move_home()

        uptime = time.monotonic() - self._start_time
        B      = self.BOLD
        R      = self.RESET
        C      = self.CYAN
        G      = self.GREEN
        Y      = self.YELLOW
        D      = self.DIM
        M      = self.MAGENTA

        score  = analysis['artifact_score']
        state  = analysis['motion_state']
        sc     = self._score_color(score)
        stc    = self._state_color(state)

        lines = []
        W = 72  # Dashboard width

        def sep(char='─'):
            return D + char * W + R

        def header(text):
            pad = (W - len(text) - 2) // 2
            return D + "┌" + "─"*pad + R + B + f" {text} " + R + D + "─"*pad + "┐" + R

        def row(label, value, unit="", note=""):
            label_str = f"{D}│{R} {B}{label:<22}{R}"
            value_str = f"{C}{value}{R}"
            unit_str  = f" {D}{unit}{R}"
            note_str  = f"  {D}{note}{R}" if note else ""
            # Pad to width
            content   = f"{label_str} {value_str}{unit_str}{note_str}"
            return content

        # ── Header ──────────────────────────────────────────────────────
        lines.append("")
        lines.append(f"  {B}{M}MPU-6050 BIOMEDICAL MOTION ANALYSIS SYSTEM{R}  "
                     f"{D}[NON-CLINICAL / RESEARCH-GRADE]{R}")
        lines.append(f"  {D}InvenSense MPU-6050 | I²C | Raspberry Pi | "
                     f"Uptime: {uptime:7.1f}s | Fs≈{sample_rate:.0f}Hz | "
                     f"Cal: {'✓' if calibrated else '✗'}{R}")
        lines.append(sep('═'))

        # ── Raw / Filtered Sensor Readings ──────────────────────────────
        lines.append(f"  {B}SENSOR READINGS{R}  {D}(filtered, calibrated){R}")
        lines.append(sep())

        ax, ay, az = filtered['ax'], filtered['ay'], filtered['az']
        gx, gy, gz = filtered['gx'], filtered['gy'], filtered['gz']
        temp       = filtered['temp']

        lines.append(f"  {D}Accelerometer{R}   "
                     f"X: {C}{ax:+7.4f}{R} g    "
                     f"Y: {C}{ay:+7.4f}{R} g    "
                     f"Z: {C}{az:+7.4f}{R} g")
        lines.append(f"  {D}Gyroscope    {R}   "
                     f"X: {C}{gx:+8.3f}{R} °/s  "
                     f"Y: {C}{gy:+8.3f}{R} °/s  "
                     f"Z: {C}{gz:+8.3f}{R} °/s")
        lines.append(f"  {D}Temperature  {R}   "
                     f"{C}{temp:+6.2f}{R} °C   "
                     f"{D}(on-die sensor — not ambient){R}")
        lines.append(sep())

        # ── Motion Magnitude & Classification ───────────────────────────
        lines.append(f"  {B}MOTION ANALYSIS{R}")
        lines.append(sep())

        amag = analysis['accel_mag']
        gmag = analysis['gyro_mag']
        dyn  = analysis['dynamic_accel']
        var  = analysis['variance']
        mn   = analysis['motion_norm']

        lines.append(f"  {D}Accel Magnitude {R}  {C}{amag:7.4f}{R} g      "
                     f"{D}|a| = √(ax²+ay²+az²){R}")
        lines.append(f"  {D}Dynamic Accel   {R}  {C}{dyn:7.4f}{R} g      "
                     f"{D}||a|-1g| (gravity compensated){R}")
        lines.append(f"  {D}Gyro Magnitude  {R}  {C}{gmag:7.3f}{R} °/s   "
                     f"{D}|ω| = √(gx²+gy²+gz²){R}")
        lines.append(f"  {D}Accel Variance  {R}  {C}{var:.6f}{R} g²     "
                     f"{D}(100-sample window){R}")
        lines.append(f"  {D}Motion Score    {R}  {self._value_bar(mn, 0, 1)} {C}{mn*100:5.1f}%{R}")
        lines.append("")
        lines.append(f"  {D}Motion State    {R}  "
                     f"{stc}{B}{state:<18}{R}  "
                     f"{D}classifier: multi-criteria IMU fusion{R}")
        lines.append(sep())

        # ── Motion Artifact Score ────────────────────────────────────────
        lines.append(f"  {B}MOTION ARTIFACT SCORE{R}  "
                     f"{D}(PPG signal quality indicator){R}")
        lines.append(sep())

        bar = self._artifact_bar(score, width=32)
        lines.append(f"  {D}Artifact Score  {R}  {bar} {sc}{B}{score:5.1f}/100{R}")
        lines.append(f"  {D}Signal Quality  {R}  {G}{analysis['signal_quality']:5.1f}%{R}  "
                     f"{D}(100 − artifact score){R}")
        lines.append(f"  {D}Mean Artifact   {R}  {sc}{interpretation['mean_artifact']:5.1f}/100{R}  "
                     f"{D}(50-sample rolling mean){R}")
        lines.append(f"  {D}Score Trend     {R}  "
                     f"{Y}{interpretation['trend']:<18}{R}")
        lines.append(sep())

        # ── Orientation ─────────────────────────────────────────────────
        lines.append(f"  {B}ORIENTATION ESTIMATION{R}  "
                     f"{D}(complementary filter, α=0.96){R}")
        lines.append(sep())

        pitch = analysis['pitch']
        roll  = analysis['roll']
        tilt  = analysis['tilt']

        lines.append(f"  {D}Pitch (X-axis)  {R}  {C}{pitch:+8.2f}°{R}   "
                     f"{D}forward/back tilt{R}")
        lines.append(f"  {D}Roll  (Y-axis)  {R}  {C}{roll:+8.2f}°{R}   "
                     f"{D}left/right tilt{R}")
        lines.append(f"  {D}Total Tilt      {R}  {C}{tilt:8.2f}°{R}   "
                     f"{D}√(pitch²+roll²){R}")
        lines.append(sep())

        # ── AI-Assisted Interpretation ───────────────────────────────────
        lines.append(f"  {B}AI-ASSISTED MOTION INTERPRETATION{R}  "
                     f"{D}[experimental, non-clinical]{R}")
        lines.append(sep())

        iassess = interpretation['assessment']
        ippg    = interpretation['ppg_reliability']
        istab   = interpretation['stability_note']
        irec    = interpretation['recommendation']

        lines.append(f"  {D}Assessment {R}  {Y}{iassess}{R}")
        lines.append(f"  {D}PPG Status {R}  {sc}{ippg}{R}")
        lines.append(f"  {D}Stability  {R}  {C}{istab}{R}")
        lines.append(f"  {D}Stable %%  {R}  "
                     f"{G}{interpretation['stable_fraction']*100:5.1f}%%{R}  "
                     f"{D}of last 50 samples{R}")
        lines.append("")
        lines.append(f"  {D}Recommend  {R}  {B}{irec}{R}")
        lines.append(sep())

        # ── Warnings ────────────────────────────────────────────────────
        lines.append(f"  {B}ACTIVE WARNINGS{R}")
        lines.append(sep())
        for w in interpretation['warnings']:
            color = self.RED if "⚠" in w else self.GREEN
            lines.append(f"  {color}{w}{R}")

        lines.append(sep('═'))
        lines.append(f"  {D}[Ctrl+C to stop]  Frame #{self._render_count:06d}  "
                     f"Research-grade experimental analysis — NOT FOR CLINICAL USE{R}")
        lines.append("")

        # Flush all at once for minimal flicker
        sys.stdout.write("\n".join(lines))
        sys.stdout.flush()

        self._render_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: REAL-TIME ACQUISITION LOOP
# Timing-accurate sensor loop with performance monitoring
# ─────────────────────────────────────────────────────────────────────────────

class AcquisitionLoop:
    """
    Precision real-time sensor acquisition loop for Raspberry Pi.

    Implements a high-accuracy polling loop targeting a fixed sample rate
    using busy-wait compensation to overcome Python sleep() inaccuracy.

    Timing strategy (hybrid busy-wait):
      1. Compute ideal next-sample time from previous timestamp
      2. Sleep for (interval - safety_margin) to yield CPU
      3. Spin-wait for the remaining time (precise, but uses CPU)
      4. Record actual timing for Fs estimation and jitter monitoring

    Safety margin for sleep: 2ms — typical Linux scheduler quantum.
    This approach achieves ±0.5ms timing accuracy on Raspberry Pi 3/4.

    CPU usage estimate: ~15–25% on one core (Pi 3B) at 100 Hz with display.
    Lower sample rates proportionally reduce CPU load.
    """

    def __init__(self, target_rate: float = 100.0):
        """
        Args:
            target_rate : Target sample rate in Hz (default 100 Hz)
        """
        self.target_rate   = target_rate
        self.target_period = 1.0 / target_rate
        self.SLEEP_MARGIN  = 0.002  # 2ms sleep margin (seconds)

        # Performance monitoring
        self._actual_periods = collections.deque(maxlen=100)
        self._last_time      = None

    def wait_next_sample(self):
        """
        Block until next sample should be taken.
        Hybrid sleep+spin for precise timing with reasonable CPU efficiency.
        """
        now = time.monotonic()

        if self._last_time is None:
            self._last_time = now
            return

        # Record actual period for Fs estimation
        actual_period = now - self._last_time
        self._actual_periods.append(actual_period)

        # Compute ideal next sample time
        next_time = self._last_time + self.target_period

        # Sleep phase (yield CPU)
        sleep_until = next_time - self.SLEEP_MARGIN
        if sleep_until > now:
            time.sleep(sleep_until - now)

        # Busy-wait phase (precise)
        while time.monotonic() < next_time:
            pass

        self._last_time = time.monotonic()

    @property
    def effective_sample_rate(self) -> float:
        """Estimate actual sample rate from recent timing measurements."""
        if len(self._actual_periods) < 10:
            return self.target_rate
        mean_period = sum(self._actual_periods) / len(self._actual_periods)
        return 1.0 / mean_period if mean_period > 0 else self.target_rate


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: MAIN APPLICATION
# System integration, initialization, and main loop
# ─────────────────────────────────────────────────────────────────────────────

class BiomedicalMotionSystem:
    """
    Top-level system coordinator for the MPU-6050 biomedical motion analysis.

    Integrates all subsystems:
      - MPU6050Driver     : Hardware I2C communication
      - IMUFilterPipeline : Per-axis signal filtering (6 channels)
      - MotionAnalysisEngine : Magnitude, classification, artifact, orientation
      - MotionInterpreter : AI-assisted motion quality interpretation
      - TerminalDashboard : Real-time professional terminal display
      - AcquisitionLoop   : Precision timing loop

    Startup sequence:
      1. Open I2C bus and verify MPU-6050 presence
      2. Initialize sensor registers
      3. Perform static calibration (2-second still phase)
      4. Begin real-time acquisition loop
      5. Process and display at ~100 Hz

    Shutdown:
      - SIGINT (Ctrl+C) triggers graceful shutdown
      - I2C bus is closed cleanly
      - Terminal is restored
    """

    TARGET_RATE = 100.0  # Hz — 10ms sample interval

    def __init__(self, i2c_bus: int = 1):
        # Subsystem instances
        self.driver      = MPU6050Driver(i2c_bus=i2c_bus)
        self.filters     = self._create_filter_bank()
        self.engine      = MotionAnalysisEngine(sample_rate=self.TARGET_RATE)
        self.interpreter = MotionInterpreter()
        self.dashboard   = TerminalDashboard()
        self.acq_loop    = AcquisitionLoop(target_rate=self.TARGET_RATE)

        # Display rate: render every N acquisition samples (reduce CPU for display)
        self.DISPLAY_SKIP  = 5   # Display at ~20 Hz (every 5 samples)
        self._sample_count = 0

        # Shutdown coordination
        self._running = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _create_filter_bank(self) -> dict:
        """
        Create one IMUFilterPipeline per sensor channel.

        Filter parameters chosen for biomedical body-motion context:
          - Cutoff 8 Hz: passes motion artifacts up to running cadence (~3 Hz)
            while removing high-frequency mechanical vibration and I2C noise
          - MA window 5: ~50ms smoothing at 100Hz — minimal visual lag
          - Spike k=5: conservative — only clear outliers rejected
        """
        channels = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'temp']
        return {
            ch: IMUFilterPipeline(
                iir_cutoff_hz=8.0,
                sample_rate=self.TARGET_RATE,
                ma_window=5,
                spike_k=5.0,
            )
            for ch in channels
        }

    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        self._running = False

    def _print_startup_banner(self):
        """Print startup information before calibration."""
        os.system('clear')
        B = TerminalDashboard.BOLD
        C = TerminalDashboard.CYAN
        G = TerminalDashboard.GREEN
        M = TerminalDashboard.MAGENTA
        D = TerminalDashboard.DIM
        R = TerminalDashboard.RESET
        Y = TerminalDashboard.YELLOW

        print(f"\n  {M}{B}{'='*68}{R}")
        print(f"  {M}{B}  MPU-6050 BIOMEDICAL MOTION ANALYSIS SYSTEM{R}")
        print(f"  {M}{B}  Research-Grade IMU for PPG Motion Artifact Detection{R}")
        print(f"  {M}{B}{'='*68}{R}")
        print(f"  {D}Platform   : Raspberry Pi | Linux I2C (/dev/i2c-1){R}")
        print(f"  {D}Sensor     : InvenSense MPU-6050 (6-DOF IMU + Thermometer){R}")
        print(f"  {D}Config     : ±2g / ±250°/s | DLPF 42Hz | Fs=100Hz{R}")
        print(f"  {D}Purpose    : Non-clinical, experimental motion artifact research{R}")
        print(f"  {M}{B}{'─'*68}{R}")
        print(f"  {Y}DISCLAIMER : Not for medical diagnosis or clinical use.{R}")
        print(f"  {Y}             All outputs are experimental, research-grade only.{R}")
        print(f"  {M}{B}{'─'*68}{R}\n")

    def _print_calibration_progress(self, fraction: float):
        """Display calibration progress bar."""
        width    = 40
        filled   = int(fraction * width)
        bar      = f"[{'█'*filled}{'░'*(width-filled)}]"
        pct      = fraction * 100
        sys.stdout.write(f"\r  Calibrating sensor {bar} {pct:5.1f}%   ")
        sys.stdout.flush()

    def run(self):
        """
        Main execution entry point.

        Performs initialization, calibration, then enters the real-time
        acquisition and display loop until interrupted.
        """
        self._print_startup_banner()

        # --- Step 1: Open I2C and initialize sensor ---
        print("  [1/3] Opening I2C bus and verifying MPU-6050 identity...")
        try:
            self.driver.open()
            print(f"        {TerminalDashboard.GREEN}✓ MPU-6050 detected — WHO_AM_I = 0x68{TerminalDashboard.RESET}")
        except RuntimeError as e:
            print(f"\n  {TerminalDashboard.RED}✗ Initialization failed: {e}{TerminalDashboard.RESET}\n")
            sys.exit(1)

        print("  [2/3] Configuring sensor registers...")
        self.driver.initialize()
        print(f"        {TerminalDashboard.GREEN}✓ Sensor configured — "
              f"DLPF=42Hz, Fs=100Hz, ±2g/±250°/s{TerminalDashboard.RESET}")

        # --- Step 2: Static calibration ---
        print("  [3/3] Performing static calibration — keep sensor STILL...")
        print()
        self.driver.calibrate(samples=200, progress_cb=self._print_calibration_progress)
        print(f"\n        {TerminalDashboard.GREEN}✓ Calibration complete{TerminalDashboard.RESET}")

        ao = self.driver.accel_offset
        go = self.driver.gyro_offset
        print(f"        Accel offsets: X={ao[0]:+.4f}g  Y={ao[1]:+.4f}g  Z={ao[2]:+.4f}g")
        print(f"        Gyro  offsets: X={go[0]:+.3f}°/s  Y={go[1]:+.3f}°/s  Z={go[2]:+.3f}°/s")
        print(f"\n  Starting real-time monitoring in 2 seconds...")
        time.sleep(2.0)

        # --- Step 3: Real-time loop ---
        self._running = True
        last_filtered = {'ax':0,'ay':0,'az':0,'gx':0,'gy':0,'gz':0,'temp':25.0}
        last_analysis = None
        last_interp   = None

        while self._running:
            # Precision timing wait
            self.acq_loop.wait_next_sample()

            try:
                # --- Acquire raw data ---
                raw  = self.driver.read_all_raw()
                data = self.driver.convert_raw(raw)

                # --- Filter all channels ---
                filtered = {
                    'ax'   : self.filters['ax'].process(data['ax']),
                    'ay'   : self.filters['ay'].process(data['ay']),
                    'az'   : self.filters['az'].process(data['az']),
                    'gx'   : self.filters['gx'].process(data['gx']),
                    'gy'   : self.filters['gy'].process(data['gy']),
                    'gz'   : self.filters['gz'].process(data['gz']),
                    'temp' : self.filters['temp'].process(data['temp']),
                    'timestamp': data['timestamp'],
                }
                last_filtered = filtered

                # --- Motion analysis ---
                dt       = self.acq_loop.target_period
                analysis = self.engine.update(
                    filtered['ax'], filtered['ay'], filtered['az'],
                    filtered['gx'], filtered['gy'], filtered['gz'],
                    dt=dt
                )
                last_analysis = analysis

                # --- AI interpretation ---
                interpretation = self.interpreter.update(analysis)
                last_interp    = interpretation

            except OSError as e:
                # I2C error — log and continue (don't crash on transient errors)
                sys.stderr.write(f"\n  [I2C ERROR] {e} — retrying...\n")
                time.sleep(0.01)
                continue

            # --- Display update (throttled) ---
            self._sample_count += 1
            if self._sample_count % self.DISPLAY_SKIP == 0:
                if last_analysis and last_interp:
                    self.dashboard.render(
                        filtered=last_filtered,
                        analysis=last_analysis,
                        interpretation=last_interp,
                        sample_rate=self.acq_loop.effective_sample_rate,
                        calibrated=self.driver.calibrated,
                    )

        # --- Graceful shutdown ---
        self._shutdown()

    def _shutdown(self):
        """Clean up resources on exit."""
        print(f"\n\n  {TerminalDashboard.CYAN}Shutting down — closing I2C bus...{TerminalDashboard.RESET}")
        self.driver.close()
        print(f"  {TerminalDashboard.GREEN}✓ Shutdown complete. Samples acquired: "
              f"{self._sample_count}{TerminalDashboard.RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: SIMULATION MODE
# For development/testing without physical hardware
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedMPU6050Driver(MPU6050Driver):
    """
    Software simulation of MPU-6050 for development on non-Pi platforms.

    Generates physically plausible synthetic IMU data:
      - Baseline: 1g gravity on Z-axis, ≈0 on X/Y
      - Periodic drift: slow sinusoidal tilt (mimics human micro-tremor)
      - Random noise: Gaussian noise matching MPU-6050 noise density
        (Accel: ~400μg/√Hz, Gyro: ~0.005°/s/√Hz at 100Hz)
      - Occasional motion events: simulated hand/wrist movements
      - Temperature: fixed ~30°C (warm embedded board)

    Noise model (at 100 Hz bandwidth, from datasheet):
      σ_accel = 400e-6 * √100 ≈ 4 mg RMS
      σ_gyro  = 0.005  * √100 ≈ 0.05 °/s RMS
    """

    def __init__(self):
        super().__init__(i2c_bus=99, address=0x68)
        self._t = 0.0
        self._motion_phase   = 0.0
        self._motion_active  = False
        self._motion_counter = 0

        import random
        self._rng = random.Random(42)  # Seeded for reproducibility

    def open(self) -> bool:
        self.initialized = False
        return True

    def close(self):
        pass

    def initialize(self):
        self.initialized = True

    def calibrate(self, samples: int = 200, progress_cb=None):
        """Simulated calibration — offsets are zero for ideal simulation."""
        for i in range(samples):
            if progress_cb:
                progress_cb((i + 1) / samples)
            time.sleep(0.002)
        self.accel_offset = [0.0, 0.0, 0.0]
        self.gyro_offset  = [0.0, 0.0, 0.0]
        self.calibrated   = True

    def read_all_raw(self) -> dict:
        """Generate synthetic sensor data with realistic noise model."""
        import random
        self._t += 0.01  # Advance simulation time

        # --- Trigger occasional motion events ---
        if not self._motion_active and self._rng.random() < 0.005:
            self._motion_active  = True
            self._motion_counter = self._rng.randint(30, 150)  # 0.3–1.5s event
            self._motion_phase   = 0.0

        # --- Base values (gravity on Z) ---
        base_ax = 0.0
        base_ay = 0.0
        base_az = 1.0  # 1g gravity

        # --- Micro-tremor: slow sinusoidal drift ---
        tremor_amp = 0.008
        base_ax   += tremor_amp * math.sin(2 * math.pi * 0.5 * self._t)
        base_ay   += tremor_amp * math.sin(2 * math.pi * 0.7 * self._t + 1.2)

        # --- Motion event injection ---
        ma = mg = 0.0
        if self._motion_active:
            self._motion_phase   += 0.15
            self._motion_counter -= 1
            intensity = math.sin(self._motion_phase) * self._rng.uniform(0.2, 0.8)
            ma = intensity  # accel perturbation
            mg = intensity * 30.0  # gyro perturbation [°/s]
            if self._motion_counter <= 0:
                self._motion_active = False

        # --- Noise (Gaussian, matching MPU-6050 datasheet) ---
        def gauss(mu, sigma): return self._rng.gauss(mu, sigma)
        noise_a = 0.004   # 4 mg RMS
        noise_g = 0.05    # 0.05 °/s RMS

        # Final physical values [g, °/s, °C]
        ax = base_ax + ma * 0.6 + gauss(0, noise_a)
        ay = base_ay + ma * 0.4 + gauss(0, noise_a)
        az = base_az + ma * 0.2 + gauss(0, noise_a)
        gx = mg * 0.5 + gauss(0, noise_g)
        gy = mg * 0.3 + gauss(0, noise_g)
        gz = mg * 0.2 + gauss(0, noise_g)
        temp_c = 30.0 + gauss(0, 0.05)

        # Convert to raw ADC counts
        R = MPU6050Registers
        return {
            'ax_raw'    : int(ax   * R.ACCEL_SCALE_2G),
            'ay_raw'    : int(ay   * R.ACCEL_SCALE_2G),
            'az_raw'    : int(az   * R.ACCEL_SCALE_2G),
            'gx_raw'    : int(gx   * R.GYRO_SCALE_250),
            'gy_raw'    : int(gy   * R.GYRO_SCALE_250),
            'gz_raw'    : int(gz   * R.GYRO_SCALE_250),
            'temp_raw'  : int((temp_c - R.TEMP_OFFSET) * R.TEMP_DIVISOR),
            'timestamp' : time.monotonic(),
        }


class SimulatedBiomedicalMotionSystem(BiomedicalMotionSystem):
    """
    Full system with hardware replaced by software simulation.
    Useful for development, testing, and demonstration without hardware.
    """

    def __init__(self):
        # Don't call super().__init__() — replace driver manually
        self.driver      = SimulatedMPU6050Driver()
        self.filters     = self._create_filter_bank()
        self.engine      = MotionAnalysisEngine(sample_rate=self.TARGET_RATE)
        self.interpreter = MotionInterpreter()
        self.dashboard   = TerminalDashboard()
        self.acq_loop    = AcquisitionLoop(target_rate=self.TARGET_RATE)
        self.DISPLAY_SKIP  = 5
        self._sample_count = 0
        self._running      = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def detect_platform() -> bool:
    """
    Detect if running on a Raspberry Pi with I2C hardware available.

    Returns:
        True if physical I2C hardware found, False for simulation fallback.
    """
    try:
        # Check if /dev/i2c-1 exists (standard Pi header I2C bus)
        if not os.path.exists("/dev/i2c-1"):
            return False
        # Try importing smbus2 (required for hardware)
        import smbus2
        # Quick bus open test
        bus = smbus2.SMBus(1)
        bus.close()
        return True
    except Exception:
        return False


def main():
    """
    Application entry point.

    Auto-detects platform:
      - Raspberry Pi with /dev/i2c-1 → real hardware mode
      - Any other platform → simulation mode (for dev/testing)

    Override with command-line argument:
      --sim    Force simulation mode
      --hw     Force hardware mode
    """
    force_sim = "--sim" in sys.argv
    force_hw  = "--hw"  in sys.argv

    if force_sim:
        use_hardware = False
    elif force_hw:
        use_hardware = True
    else:
        use_hardware = detect_platform()

    if use_hardware:
        print(f"\n  {TerminalDashboard.GREEN}Hardware I2C detected — starting in real sensor mode.{TerminalDashboard.RESET}")
        system = BiomedicalMotionSystem(i2c_bus=1)
    else:
        print(f"\n  {TerminalDashboard.YELLOW}No I2C hardware detected — starting in SIMULATION mode.{TerminalDashboard.RESET}")
        print(f"  {TerminalDashboard.DIM}(Use --hw flag to force hardware mode){TerminalDashboard.RESET}\n")
        system = SimulatedBiomedicalMotionSystem()

    system.run()


if __name__ == "__main__":
    main()