"""
=============================================================================
MAX30102 Pulse Oximeter & Heart Rate Monitor — Raspberry Pi
=============================================================================
Sensor   : MAX30102 (Heart Rate + SpO2)
Protocol : I2C via smbus2 (NO third-party sensor lib required)
Target   : Raspberry Pi (any model with I2C)
Python   : 3.7+

Install dependencies (these ALL exist on PyPI, including Windows for dev):
    pip install smbus2 scipy numpy

Hardware wiring:
    MAX30102 VIN  → Raspberry Pi 3.3 V  (Pin 1)
    MAX30102 GND  → Raspberry Pi GND    (Pin 6)
    MAX30102 SDA  → Raspberry Pi GPIO2  (Pin 3)
    MAX30102 SCL  → Raspberry Pi GPIO3  (Pin 5)

Enable I2C on Raspberry Pi:
    sudo raspi-config → Interface Options → I2C → Enable → Reboot

Run on Pi:
    python max30102_monitor.py
=============================================================================
"""

import time
import sys
import numpy as np
from collections import deque

# ── Optional scipy for better bandpass filter ─────────────────────────────
try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARN] scipy not found — using moving-average filter only.")
    print("       Install with:  pip install scipy\n")

# ── smbus2 for raw I2C access ─────────────────────────────────────────────
try:
    import smbus2
except ImportError:
    print("[ERROR] smbus2 not found.  pip install smbus2")
    sys.exit(1)


# =============================================================================
#  CONFIGURATION  (edit these to tune behaviour)
# =============================================================================

I2C_BUS          = 1       # Raspberry Pi I2C bus (almost always 1)
SENSOR_ADDR      = 0x57    # MAX30102 fixed I2C address

# Sampling
SAMPLE_RATE      = 100     # Hz — written to sensor register
#                            Supported: 50, 100, 200, 400, 800, 1000

# Buffer / window sizes
CALC_BUFFER      = 100     # Samples collected per calculation cycle
MA_WINDOW        = 15      # Moving-average kernel width

# Butterworth bandpass (Hz) — heart rate 40-250 BPM → 0.67-4.17 Hz
BPF_LOW          = 0.5
BPF_HIGH         = 4.0
BPF_ORDER        = 3

# Finger / quality thresholds
FINGER_AC_THRESH = 5_000   # Peak-to-peak ADC counts → finger present
AC_DC_MIN        = 0.001   # AC/DC ratio lower bound  → too weak
AC_DC_MAX        = 0.50    # AC/DC ratio upper bound  → saturated / motion

# Display
PRINT_INTERVAL   = 1.0     # Seconds between console lines
DISPLAY_HISTORY  = 5       # Readings averaged for smooth display values


# =============================================================================
#  MAX30102 REGISTER MAP  (datasheet section 8.4)
# =============================================================================

REG_FIFO_WR_PTR   = 0x04
REG_OVF_COUNTER   = 0x05
REG_FIFO_RD_PTR   = 0x06
REG_FIFO_DATA     = 0x07   # 3 bytes RED + 3 bytes IR per sample

REG_FIFO_CONFIG   = 0x08
REG_MODE_CONFIG   = 0x09
REG_SPO2_CONFIG   = 0x0A
REG_LED1_PA       = 0x0C   # RED LED pulse amplitude
REG_LED2_PA       = 0x0D   # IR  LED pulse amplitude

REG_PART_ID       = 0xFF   # Should read 0x15 for MAX30102

# Bit patterns
MODE_SPO2         = 0x03   # RED + IR (SpO2 mode)
MODE_RESET        = 0x40   # Soft reset bit

SAMPLE_RATE_MAP = {        # SR Hz → SPO2_CONFIG[4:2] bits
    50:   0x00,
    100:  0x01,
    200:  0x02,
    400:  0x03,
    800:  0x04,
    1000: 0x05,
}

PULSE_WIDTH_18BIT = 0x03   # 411 µs → 18-bit ADC
ADC_RANGE_4096    = 0x20   # 4096 nA full scale


# =============================================================================
#  LOW-LEVEL I2C HELPERS
# =============================================================================

def reg_write(bus, reg: int, value: int) -> None:
    """Write one byte to a sensor register."""
    bus.write_byte_data(SENSOR_ADDR, reg, value)


def reg_read(bus, reg: int) -> int:
    """Read one byte from a sensor register."""
    return bus.read_byte_data(SENSOR_ADDR, reg)


def reg_read_block(bus, reg: int, length: int) -> list:
    """Read multiple bytes starting at reg."""
    return bus.read_i2c_block_data(SENSOR_ADDR, reg, length)


# =============================================================================
#  SENSOR INITIALISATION
# =============================================================================

def initialize_sensor(bus) -> bool:
    """
    Configure MAX30102 registers for SpO2 + Heart-rate mode.

    Steps:
      1. Verify Part ID (0x15 = MAX30102)
      2. Soft reset
      3. Configure FIFO, SpO2, LED amplitudes, operating mode
      4. Clear FIFO pointers

    Returns True on success, False otherwise.
    """
    # 1. Verify Part ID
    try:
        part_id = reg_read(bus, REG_PART_ID)
    except Exception as e:
        print(f"[ERROR] Cannot reach sensor at 0x{SENSOR_ADDR:02X} on bus {I2C_BUS}: {e}")
        print("        Check wiring and that I2C is enabled (raspi-config).")
        return False

    if part_id != 0x15:
        print(f"[ERROR] Unexpected Part ID: 0x{part_id:02X} (expected 0x15).")
        return False

    # 2. Soft reset — clears all registers to power-on state
    reg_write(bus, REG_MODE_CONFIG, MODE_RESET)
    time.sleep(0.1)

    # 3a. FIFO config: no HW averaging, rollover enabled, almost-full=17
    reg_write(bus, REG_FIFO_CONFIG, 0x4F)

    # 3b. SpO2 config: ADC range | sample rate | pulse width
    sr_bits     = SAMPLE_RATE_MAP.get(SAMPLE_RATE, 0x01)
    spo2_config = ADC_RANGE_4096 | (sr_bits << 2) | PULSE_WIDTH_18BIT
    reg_write(bus, REG_SPO2_CONFIG, spo2_config)

    # 3c. LED amplitudes — 0x24 ≈ 7.2 mA (range 0x00-0xFF = 0-51 mA)
    reg_write(bus, REG_LED1_PA, 0x24)   # RED
    reg_write(bus, REG_LED2_PA, 0x24)   # IR

    # 3d. Operating mode: SpO2 (RED + IR both active)
    reg_write(bus, REG_MODE_CONFIG, MODE_SPO2)

    # 4. Clear FIFO pointers
    reg_write(bus, REG_FIFO_WR_PTR, 0x00)
    reg_write(bus, REG_OVF_COUNTER, 0x00)
    reg_write(bus, REG_FIFO_RD_PTR, 0x00)

    print(f"[OK] MAX30102 ready  |  Part ID=0x15  |  SR={SAMPLE_RATE} Hz  |  18-bit ADC")
    return True


# =============================================================================
#  SAMPLE READING
# =============================================================================

def read_one_sample(bus) -> tuple:
    """
    Read one RED + IR sample from the FIFO data register.

    FIFO layout in SpO2 mode (6 bytes per sample):
        Bytes 0-2 : RED sample — 18 useful bits in bits[17:0]
        Bytes 3-5 : IR  sample — 18 useful bits in bits[17:0]

    Returns (red, ir) as integers.
    """
    try:
        raw = reg_read_block(bus, REG_FIFO_DATA, 6)
        red = ((raw[0] << 16) | (raw[1] << 8) | raw[2]) & 0x3FFFF
        ir  = ((raw[3] << 16) | (raw[4] << 8) | raw[5]) & 0x3FFFF
        return red, ir
    except Exception:
        return 0, 0


def collect_samples(bus, num: int = CALC_BUFFER):
    """
    Collect `num` RED/IR sample pairs paced to SAMPLE_RATE.

    Returns (ir_array, red_array) as numpy float64 arrays,
    or (None, None) if reading fails.
    """
    ir_buf, red_buf = [], []
    interval = 1.0 / SAMPLE_RATE

    try:
        for _ in range(num):
            red, ir = read_one_sample(bus)
            red_buf.append(red)
            ir_buf.append(ir)
            time.sleep(interval)
    except Exception as e:
        print(f"[ERROR] Sample collection interrupted: {e}")
        return None, None

    return (np.array(ir_buf,  dtype=np.float64),
            np.array(red_buf, dtype=np.float64))


# =============================================================================
#  SIGNAL FILTERING
# =============================================================================

def butter_bandpass(data: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter (scipy required)."""
    nyq  = 0.5 * fs
    b, a = butter(BPF_ORDER, [BPF_LOW / nyq, BPF_HIGH / nyq], btype="band")
    return filtfilt(b, a, data)


def moving_average(data: np.ndarray, window: int = MA_WINDOW) -> np.ndarray:
    """Box-kernel moving average — output length equals input length."""
    return np.convolve(data, np.ones(window) / window, mode="same")


def filter_signal(data: np.ndarray, fs: float) -> np.ndarray:
    """
    Full pipeline:
      1. Butterworth bandpass (eliminates baseline wander + high-freq noise)
      2. Moving-average smooth (reduces sample-to-sample jitter)
    """
    if SCIPY_AVAILABLE and len(data) >= 3 * BPF_ORDER + 1:
        try:
            data = butter_bandpass(data, fs)
        except Exception:
            pass
    return moving_average(data)


# =============================================================================
#  FINGER DETECTION
# =============================================================================

def finger_detected(ir: np.ndarray) -> bool:
    """
    A placed finger causes a large pulsatile (AC) swing on the IR channel.
    If peak-to-peak amplitude < FINGER_AC_THRESH → no finger / poor contact.
    """
    return bool(len(ir) and (ir.max() - ir.min()) > FINGER_AC_THRESH)


# =============================================================================
#  SIGNAL QUALITY
# =============================================================================

def signal_quality(ir: np.ndarray) -> str:
    """
    Assess PPG signal quality via AC/DC ratio (normalised pulsatile depth).

      ratio < AC_DC_MIN → flat line / no contact      → "NO SIGNAL"
      ratio > AC_DC_MAX → saturation / motion artifact → "POOR"
      otherwise                                         → "GOOD"
    """
    if not len(ir):
        return "NO SIGNAL"
    dc = np.mean(ir)
    if dc == 0:
        return "NO SIGNAL"
    ratio = (ir.max() - ir.min()) / 2.0 / dc
    if   ratio < AC_DC_MIN: return "NO SIGNAL"
    elif ratio > AC_DC_MAX: return "POOR"
    else:                   return "GOOD"


# =============================================================================
#  HEART RATE (BPM)
# =============================================================================

def calculate_bpm(ir: np.ndarray, fs: float):
    """
    Estimate heart rate from filtered IR PPG via peak detection.

    Algorithm:
      1. Filter signal (bandpass + smooth)
      2. Threshold = mean of filtered signal
      3. Find local maxima above threshold with minimum spacing
      4. Average inter-peak interval → BPM

    Returns BPM float (30-250 range) or None.
    """
    if len(ir) < 30:
        return None

    sig       = filter_signal(ir.copy(), fs)
    threshold = np.mean(sig)
    min_dist  = int(fs * 60 / 250)   # cap at 250 BPM

    peaks = []
    for i in range(1, len(sig) - 1):
        if sig[i] > threshold and sig[i] > sig[i-1] and sig[i] > sig[i+1]:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)

    if len(peaks) < 2:
        return None

    avg_interval_s = np.mean(np.diff(peaks)) / fs
    bpm = 60.0 / avg_interval_s

    return round(bpm, 1) if 30 <= bpm <= 250 else None


# =============================================================================
#  SpO2
# =============================================================================

def calculate_spo2(ir: np.ndarray, red: np.ndarray):
    """
    Estimate blood oxygen saturation (SpO2) via Ratio-of-Ratios (RoR):

        R    = (AC_red / DC_red) / (AC_ir / DC_ir)
        SpO2 ≈ 104 − 17 × R

    (Empirical calibration — real clinical devices use factory-measured curves.)
    Reference: Webster J.G., Design of Pulse Oximeters (1997).

    Returns SpO2 float clamped to [70, 100] %, or None.
    """
    if len(ir) < 10 or len(red) < 10:
        return None

    dc_ir, dc_red = np.mean(ir), np.mean(red)
    if dc_ir == 0 or dc_red == 0:
        return None

    ac_ir  = (ir.max()  - ir.min())  / 2.0
    ac_red = (red.max() - red.min()) / 2.0
    if ac_ir == 0:
        return None

    R    = (ac_red / dc_red) / (ac_ir / dc_ir)
    spo2 = 104.0 - 17.0 * R

    return round(max(70.0, min(100.0, spo2)), 1)


# =============================================================================
#  DISPLAY
# =============================================================================

def print_reading(bpm, spo2, quality: str, finger: bool) -> None:
    """Print one formatted real-time line to stdout."""
    ts     = time.strftime("%H:%M:%S")
    bpm_s  = f"{bpm:>6.1f} BPM"  if bpm  is not None else "  ---  BPM"
    spo2_s = f"{spo2:>5.1f} %"   if spo2 is not None else "  ---  %"
    icon   = {"GOOD": "●", "POOR": "◐", "NO SIGNAL": "○"}.get(quality, "?")
    warn   = "" if finger else "   ⚠  NO FINGER — place finger firmly on sensor"
    print(f"[{ts}]  ❤  {bpm_s}   🩸 SpO2: {spo2_s}   Signal: {icon} {quality:<10}{warn}")


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    print("=" * 68)
    print("  MAX30102 Real-Time Pulse Oximeter & Heart Rate — Raspberry Pi")
    print("=" * 68)
    print(f"  I2C bus     : {I2C_BUS}  (address 0x{SENSOR_ADDR:02X})")
    print(f"  Sample rate : {SAMPLE_RATE} Hz")
    print(f"  Buffer      : {CALC_BUFFER} samples per cycle (~{CALC_BUFFER/SAMPLE_RATE:.1f} s)")
    print(f"  Filter      : {'Butterworth BPF + Moving Average' if SCIPY_AVAILABLE else 'Moving Average only'}")
    print("=" * 68)
    print("  Place your finger firmly on the sensor.  Ctrl+C to quit.\n")

    # Open I2C bus
    try:
        bus = smbus2.SMBus(I2C_BUS)
    except Exception as e:
        print(f"[ERROR] Cannot open I2C bus {I2C_BUS}: {e}")
        print("        Run: sudo raspi-config → Interface Options → I2C → Enable")
        sys.exit(1)

    # Configure sensor registers
    if not initialize_sensor(bus):
        bus.close()
        sys.exit(1)

    # Rolling history for smoothed display values
    bpm_hist  = deque(maxlen=DISPLAY_HISTORY)
    spo2_hist = deque(maxlen=DISPLAY_HISTORY)
    last_print = 0.0

    try:
        while True:
            # ── Collect raw samples ────────────────────────────────────────
            ir, red = collect_samples(bus, CALC_BUFFER)

            if ir is None:
                print("[WARN] Sample read failed — retrying in 2 s...")
                time.sleep(2)
                continue

            # ── Assess signal ──────────────────────────────────────────────
            finger  = finger_detected(ir)
            quality = signal_quality(ir)
            bpm     = None
            spo2    = None

            # ── Calculate vitals only when signal is usable ────────────────
            if finger and quality == "GOOD":
                bpm  = calculate_bpm(ir, float(SAMPLE_RATE))
                spo2 = calculate_spo2(ir, red)
                if bpm  is not None: bpm_hist.append(bpm)
                if spo2 is not None: spo2_hist.append(spo2)

            # ── Smoothed display values ────────────────────────────────────
            show_bpm  = round(np.mean(bpm_hist),  1) if bpm_hist  else None
            show_spo2 = round(np.mean(spo2_hist), 1) if spo2_hist else None

            # ── Print at configured interval ───────────────────────────────
            now = time.time()
            if now - last_print >= PRINT_INTERVAL:
                print_reading(show_bpm, show_spo2, quality, finger)
                last_print = now

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    finally:
        try:
            reg_write(bus, REG_MODE_CONFIG, 0x80)   # SHDN bit — power down
            bus.close()
            print("[INFO] Sensor powered down. Bus closed.")
        except Exception:
            pass


if __name__ == "__main__":
    main()