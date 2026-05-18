# BioSense-Pi — Real-Time Biomedical Monitoring System

**A research-grade multi-modal biomedical sensing platform for Raspberry Pi 4/5**

BioSense-Pi is a sophisticated distributed-acquisition and real-time signal processing system integrating three independent sensor subsystems (optical photoplethysmography, inertial measurement, and infrared imaging) with advanced DSP pipelines and a professional web-based monitoring dashboard.

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Hardware Architecture](#hardware-architecture)
3. [Software Architecture](#software-architecture)
4. [Sensor Subsystems](#sensor-subsystems)
5. [Dashboard & Visualization](#dashboard--visualization)
6. [Installation & Setup](#installation--setup)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [Key Features](#key-features)
10. [Disclaimer](#disclaimer)

---

## System Overview

BioSense-Pi is a **distributed, multi-threaded biomedical acquisition platform** designed to simultaneously capture, process, and visualize three complementary biomedical signals:

| **Signal** | **Modality** | **Chip** | **I²C Addr** | **Rate** | **Output Metrics** |
|-----------|-----------|---------|-------------|---------|-------------------|
| **Photoplethysmography (PPG)** | Transmissive optical | MAX30102 | 0x57 | 100 Hz | Heart Rate (BPM), SpO₂ (%), Perfusion Index (PI), Signal Quality |
| **Inertial Motion** | 6-DoF IMU | MPU-6050 | 0x68/0x69 | 100 Hz | Accelerometer (g), Gyroscope (°/s), Tremor, Motion Severity, Temperature |
| **Imaging PPG (iPPG)** | Remote optical (IR) | NoIR Camera v2/v3 | CSI | 30 Hz | Heartbeat (BPM), Signal SNR, Frame Quality, Tissue Segmentation |

**Real-time Dashboard** powered by Streamlit provides live visualization of all three modalities with per-frame quality metrics, health scoring, and system diagnostics.

---

## Hardware Architecture

### Target Platform
- **Primary:** Raspberry Pi 4 Model B (4 GB RAM recommended) or **Raspberry Pi 5**
- **OS:** Raspberry Pi OS Bookworm (64-bit) or Ubuntu 22.04 LTS
- **Python:** 3.11+ (Bookworm default)

### Sensor Hardware Stack

#### 1. **MAX30102 PPG Sensor**
- **Interface:** I²C (0x57), 400 kHz standard clock
- **Specs:**
  - Dual-wavelength LEDs (Red 660 nm + IR 880 nm)
  - 18-bit ADC resolution @ 100 Hz native sampling
  - Onboard ambient subtraction
  - Die temperature sensor (±1°C accuracy)
  - Automatic LED current control via ALC register

#### 2. **MPU-6050 Inertial Measurement Unit**
- **Interface:** I²C (0x68 or 0x69 via AD0 pin), 400 kHz
- **Specs:**
  - 3-axis Accelerometer (±2g to ±16g selectable, 16-bit)
  - 3-axis Gyroscope (±250°/s to ±2000°/s selectable, 16-bit)
  - ±1°C integrated die thermometer
  - Digital Low-Pass Filter (DLPF) @ 3 dB cutoff configurable 5–250 Hz
  - Burst-read capable (14-byte transaction → accel + temp + gyro)

#### 3. **NoIR Camera Module**
- **Variants:** OV5647 (v2, legacy) or **IMX219** (v3, recommended)
- **Specs:**
  - 8 MP resolution (3280 × 2464)
  - Operates at 850 nm (IR-transparent, no IR filter)
  - Supported frame rates: 30 fps @ 480p (default), 60 fps @ 240p
  - CSI-2 interface (dual lane, 4 Gbps max throughput)
  - Quantum efficiency peak in red channel at 850 nm

#### 4. **IR LED Illuminator (Optional)**
- 850 nm IR LED array, GPIO-controlled PWM dimming
- Typical current: 50–100 mA @ 3.3V
- Improves signal-to-noise ratio in low-light environments
- Reference: [LED placement in enclosure](./3d/box_with_lid.step)

### I²C Bus Configuration
```
┌─────────────────────────────────┐
│    Raspberry Pi GPIO Header      │
├──────────────────────────────────┤
│  GPIO 2  (SDA) ─────┬──→ MAX30102 (0x57)
│  GPIO 3  (SCL) ─────┼──→ MPU-6050 (0x68)
│  GPIO 4  (GND) ─────┤
│  (3.3V via pin 1/17)│
└──────────────────────┘
```

Pull-up resistors (≈4.7 kΩ) required on SDA/SCL to 3.3V if not on breakout board.

---

## Software Architecture

### Core State Machine

```
    ┌─────────────────────────────────────────────────────┐
    │              BioSense-Pi State Machine               │
    └─────────────────────────────────────────────────────┘
    
    INIT
      │ (Verify GPIO, load calibration)
      ↓
    SENSOR_CHECK
      │ (WHO_AM_I, I²C probe, register validation)
      ├─→ FAIL → ERROR_STATE
      ↓
    CALIBRATION
      │ (Gyro bias estimation, accel gravity alignment, white-balance)
      ├─→ FAIL → ERROR_STATE
      ↓
    RUNNING ←──────────────────────────────┐
      │ (Acquisition + DSP threads active)  │
      │ (Streamlit dashboard ingests data)  │
      ├─→ DEGRADED_MODE (1+ sensors unhealthy)
      │                      │
      ↓                      ↓
    ERROR_RECOVERY ← ← ← ← ←
      │ (Attempt reconnect, re-init failed sensor)
      │
      ├─→ SUCCESS → RUNNING
      ├─→ FAIL → SHUTDOWN
      ↓
    SHUTDOWN
      (Graceful thread join, GPIO cleanup)
```

### Thread Model

BioSense-Pi uses **daemon threads** for acquisition and processing, with a **main supervisor thread** managing state transitions and watchdog monitoring:

| **Thread Name** | **Priority** | **Rate** | **Role** |
|---|---|---|---|
| `main` | Normal | — | State machine orchestrator, watchdog supervisor |
| `acq-camera` | High | 30 Hz | Picamera2 frame acquisition (CSI DMA) |
| `acq-ppg` | High | 100 Hz | MAX30102 I²C FIFO polling |
| `acq-imu` | High | 100 Hz | MPU-6050 I²C burst polling |
| `proc-optical` | Normal | 30 Hz | IR image processing, CLAHE, tissue segmentation |
| `proc-biosignal` | Normal | 100 Hz | PPG DSP (BW filtering, AC/DC ratio, peak detection) |
| `proc-motion` | Normal | 100 Hz | IMU DSP (tremor analysis, artifact scoring) |
| `watchdog` | High | 1 Hz | Thread health monitoring, deadlock detection |
| `display` | Low | 4 Hz | Terminal dashboard (Curses) + OpenCV overlay |

All non-main threads are **daemon threads**, so KeyboardInterrupt or SIGTERM cleanly terminates the process.

### Thread Safety Model

- **Queues (thread-safe):** `queue.Queue` for inter-thread communication (no locks needed)
- **Ring Buffers:** Lock-free bounded circular stores with atomic head/tail pointers
- **Shared State:** Read-only snapshots via atomic reference swaps (minimal lock contention)

---

## Sensor Subsystems

### 1. MAX30102 PPG Subsystem (`max30102_subsystem.py`)

**Purpose:** Real-time heart rate and arterial oxygen saturation estimation via transmissive PPG

#### Components

| Class | Role |
|-------|------|
| `MAX30102Driver` | Low-level I²C register I/O, FIFO management, LED control |
| `AmbientBaselineEstimator` | Adaptive ambient offset compensation (100 Hz running mean) |
| `LEDAutoTuner` | Closed-loop PID controller: optimizes RED + IR LED current for max dynamic range |
| `FingerDetector` | Multi-criteria contact gating: DC level + AC magnitude + smoothness |
| `PPGRingBuffer` | 5-second circular sample history (500 samples @ 100 Hz) |
| `DSPPipeline` | IIR Butterworth filter chain (HP 0.5 Hz, LP 4 Hz), Welch PSD, AC/DC extraction |
| `AdaptivePeakDetector` | Refractory gating + amplitude-adaptive threshold for R-peak detection |
| `BiometricsEngine` | BPM calculation, SpO₂ estimation (empirical calibration), PI, Signal Quality Index (SQI) |
| `SensorHealthScorer` | Per-cycle hardware diagnostics (LED saturation, FIFO overflow, temperature drift) |

#### Register Map (MAX30102)

```
0x00  INT_STATUS1        → FIFO almost full, PPG ready, ALC overflow
0x02  INT_ENABLE1        ← Enable interrupts
0x04  FIFO_WR_PTR        → Current write pointer (0–31)
0x06  FIFO_RD_PTR        ← Read pointer (host updates)
0x07  FIFO_DATA          → 3×(RED[18:0] + IR[18:0]) per read
0x08  FIFO_CONFIG        ← SMP_AVE, FIFO_ROLLOVER, FIFO_A_FULL
0x09  MODE_CONFIG        ← SPO2 mode (0x03), power management
0x0A  SPO2_CONFIG        ← ADC range, sample rate, LED pulse width
0x0C  LED1_PA            ← RED LED pulse amplitude (0x00–0xFF)
0x0D  LED2_PA            ← IR LED pulse amplitude (0x00–0xFF)
0x1F  DIE_TEMP           → Temperature (signed int8, ±0.0625°C LSB)
```

#### Data Processing Pipeline

```
Raw FIFO (RED, IR) @ 100 Hz
    ↓
[Ambient Baseline Compensation]
    ↓
[LED Auto-Tuner (PID)] → Optimize dynamic range
    ↓
[Finger Detection Gate]  → Reject low-contact cycles
    ↓
[IIR Butterworth Chain]  → HP 0.5 Hz, LP 4 Hz
    ↓
[AC/DC Extraction]       → Welch PSD, RMS envelopes
    ↓
[Adaptive Peak Detector] → R-peak location ± 200 ms refractory
    ↓
[Biometrics Engine]      → BPM, SpO₂, PI, SQI
    ↓
Metrics Queue → Dashboard
```

#### SpO₂ Calibration

SpO₂ = 110 − 25 × R, where R = (AC_red/DC_red) / (AC_ir/DC_ir)

*Note: Empirical 2-point calibration recommended for clinical accuracy; factory defaults for research use.*

---

### 2. MPU-6050 IMU Subsystem (`mpu6050_subsystem.py`)

**Purpose:** High-fidelity motion capture, artifact detection, and tremor characterization

#### Components

| Class | Role |
|-------|------|
| `MPU6050Driver` | Smbus2 register I/O, burst-read (14-byte accel+temp+gyro), DLPF tuning |
| `CalibrationEngine` | Gyro bias estimation (±200 samples @ rest), accel gravity alignment, temp correction |
| `IMURingBuffer` | 10-second history (1000 samples @ 100 Hz, <1 ms random access) |
| `DSPStage` | Cascaded IIR, gravity subtraction, motion-adaptive filtering |
| `TremorAnalyzer` | FFT-based (0–10 Hz), Welch PSD peak detection, amplitude quantification |
| `MotionClassifier` | 4-tier severity scoring (still < light < moderate < vigorous) |
| `ArtifactRejector` | Statistical gating: flags PPG as unreliable during high-motion periods |
| `MotionConfidenceScorer` | Multi-criteria fusion metric for sensor fusion pipelines |

#### Register Map (MPU-6050)

```
0x19  SMPLRT_DIV         ← Sample Rate = 8000 Hz / (1 + SMPLRT_DIV)
0x1A  CONFIG             ← DLPF_CFG (0–6): 5 Hz to 250 Hz 3dB cutoff
0x1B  GYRO_CONFIG        ← FS_SEL: 00=±250°/s, 01=±500, 10=±1000, 11=±2000
0x1C  ACCEL_CONFIG       ← AFS_SEL: 00=±2g, 01=±4g, 10=±8g, 11=±16g
0x3B  ACCEL_XOUT_H (…)   → 14-byte burst from here (accel, temp, gyro)
0x41  TEMP_OUT_H         → Raw temperature (MSB), °C = raw/340 + 36.53
0x6B  PWR_MGMT_1         ← DEVICE_RESET, SLEEP, CLKSEL
0x75  WHO_AM_I           → 0x68 (readonly, ID validation)
```

#### Tremor Analysis

Raw accel → Gravity subtraction → IIR filter (0.5–10 Hz) → FFT (512-point, Hann window) → Peak detection

**Tremor Severity:**
```
Still:      rms_accel < 0.1 g
Light:      0.1 g ≤ rms_accel < 0.3 g
Moderate:   0.3 g ≤ rms_accel < 0.7 g
Vigorous:   rms_accel ≥ 0.7 g
```

---

### 3. NoIR Camera Subsystem (`noir_camera_subsystem.py`)

**Purpose:** Remote optical PPG (imaging PPG) via IR reflectance imaging

#### Optical Physics

At λ = 850 nm, dominant tissue chromophores are HbO₂, Hb, and melanin. The OV5647/IMX219 sensor achieves ≈30–40% quantum efficiency in the red channel vs. <5% in blue → use **R×0.75 + G×0.20 + B×0.05** for maximum SNR.

The cardiac cycle modulates reflected IR intensity by ~0.5–2% (AC/DC ratio). Extraction requires aggressive bandpass filtering and motion gating.

#### Components

| Class | Role |
|-------|------|
| `CameraHardwareDriver` | Picamera2 lifecycle, zero-copy DMA frames, resolution/FPS control |
| `AdaptiveExposureCtrl` | PI feedback controller: ROI histogram mean → exposure time tuning |
| `IRChannelExtractor` | Weighted BGR→IR blend (in-place), sensor-specific quantum-efficiency tuning |
| `CLAHEProcessor` | Contrast-limited adaptive histogram equalization per tile (8×8) |
| `GaussianPrefilter` | Separable Gaussian blur (σ=1.5 px, two-pass) for noise reduction |
| `AdaptiveTissueSegmenter` | Otsu thresholding + morphological ops + ellipse fitting → skin ROI |
| `MotionQualityGate` | Frame-diff median absolute deviation + Laplacian variance → reject blurry/fast-motion frames |
| `OpticalSignalBuffer` | 2-second ring buffer (60 frames @ 30 Hz) + Welch spectral analysis |
| `iPPGEngine` | Temporal AC extraction (1st derivative + bandpass 0.7–3 Hz) → BPM proxy |
| `FrameQualityScorer` | Multi-component confidence: tissue contact, blur, motion, dynamic range [0, 1] |

#### iPPG Data Flow

```
Raw RGB Frame (30 Hz) [480p]
    ↓
[Adaptive Exposure Control]  → Optimize brightness
    ↓
[IR Channel Extraction]      → R×0.75 + G×0.20 + B×0.05
    ↓
[CLAHE + Gaussian Prefilter] → Local contrast boost + noise reduction
    ↓
[Tissue Segmentation]        → Detect skin/face ROI
    ↓
[Motion Quality Gate]        → Reject blur + fast motion
    ↓
[Temporal AC Extraction]     → 1st derivative, bandpass 0.7–3 Hz
    ↓
[iPPG Engine]                → Welch PSD peak → BPM, SNR
    ↓
Frame Quality Score [0, 1]
    ↓
Optical Signal Queue → Dashboard
```

#### Scientific References

- Verkruysse et al. (2008) "Remote plethysmographic imaging" — *Opt. Express* 16(26)
- de Haan & Jeanne (2013) "Robust pulse-rate from chrominance-based rPPG" — *IEEE TBIOM*
- Wang et al. (2017) "Algorithmic principles of remote-PPG" — *IEEE J. Biomed. Health Inform.*
- Pilz et al. (2018) "Local group invariance for heart rate estimation from facial video"

---

## Dashboard & Visualization

### Streamlit Web Interface (`biosense_dashboard/dashboard/app.py`)

**Launch command:**
```bash
streamlit run biosense_dashboard/dashboard/app.py --server.headless true --server.port 8501
```

**URL:** `http://localhost:8501` (or `http://<pi-ip>:8501` from remote)

#### Dashboard Components

| Component | File | Content |
|-----------|------|---------|
| **Header** | `components/header.py` | Status indicators, system uptime, error alerts |
| **Metric Cards** | `components/metric_cards.py` | Live BPM, SpO₂ (%), PI, motion severity, temperature |
| **Camera Panel** | `components/camera_panel.py` | Real-time NoIR preview, IR false-color overlay, tissue segmentation mask |
| **System Panel** | `components/system_panel.py` | CPU %, RAM usage, I²C bus health, thread status, sensor diagnostics |
| **AI Panel** | `components/ai_panel.py` | Placeholder for ML-based anomaly detection / arrhythmia flags |
| **PPG Graph** | `graphs/ppg_graph.py` | Real-time PPG waveform (RED + IR), 10-second rolling window (Plotly) |
| **IMU Graph** | `graphs/imu_graph.py` | Accelerometer (g) + Gyroscope (°/s) traces, tremor spectrogram |
| **Optical Graph** | `graphs/optical_graph.py` | iPPG temporal signal, Welch PSD (0–5 Hz), frame quality timeline |

#### Data Flow (Dashboard)

```
SensorManager (spawns threads)
    ↓
[worker_acq_ppg] → queue_ppg
[worker_acq_imu] → queue_imu
[worker_acq_camera] → queue_camera
    ↓
DataHub (thread-safe snapshots)
    ↓
Streamlit app (re-run @ 4 Hz)
    ↓
[Render components]
    ↓
HTML/CSS/Plotly → Browser
```

#### Live Update Rate
- **Dashboard refresh:** 4 Hz (250 ms rerun)
- **Sensor acquisition:** 100 Hz (PPG, IMU), 30 Hz (Camera)
- **Graph scroll window:** 10 s (PPG/IMU), 30 s (optical)

#### Custom Styling (`styles/main.css`)
Dark biomedical research theme:
- Color scheme: Dark blue/purple background, neon cyan/green accents
- Mono-space fonts for numeric displays
- Responsive grid layout for Raspberry Pi 7" touchscreen

---

## Installation & Setup

### Prerequisites

- **Raspberry Pi 4 Model B+ or Raspberry Pi 5** (4 GB RAM minimum, 8 GB recommended)
- **Raspberry Pi OS Bookworm** (or Ubuntu 22.04 LTS) — 64-bit recommended
- **Python 3.11+**
- **SSH access** (or local terminal)

### Hardware Assembly

1. **I²C Connections (GPIO header):**
   - MAX30102: SDA→GPIO2, SCL→GPIO3, GND→GND, 3V3→3V3
   - MPU-6050: SDA→GPIO2, SCL→GPIO3, GND→GND, 3V3→3V3
   - Add 4.7 kΩ pull-ups to 3.3V if not on breakout boards

2. **Camera:**
   - NoIR Camera v2/v3 → CSI-2 ribbon to Camera port
   - Optional: 850 nm IR LED on GPIO pin (default GPIO17 PWM)

3. **Enclosure (Optional):**
   - See [`3d/box_with_lid.step`](./3d/box_with_lid.step) for Fusion 360 CAD

### Software Installation

#### 1. Clone Repository & Create Virtual Environment

```bash
cd /home/pi
git clone <repo-url> biosense-pi
cd biosense-pi

# Create Python 3.11 venv
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\Activate.ps1  # Windows PowerShell
```

#### 2. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel

# Core dependencies
pip install -r requirements.txt

# Typical stack:
# - numpy, scipy          (signal processing)
# - picamera2             (NoIR camera, Pi OS only)
# - smbus2                (I²C communication)
# - streamlit             (dashboard web framework)
# - plotly                (interactive graphs)
# - opencv-python-headless (image processing)
```

**Note:** `picamera2` only works on Raspberry Pi OS; desktop testing requires mock stubs.

#### 3. Enable I²C & Camera on Raspberry Pi

```bash
# Via raspi-config (interactive)
sudo raspi-config

# → Interface Options
# → I2C  → Enable
# → Camera → Enable (if using)

# Or via command-line:
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_camera 0
```

**Verify I²C:**
```bash
i2cdetect -y 1
```

Expected output:
```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         
10:                         
50:                         
60:        68                        # MPU-6050
70:  
```

(MAX30102 at 0x57 should appear if connected.)

---

## Usage

### Quick Start (Local Development)

```bash
cd /path/to/biosense-pi

# Activate venv
source venv/bin/activate

# Start the sensor acquisition + dashboard
python main.py
```

The system will:
1. Verify sensor connectivity (WHO_AM_I checks, I²C probing)
2. Perform calibration (gyro bias, gravity alignment)
3. Spawn acquisition threads
4. Start the Streamlit dashboard

### Raspberry Pi Deployment

```bash
# SSH into Pi
ssh pi@<pi-ip>

cd /home/pi/biosense-pi
source venv/bin/activate
python main.py
```

Access dashboard from workstation:
```
http://<pi-ip>:8501
```

### Testing Individual Sensors

#### Test MAX30102 (PPG)
```bash
cd Tess
python sensors_test/max30102_monitor.py
```

#### Test MPU-6050 (IMU)
```bash
python sensors_test/mpu6050.py
```

#### Test NoIR Camera
```bash
bash test_noir.sh
# or
python sensors_test/noir.py
```

#### Low-level I²C Verification
```bash
i2cdetect -y 1
i2cdump -y 1 0x57  # MAX30102 registers
i2cdump -y 1 0x68  # MPU-6050 registers
```

---

## Project Structure

```
biosense-pi/
├── main.py                        ← Main entry point (state machine + thread orchestration)
├── max30102_subsystem.py          ← PPG sensor acquisition & DSP
├── mpu6050_subsystem.py           ← IMU acquisition & tremor analysis
├── noir_camera_subsystem.py       ← Camera acquisition & iPPG pipeline
├── requirements.txt               ← Python dependencies
├── README.md                      ← This file
├── to-do.txt                      ← Development tasks
│
├── biosense_dashboard/
│   ├── README.md
│   └── dashboard/
│       ├── app.py                 ← Streamlit app entry point
│       ├── components/
│       │   ├── __init__.py
│       │   ├── header.py          ← Status bar + alerts
│       │   ├── metric_cards.py    ← BPM, SpO2, motion cards
│       │   ├── camera_panel.py    ← Live camera + IR false-color
│       │   ├── system_panel.py    ← CPU/RAM/I2C health
│       │   └── ai_panel.py        ← ML/anomaly detection stubs
│       ├── graphs/
│       │   ├── __init__.py
│       │   ├── ppg_graph.py       ← PPG waveform (Plotly)
│       │   ├── imu_graph.py       ← Accel/Gyro traces
│       │   └── optical_graph.py   ← iPPG signal + Welch PSD
│       ├── styles/
│       │   └── main.css           ← Dark biomedical theme
│       └── utils/
│           ├── __init__.py
│           ├── data_hub.py        ← Thread-safe data snapshots
│           ├── sensor_mgr.py      ← Thread lifecycle manager
│           └── formatting.py      ← Metric display utils
│
├── Tess/                          ← Test suite
│   ├── test_i2c_veri.sh          ← I²C verification
│   ├── test_noir.sh              ← Camera quick-check
│   └── sensors_test/
│       ├── max30102_monitor.py    ← Standalone PPG test
│       ├── mpu6050.py            ← Standalone IMU test
│       └── noir.py               ← Standalone camera test
│
├── 3d/                            ← Hardware enclosure CAD
│   ├── box_with_lid.ipt          ← Inventor Assembly
│   └── box_with_lid.step         ← STEP export (for 3D printing)
│
└── .git/                          ← Version control
```

---

## Key Features

### ✅ **Multi-Modal Biomedical Sensing**
- Simultaneous optical (PPG + iPPG), inertial (IMU), and thermal data streams
- Independent per-sensor acquisition at native rates (100 Hz PPG/IMU, 30 Hz camera)

### ✅ **Advanced Digital Signal Processing**
- IIR Butterworth filtering (HP 0.5 Hz, LP 4 Hz for PPG)
- Welch power spectral density analysis (Hann windows, 50% overlap)
- Adaptive peak detection with refractory gating
- Tremor & motion severity classification (4-tier scale)

### ✅ **Robust Sensor Fusion**
- LED auto-tuning (closed-loop PID for MAX30102 red/IR balance)
- Adaptive exposure control (Picamera2 histogram feedback)
- Finger contact detection (multi-criteria gating: DC, AC, smoothness)
- Motion artifact flagging (PPG signal unreliability during high motion)

### ✅ **Real-Time Dashboard**
- Live waveform visualization (Plotly interactive graphs)
- Per-frame quality metrics (tissue contact, blur, motion gates)
- System diagnostics (CPU, RAM, I²C bus health, thread status)
- 4 Hz responsive UI (Streamlit, Raspberry Pi–friendly)

### ✅ **Production-Grade Logging & Diagnostics**
- Structured JSON logging with timestamp + thread ID
- Per-cycle hardware health scoring
- Sensor self-diagnostics (temperature drift, I²C errors, FIFO overflow)
- Graceful degradation (continues operation with healthy sensors only)

### ✅ **Thread Safety & Performance**
- Lock-free ring buffers for high-frequency data
- Queue-based inter-thread communication (no shared mutable state)
- Daemon threads + graceful SIGTERM/KeyboardInterrupt handling
- Watchdog monitor (1 Hz) detects deadlocks and unresponsive threads

---

## Disclaimer

⚠️ **NOT A MEDICAL DEVICE**

BioSense-Pi is a **research prototype** designed for educational and investigational purposes only.

- All sensor outputs (heart rate, SpO₂, motion metrics) are **experimental**.
- No output constitutes **medical advice, clinical diagnosis, or treatment recommendation**.
- The system has **not been validated** against clinical gold standards.
- Do **not** use for medical decision-making or patient monitoring.
- SpO₂ calibration is empirical; clinical accuracy requires supervised multi-point calibration.
- iPPG (imaging PPG) is an emerging, unvalidated modality; treat results as research values only.

**Use at your own risk.** Consult with biomedical engineers and clinicians before any medical application.

---

## Troubleshooting

### I²C Devices Not Detected
```bash
# Verify bus voltage
sudo i2cdetect -y 1

# Check kernel modules
lsmod | grep i2c

# Enable I²C if needed
sudo raspi-config nonint do_i2c 0
sudo reboot
```

### Camera Not Detected
```bash
# Verify CSI ribbon connection (gold contacts face inward)
# Enable camera in raspi-config:
sudo raspi-config nonint do_camera 0
sudo reboot

# Test with:
libcamera-hello --fullscreen
```

### Streamlit Dashboard Won't Load
```bash
# Check port is free:
lsof -i :8501

# Kill any existing process:
kill -9 <PID>

# Restart:
streamlit run biosense_dashboard/dashboard/app.py
```

---

## Contributing & Future Work

See [`to-do.txt`](./to-do.txt) for planned enhancements:
- [ ] ONNX-based arrhythmia detection model
- [ ] PostgreSQL data logging backend
- [ ] REST API for external integrations
- [ ] Kalman filtering for sensor fusion
- [ ] Multi-person support (face detection + per-ROI tracking)

---

## License

[Specify your license here, e.g., MIT, GPL-3.0, Apache-2.0]

---

## Contact & Support

**Author:** BioSense-Pi Research Team  
**Repository:** [GitHub URL]  
**Issues:** [GitHub Issues] or Email  

For scientific collaboration or questions, contact the development team.