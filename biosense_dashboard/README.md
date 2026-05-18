# BioSense-Pi — Real-Time Biomedical Monitoring Dashboard

A professional research-grade Streamlit dashboard integrating three hardware subsystems:

| Subsystem | Chip | Interface | Rate |
|-----------|------|-----------|------|
| PPG / SpO₂ | MAX30102 | I²C 0x57 | 100 Hz |
| IMU / Motion | MPU-6050 | I²C 0x68 | 100 Hz |
| Optical iPPG | NoIR Camera | CSI | 30 Hz |

> ⚠ **NOT A MEDICAL DEVICE.** All outputs are for research only.

---

## Project Structure

```
biosense_dashboard/
├── dashboard/
│   ├── app.py                  ← Streamlit entry point
│   ├── components/
│   │   ├── header.py           ← Status bar
│   │   ├── metric_cards.py     ← Live BPM, SpO2, motion cards
│   │   ├── camera_panel.py     ← NoIR camera preview + IR false-colour
│   │   ├── system_panel.py     ← CPU/RAM/sensor health
│   │   └── ai_panel.py         ← AI integration placeholders
│   ├── graphs/
│   │   ├── ppg_graph.py        ← Real-time PPG waveform (Plotly)
│   │   ├── imu_graph.py        ← Accelerometer + gyroscope graphs
│   │   └── optical_graph.py    ← iPPG optical signal analysis
│   ├── utils/
│   │   ├── data_hub.py         ← Thread-safe shared state bus
│   │   ├── sensor_mgr.py       ← Acquisition thread lifecycle manager
│   │   └── formatting.py       ← Biomedical value formatters
│   └── styles/
│       └── main.css            ← Dark biomedical UI theme
├── max30102_subsystem.py       ← MAX30102 driver (existing)
├── mpu6050_subsystem.py        ← MPU-6050 driver (existing)
├── noir_camera_subsystem.py    ← NoIR camera driver (existing)
├── requirements.txt
└── .streamlit/config.toml      ← Dark theme + server config
```

---

## Setup

### 1. Install dependencies

```bash
# On Raspberry Pi or development machine
pip install -r requirements.txt

# Hardware-specific (Raspberry Pi only)
pip install smbus2 picamera2 opencv-python-headless
```

### 2. Enable I²C on Raspberry Pi

```bash
sudo raspi-config
# Interface Options → I2C → Enable

# Verify sensors are detected
i2cdetect -y 1
# Expected: 0x57 (MAX30102), 0x68 (MPU-6050)
```

### 3. Enable fast I²C (recommended)

```bash
# /boot/firmware/config.txt
dtparam=i2c_arm=on
dtparam=i2c_arm_baudrate=400000
```

### 4. Copy subsystem files

Place the three subsystem files in the project root (same level as `dashboard/`):
- `max30102_subsystem.py`
- `mpu6050_subsystem.py`
- `noir_camera_subsystem.py`

### 5. Run the dashboard

```bash
cd biosense_dashboard
streamlit run dashboard/app.py
```

Open in browser: `http://localhost:8501`  
On Raspberry Pi over network: `http://<pi-ip>:8501`

---

## Simulation Mode

When hardware is unavailable (development machine), all three subsystems
automatically fall back to built-in synthetic signal generators:

- **PPG sim**: synthetic 66 BPM cardiac waveform with harmonics
- **IMU sim**: 7 Hz physiological tremor + gravity
- **Camera sim**: Gaussian IR blob with 1.1 Hz pulsatile modulation

No code changes required — detection is automatic.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit UI Thread (4 Hz rerun via auto-refresh)      │
│  app.py → components/* + graphs/*                       │
│               ↑  snapshot() / get_buffer()              │
│           DataHub (thread-safe ring buffers)            │
│      ↑ update_snapshot()   ↑ push_*_sample()            │
├──────────────────┬──────────────────┬───────────────────┤
│ acq-ppg          │ acq-imu          │ acq-camera        │
│ worker_acq_ppg() │ worker_acq_imu() │ worker_acq_camera │
│ MAX30102 100 Hz  │ MPU-6050 100 Hz  │ NoIR 30 Hz        │
└──────────────────┴──────────────────┴───────────────────┘
```

All acquisition threads run as daemon threads managed by `SensorManager`.
Failed threads are automatically restarted every 5 seconds.

---

## Performance (Raspberry Pi 4)

| Component | CPU (idle sensor) |
|-----------|------------------|
| PPG thread | ~1% |
| IMU thread | ~1% |
| Camera thread | ~4% |
| Streamlit UI | ~3% (4 Hz) |
| **Total** | **~9%** |

---

## Adding AI Modules

The `ai_panel.py` component is designed as an integration point.
To add a real model:

```python
# In ai_panel.py or a new module:
from my_model import AnomalyDetector

detector = AnomalyDetector.load("models/anomaly_v1.pt")

def get_anomaly_score(snap: SensorSnapshot) -> float:
    features = extract_features(snap)
    return detector.predict(features)
```

Then replace the mock `anomaly_score` computation in `render_ai_panel()`.

---

## Disclaimer

This system is a **research prototype** and does **not** constitute a medical
device under EU MDR, FDA 21 CFR Part 11, or any other regulatory framework.
All biometric outputs (BPM, SpO₂, iPPG, tremor analysis) are experimental
values that have **not been clinically validated**.

Do not use for patient monitoring, clinical decision-making, or any
application where human health or safety could be affected.
