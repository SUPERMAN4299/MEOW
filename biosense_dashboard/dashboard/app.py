"""
BioSense-Pi  —  Real-Time Biomedical Monitoring Dashboard  v1.0
===============================================================
Streamlit frontend integrating MAX30102, MPU-6050, and NoIR camera subsystems.
Designed for Raspberry Pi 4/5 with a dark biomedical research UI.

Usage:
    streamlit run dashboard/app.py --server.headless true --server.port 8501
"""

import sys
import os
import time
import queue
import threading
import logging
from pathlib import Path

import numpy as np
import streamlit as st

# ── Path setup: allow imports from project root ───────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Internal modules ──────────────────────────────────────────────────────────
from utils.data_hub   import DataHub
from utils.sensor_mgr import SensorManager
from components.header     import render_header
from components.metric_cards import render_metric_cards
from components.camera_panel import render_camera_panel
from components.system_panel import render_system_panel
from components.ai_panel     import render_ai_panel
from graphs.ppg_graph        import render_ppg_graph
from graphs.imu_graph        import render_imu_graph
from graphs.optical_graph    import render_optical_graph

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("biosense.app")

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioSense-Pi | Biomedical Monitor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "BioSense-Pi — Real-Time Biomedical Monitoring Dashboard v1.0",
    },
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
CSS_PATH = Path(__file__).parent / "styles" / "main.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# ── Session-state: shared DataHub + SensorManager (created once per session) ─
if "hub" not in st.session_state:
    st.session_state.hub = DataHub()
    st.session_state.sensor_mgr = SensorManager(st.session_state.hub)
    st.session_state.sensor_mgr.start_all()
    log.info("Session initialised — SensorManager started.")

hub: DataHub = st.session_state.hub
mgr: SensorManager = st.session_state.sensor_mgr

# ── Auto-refresh (Streamlit reruns the script every N ms) ─────────────────────
# 250 ms  →  4 Hz visual refresh (Raspberry Pi–friendly)
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    st_autorefresh(interval=250, limit=None, key="auto_refresh")
except ImportError:
    # Fallback: use st.empty + time.sleep rerun pattern
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.monotonic()
    elapsed = time.monotonic() - st.session_state.last_refresh
    if elapsed < 0.25:
        time.sleep(max(0.0, 0.25 - elapsed))
    st.session_state.last_refresh = time.monotonic()
    st.rerun()

# ── Read current snapshot ─────────────────────────────────────────────────────
snap = hub.snapshot()

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
render_header(snap, mgr)

st.markdown("---")

# ── Row 1: Camera + Metric Cards ──────────────────────────────────────────────
col_cam, col_metrics = st.columns([1, 1], gap="medium")

with col_cam:
    render_camera_panel(hub)

with col_metrics:
    render_metric_cards(snap)

st.markdown("---")

# ── Row 2: Real-time PPG graph ────────────────────────────────────────────────
with st.container():
    st.markdown("### 🫀 Real-Time PPG Signal  *(MAX30102)*")
    render_ppg_graph(hub)

# ── Row 3: IMU graphs ─────────────────────────────────────────────────────────
with st.container():
    st.markdown("### 📡 Inertial Sensor  *(MPU-6050)*")
    render_imu_graph(hub)

# ── Row 4: Optical Signal Analysis ───────────────────────────────────────────
with st.container():
    st.markdown("### 🔬 Optical Signal Analysis  *(NoIR Camera iPPG)*")
    render_optical_graph(hub)

# ── Row 5: System diagnostics + AI ───────────────────────────────────────────
col_sys, col_ai = st.columns([1, 1], gap="medium")

with col_sys:
    render_system_panel(snap, mgr)

with col_ai:
    render_ai_panel(snap)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#3a4a5c;font-size:0.75rem;padding:1rem 0'>"
    "BioSense-Pi v1.0 — Research Platform — "
    "<span style='color:#e74c3c'>⚠ NOT A MEDICAL DEVICE — NOT FOR CLINICAL USE</span>"
    "</div>",
    unsafe_allow_html=True,
)
