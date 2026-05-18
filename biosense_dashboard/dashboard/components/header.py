"""
components/header.py — Top header bar for BioSense-Pi dashboard.
"""
import time
import streamlit as st

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


def render_header(snap, mgr) -> None:
    """Render the top status bar with title, FPS, CPU, and sensor badges."""
    statuses = mgr.status()

    # Sensor connection badges
    def _badge(label: str, active: bool, sim: bool = False) -> str:
        if active:
            col = "#00e676"; txt = "LIVE"
        elif sim:
            col = "#ffeb3b"; txt = "SIM"
        else:
            col = "#f44336"; txt = "OFF"
        return (
            f"<span style='background:{col};color:#000;border-radius:6px;"
            f"padding:2px 8px;font-size:0.75rem;font-weight:700;margin:0 3px'>"
            f"{label}: {txt}</span>"
        )

    ppg_active    = statuses["ppg"].running
    imu_active    = statuses["imu"].running
    camera_active = statuses["camera"].running

    # CPU / RAM
    cpu_str = ram_str = "---"
    if _PSUTIL:
        try:
            cpu_str = f"{psutil.cpu_percent(interval=None):.0f}%"
            ram_str = f"{psutil.virtual_memory().percent:.0f}%"
        except Exception:
            pass

    fps = snap.camera_fps or 0.0
    ppg_fs = snap.ppg_effective_fs or 100.0

    badges = (
        _badge("PPG",    ppg_active,    sim=not statuses["ppg"].hw_active)
        + _badge("IMU",    imu_active,    sim=not statuses["imu"].hw_active)
        + _badge("CAMERA", camera_active, sim=not statuses["camera"].hw_active)
    )

    st.markdown(
        f"""
        <div style='
            background:linear-gradient(90deg,#0d1117,#161b22);
            border:1px solid #21262d;
            border-radius:12px;
            padding:14px 20px;
            display:flex;
            align-items:center;
            justify-content:space-between;
            margin-bottom:4px;
        '>
          <div>
            <span style='font-size:1.5rem;font-weight:800;color:#00e5ff;
                         letter-spacing:2px'>🫀 BIOSENSE-Pi</span>
            <span style='color:#546e7a;font-size:0.85rem;margin-left:12px'>
                Real-Time Biomedical Monitor v1.0</span>
          </div>
          <div style='display:flex;align-items:center;gap:16px;'>
            <span style='color:#546e7a;font-size:0.8rem'>
              CAM <b style='color:#80cbc4'>{fps:.1f}</b> fps &nbsp;|&nbsp;
              PPG <b style='color:#80cbc4'>{ppg_fs:.0f}</b> Hz &nbsp;|&nbsp;
              CPU <b style='color:#80cbc4'>{cpu_str}</b> &nbsp;|&nbsp;
              RAM <b style='color:#80cbc4'>{ram_str}</b>
            </span>
            <div>{badges}</div>
          </div>
        </div>
        <div style='text-align:right;color:#b71c1c;font-size:0.7rem;margin-bottom:4px;'>
          ⚠ RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE — NOT FOR CLINICAL USE
        </div>
        """,
        unsafe_allow_html=True,
    )
