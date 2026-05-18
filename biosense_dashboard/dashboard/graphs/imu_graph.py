"""
graphs/imu_graph.py — Real-time IMU signal graphs (MPU-6050).

Renders two sub-panels:
  Left  : Accelerometer X/Y/Z + magnitude
  Right : Gyroscope X/Y/Z + motion severity

Motion artifact zones are shaded when motion_state ≠ STABLE to
visually link the PPG validity metric with IMU excursions.
"""
import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

_DARK_BG    = "#0d1117"
_GRID_COL   = "#161b22"
_AXIS_COL   = "#546e7a"

_ACCEL_COLS = ["#00e5ff", "#f44336", "#69f0ae"]   # X, Y, Z
_GYRO_COLS  = ["#ffeb3b", "#ff7043", "#ce93d8"]    # X, Y, Z


def _make_layout(title: str, ytitle: str) -> dict:
    return dict(
        paper_bgcolor=_DARK_BG,
        plot_bgcolor =_DARK_BG,
        font         =dict(color="#90a4ae", size=10),
        margin       =dict(l=40, r=10, t=30, b=40),
        height       =220,
        title        =dict(text=title, font=dict(color="#80cbc4", size=12),
                           x=0.01, y=0.97),
        xaxis        =dict(title="Time (s)", color=_AXIS_COL,
                           gridcolor=_GRID_COL, zerolinecolor=_GRID_COL,
                           tickfont=dict(color=_AXIS_COL)),
        yaxis        =dict(title=ytitle, color=_AXIS_COL,
                           gridcolor=_GRID_COL,
                           tickfont=dict(color=_AXIS_COL)),
        legend       =dict(orientation="h", y=1.08, x=0,
                           bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        hovermode    ="x unified",
    )


def render_imu_graph(hub) -> None:
    if not _PLOTLY:
        st.warning("plotly not installed — IMU graphs unavailable.")
        return

    bufs = hub.get_imu_buffers()
    if len(bufs["ax"]) < 4:
        st.info("Waiting for IMU data…")
        return

    ts    = bufs["ts"]
    t_rel = ts - ts[-1]
    snap  = hub.snapshot()

    col_left, col_right = st.columns(2)

    # ── Accelerometer ──────────────────────────────────────────────────────
    with col_left:
        fig_a = go.Figure()
        for arr, col, name in zip(
                [bufs["ax"], bufs["ay"], bufs["az"]],
                _ACCEL_COLS,
                ["Ax", "Ay", "Az"]):
            fig_a.add_trace(go.Scatter(
                x=t_rel, y=arr, name=name, mode="lines",
                line=dict(color=col, width=1.2)))

        # Magnitude
        fig_a.add_trace(go.Scatter(
            x=t_rel, y=bufs["mag"], name="|A|", mode="lines",
            line=dict(color="#ffffff", width=1.0, dash="dot")))

        # Motion artifact band
        _shade_motion(fig_a, t_rel, snap.motion_state)

        fig_a.update_layout(**_make_layout(
            "Accelerometer — MPU-6050", "Acceleration [g]"))
        st.plotly_chart(fig_a, use_container_width=True,
                        config={"displayModeBar": False})

        # Current values row
        c1, c2, c3 = st.columns(3)
        c1.metric("Ax", f"{snap.accel_x:+.3f} g")
        c2.metric("Ay", f"{snap.accel_y:+.3f} g")
        c3.metric("Az", f"{snap.accel_z:+.3f} g")

    # ── Gyroscope ──────────────────────────────────────────────────────────
    with col_right:
        fig_g = go.Figure()
        for arr, col, name in zip(
                [bufs["gx"], bufs["gy"], bufs["gz"]],
                _GYRO_COLS,
                ["Gx", "Gy", "Gz"]):
            fig_g.add_trace(go.Scatter(
                x=t_rel, y=arr, name=name, mode="lines",
                line=dict(color=col, width=1.2)))

        _shade_motion(fig_g, t_rel, snap.motion_state)

        fig_g.update_layout(**_make_layout(
            "Gyroscope — MPU-6050", "Angular Rate [°/s]"))
        st.plotly_chart(fig_g, use_container_width=True,
                        config={"displayModeBar": False})

        c1, c2, c3 = st.columns(3)
        c1.metric("Gx", f"{snap.gyro_x:+.2f} °/s")
        c2.metric("Gy", f"{snap.gyro_y:+.2f} °/s")
        c3.metric("Gz", f"{snap.gyro_z:+.2f} °/s")

    # ── Motion summary bar ─────────────────────────────────────────────────
    state_colours = {
        "STABLE": "#00e676", "LOW MOTION": "#69f0ae",
        "MEDIUM MOTION": "#ffeb3b", "HIGH MOTION": "#ff9800",
        "INVALID SIGNAL": "#f44336",
    }
    state_col = state_colours.get(snap.motion_state, "#546e7a")

    st.markdown(
        f"<div style='background:#0d1117;border:1px solid #21262d;"
        f"border-left:4px solid {state_col};border-radius:8px;"
        f"padding:8px 14px;display:flex;gap:24px;align-items:center'>"
        f"<span style='color:{state_col};font-weight:700'>{snap.motion_state}</span>"
        f"<span style='color:#546e7a;font-size:0.8rem'>Severity M={snap.dynamic_accel:.4f} g"
        f" · Pitch {snap.pitch_deg:.1f}° · Roll {snap.roll_deg:.1f}°"
        f" · Tilt {snap.tilt_deg:.1f}°"
        f" · PPG Valid {snap.ppg_validity_pct:.0f}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _shade_motion(fig, t_rel: np.ndarray, motion_state: str) -> None:
    """Add a red translucent band across the full time range when motion is high."""
    if motion_state in ("HIGH MOTION", "INVALID SIGNAL"):
        fig.add_vrect(
            x0=float(t_rel[0]), x1=float(t_rel[-1]),
            fillcolor="rgba(244,67,54,0.08)",
            layer="below", line_width=0,
            annotation_text="MOTION ARTEFACT",
            annotation_font=dict(color="#f44336", size=9),
            annotation_position="top left",
        )
