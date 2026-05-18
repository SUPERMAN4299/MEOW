"""
graphs/ppg_graph.py — Real-time PPG waveform graph using Plotly.

Renders the IR and RED photoplethysmography signals from the MAX30102
as scrolling waveforms.  The graph is updated on every Streamlit rerun
(4 Hz) with the latest ring-buffer data.

Design choices:
  · Dark biomedical theme (matching the dashboard).
  · Normalised display so both IR and RED fit on the same Y-axis.
  · Plotly go.Figure for fine-grained layout control.
  · X-axis shows relative time in seconds (most recent sample = 0).
  · Downsampling: at 100 Hz, 500 samples = 5 s.  We display all of them
    (Plotly WebGL handles 500 points at ~0 cost on modern browsers).
"""
import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

# Shared layout template
_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor ="#0d1117",
    font         =dict(color="#90a4ae", size=11),
    margin       =dict(l=40, r=10, t=30, b=40),
    height       =200,
    xaxis        =dict(
        title="Time (s)", color="#546e7a",
        gridcolor="#161b22", zerolinecolor="#21262d",
        tickfont=dict(color="#546e7a"),
    ),
    yaxis        =dict(
        title="Normalised",
        color="#546e7a", gridcolor="#161b22",
        tickfont=dict(color="#546e7a"),
    ),
    legend       =dict(
        orientation="h", y=1.08, x=0,
        bgcolor="rgba(0,0,0,0)", font=dict(size=10)
    ),
    hovermode    ="x unified",
)


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; returns zeros if flat."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def render_ppg_graph(hub) -> None:
    if not _PLOTLY:
        st.warning("plotly not installed — PPG graph unavailable.")
        return

    bufs = hub.get_ppg_buffers()
    ir   = bufs["ir"]
    red  = bufs["red"]
    ts   = bufs["ts"]

    if len(ir) < 4:
        st.info("Waiting for PPG data…")
        return

    # Relative time axis (seconds, 0 = most recent)
    t_rel = ts - ts[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_rel, y=_normalise(ir),
        name="IR (880 nm)", mode="lines",
        line=dict(color="#00e5ff", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=t_rel, y=_normalise(red),
        name="RED (660 nm)", mode="lines",
        line=dict(color="#f44336", width=1.5),
    ))

    snap = hub.snapshot()
    # Annotate BPM if available
    if snap.bpm:
        fig.add_annotation(
            x=t_rel[0], y=0.95,
            text=f"BPM: {snap.bpm:.0f}",
            showarrow=False,
            font=dict(color="#00e676", size=13, family="monospace"),
            xanchor="left",
        )

    layout = dict(
        **_LAYOUT,
        title=dict(
            text="PPG Waveforms — MAX30102",
            font=dict(color="#00e5ff", size=13), x=0.01, y=0.97
        ),
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # SQI indicator
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Signal Quality", f"{snap.ppg_sqi:.0f}%", label_visibility="visible")
    col2.metric("SQI Label", snap.ppg_sqi_label)
    col3.metric("DC IR", f"{snap.dc_ir_ppg:,.0f}")
    col4.metric("AC RMS", f"{snap.ac_ir_ppg:.0f}")
