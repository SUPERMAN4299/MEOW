"""
graphs/optical_graph.py — Optical signal analysis panel (NoIR camera iPPG).

Renders:
  Left  : Raw ROI intensity signal (scrolling)
  Right : Spectral analysis bar showing iPPG spectral quality

All iPPG outputs are labelled EXPERIMENTAL and non-clinical.
"""
import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

_DARK_BG  = "#0d1117"
_GRID_COL = "#161b22"
_AXIS_COL = "#546e7a"


def render_optical_graph(hub) -> None:
    if not _PLOTLY:
        st.warning("plotly not installed — optical graph unavailable.")
        return

    opt  = hub.get_optical_buffer()
    raw  = opt["raw"]
    ts   = opt["ts"]
    snap = hub.snapshot()

    col_wave, col_spec = st.columns([3, 2])

    # ── Raw ROI waveform ──────────────────────────────────────────────────
    with col_wave:
        if len(raw) < 4:
            st.info("Waiting for optical data…")
        else:
            t_rel = ts - ts[-1]
            # IIR detrend for display (remove DC)
            tau   = 0.995
            dc_est= np.empty_like(raw)
            s     = raw[0]
            for i, x in enumerate(raw):
                s = tau * s + (1 - tau) * x
                dc_est[i] = s
            ac_sig = raw - dc_est

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_rel, y=raw,
                name="Raw ROI [LSB]", mode="lines",
                line=dict(color="#80cbc4", width=1.2),
            ))
            fig.add_trace(go.Scatter(
                x=t_rel, y=ac_sig * 50 + raw.mean(),
                name="AC (×50)", mode="lines",
                line=dict(color="#00e5ff", width=1.0, dash="dot"),
            ))

            fig.update_layout(
                paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_BG,
                font=dict(color="#90a4ae", size=10),
                margin=dict(l=40, r=10, t=30, b=40), height=200,
                title=dict(
                    text="iPPG ROI Intensity — EXPERIMENTAL NON-CLINICAL",
                    font=dict(color="#80cbc4", size=11), x=0.01, y=0.97),
                xaxis=dict(title="Time (s)", color=_AXIS_COL,
                           gridcolor=_GRID_COL,
                           tickfont=dict(color=_AXIS_COL)),
                yaxis=dict(title="Intensity [LSB]", color=_AXIS_COL,
                           gridcolor=_GRID_COL,
                           tickfont=dict(color=_AXIS_COL)),
                legend=dict(orientation="h", y=1.08, x=0,
                            bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

    # ── Spectral quality panel ────────────────────────────────────────────
    with col_spec:
        st.markdown(
            "<div style='background:#0d1117;border:1px solid #21262d;"
            "border-radius:8px;padding:12px 14px;height:200px;"
            "display:flex;flex-direction:column;justify-content:center'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<span style='color:#80cbc4;font-size:0.78rem;font-weight:600;"
            "letter-spacing:1px'>iPPG SPECTRAL METRICS</span>",
            unsafe_allow_html=True,
        )

        items = [
            ("BPM Proxy", "EXPERIMENTAL",
             "---" if not snap.bpm else f"~{snap.bpm:.0f}",
             "#ffeb3b"),
            ("Optical SNR",     "dB",   f"{snap.optical_snr_db:.1f}", "#00bcd4"),
            ("Optical Quality", "",     snap.optical_quality_label,  "#00e676"),
            ("Camera FPS",      "fps",  f"{snap.camera_fps:.1f}",    "#80cbc4"),
            ("AC RMS",          "LSB",  f"{snap.optical_ac_rms:.3f}", "#ce93d8"),
        ]
        for label, unit, val, col in items:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:3px 0;border-bottom:1px solid #161b22'>"
                f"<span style='color:#90a4ae;font-size:0.78rem'>{label}</span>"
                f"<span style='color:{col};font-size:0.78rem;font-weight:700'>"
                f"{val} <span style='color:#546e7a;font-size:0.68rem'>{unit}</span>"
                f"</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # Warning disclaimer
    st.caption(
        "⚠ iPPG BPM proxy is an EXPERIMENTAL non-clinical estimate derived "
        "from imaging photoplethysmography.  Not validated against a certified "
        "medical device.  Do not use for clinical decision-making."
    )
