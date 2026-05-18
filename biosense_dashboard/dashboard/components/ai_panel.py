"""
components/ai_panel.py — AI/ML placeholder panel for future integration.

Displays mock AI confidence scores and anomaly detection placeholders.
Designed as integration points for future ML models (e.g. LSTM-based
anomaly detection, SpO2 regression, stress estimation).
"""
import time
import math
import streamlit as st


def _gauge(label: str, value: float, colour: str, unit: str = "%") -> str:
    """Mini circular-ish gauge using CSS."""
    pct = min(100, max(0, value))
    return f"""
    <div style='display:inline-block;text-align:center;margin:6px 8px;'>
      <div style='
        width:70px;height:70px;border-radius:50%;
        background: conic-gradient({colour} {pct*3.6:.0f}deg, #21262d 0deg);
        display:flex;align-items:center;justify-content:center;
        position:relative;
      '>
        <div style='
          width:54px;height:54px;border-radius:50%;background:#0d1117;
          display:flex;align-items:center;justify-content:center;
          flex-direction:column;
        '>
          <span style='color:{colour};font-size:0.9rem;font-weight:700'>{value:.0f}</span>
          <span style='color:#546e7a;font-size:0.55rem'>{unit}</span>
        </div>
      </div>
      <div style='color:#546e7a;font-size:0.65rem;margin-top:4px;max-width:70px'>{label}</div>
    </div>
    """


def render_ai_panel(snap) -> None:
    """
    Render the AI/ML future-integration panel with placeholder metrics.

    In a production deployment, these values would be populated by:
      · An LSTM or transformer-based anomaly detector on the PPG window
      · A neural-network SpO2 regressor trained on paired datasets
      · A multi-modal stress index fusing PPG + IMU features
    """
    st.markdown(
        """<div style='background:#0d1117;border:1px solid #21262d;
                border-radius:10px;padding:12px 16px;'>
           <span style='color:#ce93d8;font-size:0.85rem;font-weight:700;
                        letter-spacing:1px'>🤖 AI PREDICTION MODULE</span>
           <span style='color:#546e7a;font-size:0.72rem;margin-left:8px'>
             [PLACEHOLDER — NOT DEPLOYED]</span>""",
        unsafe_allow_html=True,
    )

    # ── Derived mock confidence from real sensor values ────────────────────
    # These are NOT AI outputs — they interpolate real sensor SQI/motion
    # values to give plausible placeholder numbers that respond to signal changes.
    t = time.monotonic()
    sqi_norm  = snap.ppg_sqi / 100.0
    mot_ok    = 1.0 if snap.motion_state == "STABLE" else 0.4
    opt_conf  = snap.optical_quality_conf

    anomaly_score = max(0.0, 100.0 - snap.ppg_sqi * 0.6 - mot_ok * 20
                        + 8 * math.sin(t * 0.3))
    anomaly_score = min(100.0, anomaly_score)

    stress_idx = max(0.0, min(100.0,
        40.0 + snap.dynamic_accel * 120 + (1 - mot_ok) * 30
        + 5 * math.sin(t * 0.15)))

    spo2_conf = max(0.0, min(100.0,
        sqi_norm * 60 + opt_conf * 30 + mot_ok * 10))

    signal_quality = max(0.0, min(100.0,
        snap.ppg_sqi * 0.5 + opt_conf * 35 + mot_ok * 15))

    # ── Gauge row ─────────────────────────────────────────────────────────
    gauges_html = (
        _gauge("Anomaly Risk",  anomaly_score, "#f44336")
        + _gauge("Stress Index",   stress_idx,    "#ff9800")
        + _gauge("SpO₂ Conf",      spo2_conf,     "#00e676")
        + _gauge("Signal Quality", signal_quality, "#00bcd4")
    )
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;justify-content:center;"
        f"margin:8px 0'>{gauges_html}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Status rows ───────────────────────────────────────────────────────
    items = [
        ("🔍 Anomaly Detector",   "LSTM model",         "Pending training data"),
        ("😰 Stress Estimator",   "HRV + IMU fusion",   "Placeholder active"),
        ("🧬 SpO₂ Regressor",     "CNN on PPG waveform","Calibration required"),
        ("🫁 Breathing Rate",     "FFT on chest motion","Prototype stage"),
        ("🧠 Oxidative Stress",   "Multi-modal model",  "Research phase"),
    ]
    for icon_label, model, status in items:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:4px 0;border-bottom:1px solid #21262d'>"
            f"<span style='color:#b0bec5;font-size:0.78rem'>{icon_label}</span>"
            f"<span style='color:#546e7a;font-size:0.72rem'>{model}</span>"
            f"<span style='color:#546e7a;font-size:0.7rem;font-style:italic'>{status}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "⚠ AI outputs are non-validated placeholders for integration testing. "
        "No clinical interpretation should be made from these values."
    )
