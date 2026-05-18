"""
components/metric_cards.py — Live biomedical metric cards.

Displays BPM, SpO2, Temperature, Motion, and Tremor in styled cards
designed to resemble a clinical patient monitoring workstation.
"""
import streamlit as st
from utils.formatting import (
    fmt_bpm, fmt_spo2, fmt_pi, fmt_temp,
    bpm_colour, spo2_colour, motion_colour, quality_colour, sqi_colour
)


def _card(icon: str, label: str, value: str, unit: str,
          colour: str, sublabel: str = "") -> str:
    return f"""
    <div style='
        background:#0d1117;
        border:1px solid #21262d;
        border-left:4px solid {colour};
        border-radius:10px;
        padding:14px 16px;
        margin-bottom:10px;
    '>
      <div style='color:#546e7a;font-size:0.75rem;font-weight:600;
                  letter-spacing:1px;text-transform:uppercase'>{icon} {label}</div>
      <div style='margin-top:4px;display:flex;align-items:baseline;gap:6px'>
        <span style='color:{colour};font-size:2.1rem;font-weight:800;
                     font-variant-numeric:tabular-nums'>{value}</span>
        <span style='color:#546e7a;font-size:0.85rem'>{unit}</span>
      </div>
      <div style='color:#546e7a;font-size:0.72rem;margin-top:2px'>{sublabel}</div>
    </div>
    """


def render_metric_cards(snap) -> None:
    """Render all live metric cards into a 2-column grid."""
    c1, c2 = st.columns(2)

    with c1:
        # BPM
        bpm_col = bpm_colour(snap.bpm)
        bpm_sub = f"SQI: {snap.ppg_sqi:.0f}% · {snap.ppg_sqi_label}"
        st.markdown(_card("🫀", "Heart Rate", fmt_bpm(snap.bpm), "BPM",
                          bpm_col, bpm_sub), unsafe_allow_html=True)

        # SpO2
        spo2_col = spo2_colour(snap.spo2)
        r_info = f"R-ratio: {snap.perfusion_index:.3f}" if snap.perfusion_index else ""
        st.markdown(_card("🩸", "SpO₂", fmt_spo2(snap.spo2), "",
                          spo2_col, r_info), unsafe_allow_html=True)

        # Perfusion Index
        pi_col = "#00bcd4" if snap.perfusion_index and snap.perfusion_index > 0.3 else "#546e7a"
        st.markdown(_card("💧", "Perfusion Index", fmt_pi(snap.perfusion_index), "",
                          pi_col, "AC/DC ratio (non-clinical)"), unsafe_allow_html=True)

    with c2:
        # Motion state
        mot_col = motion_colour(snap.motion_state)
        mot_sub = (f"M={snap.dynamic_accel:.4f} g · Tilt={snap.tilt_deg:.1f}°"
                   f" · Valid={snap.ppg_validity_pct:.0f}%")
        st.markdown(_card("📡", "Motion State", snap.motion_state, "",
                          mot_col, mot_sub), unsafe_allow_html=True)

        # Die / ambient temperature
        temp_col = "#ff7043" if snap.die_temp_c > 40 else "#80cbc4"
        st.markdown(_card("🌡️", "Die Temperature", fmt_temp(snap.die_temp_c), "",
                          temp_col, "MPU-6050 internal sensor"), unsafe_allow_html=True)

        # Tremor
        if snap.tremor_detected:
            tr_col = "#ff9800"
            tr_val = f"{snap.tremor_hz:.2f} Hz"
            tr_sub = f"Band: {snap.tremor_band} · SNR: {snap.tremor_snr_db:.1f} dB"
        else:
            tr_col = "#546e7a"
            tr_val = "NONE"
            tr_sub = "No tremor detected"
        st.markdown(_card("〰️", "Tremor", tr_val, "", tr_col, tr_sub),
                    unsafe_allow_html=True)

    # Optical quality bar (full width)
    opt_col = quality_colour(snap.optical_quality_label)
    opt_sub = (f"ROI mean: {snap.roi_mean_ir:.1f} LSB · "
               f"SNR: {snap.optical_snr_db:.1f} dB · "
               f"Conf: {snap.optical_quality_conf*100:.0f}%")
    st.markdown(_card("🔬", "Optical Quality (iPPG)",
                      snap.optical_quality_label,
                      f"· Exp {snap.exposure_us//1000:.0f}ms · Gain ×{snap.analogue_gain:.1f}",
                      opt_col, opt_sub), unsafe_allow_html=True)
