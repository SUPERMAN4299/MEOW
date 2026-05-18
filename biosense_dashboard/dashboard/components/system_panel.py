"""
components/system_panel.py — System health and diagnostics panel.
"""
import time
import streamlit as st

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


def _bar(value: float, max_val: float = 100.0,
         colour: str = "#00e676", height: int = 6) -> str:
    pct = min(100.0, max(0.0, value / max(max_val, 1e-6) * 100.0))
    bar_col = colour
    if pct > 80: bar_col = "#f44336"
    elif pct > 60: bar_col = "#ffeb3b"
    return (
        f"<div style='background:#21262d;border-radius:3px;height:{height}px;width:100%'>"
        f"<div style='background:{bar_col};width:{pct:.0f}%;height:100%;border-radius:3px'></div>"
        f"</div>"
    )


def render_system_panel(snap, mgr) -> None:
    st.markdown(
        """<div style='background:#0d1117;border:1px solid #21262d;
                border-radius:10px;padding:12px 16px;'>
           <span style='color:#00e5ff;font-size:0.85rem;font-weight:700;
                        letter-spacing:1px'>⚙️ SYSTEM DIAGNOSTICS</span>""",
        unsafe_allow_html=True,
    )

    # CPU / RAM / Disk
    if _PSUTIL:
        cpu   = psutil.cpu_percent(interval=None)
        ram   = psutil.virtual_memory()
        temps = {}
        try:
            temps = psutil.sensors_temperatures() or {}
        except Exception:
            pass

        st.markdown(f"**CPU** {cpu:.0f}%", unsafe_allow_html=False)
        st.markdown(_bar(cpu), unsafe_allow_html=True)

        st.markdown(f"**RAM** {ram.used/1e6:.0f} MB / {ram.total/1e6:.0f} MB "
                    f"({ram.percent:.0f}%)")
        st.markdown(_bar(ram.percent), unsafe_allow_html=True)

        # Pi CPU temperature
        cpu_temp = None
        for name, entries in temps.items():
            for e in entries:
                if "cpu" in name.lower() or "temp" in name.lower():
                    cpu_temp = e.current
                    break
        if cpu_temp is not None:
            st.markdown(f"**Pi Temp** {cpu_temp:.1f} °C")
            st.markdown(_bar(cpu_temp, 85, "#ff7043"), unsafe_allow_html=True)
    else:
        st.info("psutil not available — install for system metrics.")

    st.markdown("---")

    # Sensor statuses
    statuses = mgr.status()
    for key, s in statuses.items():
        icon   = "🟢" if s.running else "🔴"
        mode   = "HW" if s.hw_active else "SIM"
        restarts = f" (↺{s.restart_cnt})" if s.restart_cnt else ""
        st.markdown(
            f"{icon} **{s.name}**  "
            f"<span style='color:#546e7a;font-size:0.78rem'>[{mode}]{restarts}</span>",
            unsafe_allow_html=True,
        )
        if s.error_msg:
            st.caption(f"⚠ {s.error_msg[:80]}")

    st.markdown("---")

    # I2C / signal health
    col1, col2 = st.columns(2)
    col1.metric("I²C Errors", snap.i2c_error_count)
    col2.metric("PPG Validity", f"{snap.ppg_validity_pct:.0f}%")

    st.markdown("</div>", unsafe_allow_html=True)
