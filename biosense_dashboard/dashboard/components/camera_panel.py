"""
components/camera_panel.py — NoIR camera live preview panel.

Displays the latest captured frame from the camera subsystem with
an IR-optimised display.  The frame is rendered at 4 Hz (driven by
Streamlit's auto-refresh) to keep CPU usage minimal on Raspberry Pi.

Fallback: when no frame is available, renders a placeholder test pattern.
"""
import time
import numpy as np
import streamlit as st

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


def _ir_false_colour(gray: np.ndarray) -> np.ndarray:
    """
    Apply a false-colour map to a grayscale IR image for perceptual clarity.
    Uses OpenCV COLORMAP_INFERNO which maps dark → purple, bright → yellow/white.
    This gives a heat-map appearance ideal for near-IR tissue imaging.
    """
    if not _CV2:
        # Fallback: replicate to 3-channel BGR
        return np.stack([gray, gray, gray], axis=2)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def _make_placeholder(h: int = 240, w: int = 320) -> np.ndarray:
    """
    Generate a synthetic placeholder frame when no camera is connected.
    Renders a crosshair + text on a dark background.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (12, 18, 24)  # dark blue-grey
    # Draw crosshair
    cx, cy = w // 2, h // 2
    if _CV2:
        cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 80, 80), 1)
        cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 80, 80), 1)
        cv2.circle(frame, (cx, cy), 40, (0, 60, 60), 1)
        cv2.putText(frame, "NoIR Camera", (cx - 58, cy - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 120), 1)
        cv2.putText(frame, "NO SIGNAL", (cx - 50, cy + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 60, 200), 1)
    return frame


def render_camera_panel(hub) -> None:
    """
    Render the NoIR camera preview inside a styled container.
    Converts BGR → RGB for st.image display.
    """
    st.markdown(
        """
        <div style='background:#0d1117;border:1px solid #21262d;
             border-radius:10px;padding:10px 12px 4px 12px;margin-bottom:4px'>
          <span style='color:#546e7a;font-size:0.75rem;font-weight:600;
                       letter-spacing:1px'>📷 NOIR CAMERA  ·  850 nm IR  ·  EXPERIMENTAL</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    frame = hub.get_camera_frame()

    if frame is None:
        frame = _make_placeholder()
        caption_txt = "⚫ Camera: no signal — simulation placeholder"
    else:
        caption_txt = "🟢 Camera: live feed (IR false-colour)"

    # Convert BGR → grayscale → false colour for IR display
    if frame.ndim == 3 and frame.shape[2] == 3 and _CV2:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = _ir_false_colour(gray)
        # Convert BGR → RGB for Streamlit
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    elif frame.ndim == 3:
        # Already RGB or single-channel replicated
        display = frame[:, :, ::-1] if frame.shape[2] == 3 else frame
    else:
        display = np.stack([frame, frame, frame], axis=2)

    snap = hub.snapshot()
    # ROI overlay: draw a rectangle on the display image
    if _CV2 and snap.finger_present_camera:
        h, w = display.shape[:2]
        x1 = int(w * 0.275); y1 = int(h * 0.225)
        x2 = int(w * 0.725); y2 = int(h * 0.775)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 128), 2)
        cv2.putText(display, "ROI", (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)

    st.image(display, use_container_width=True, caption=caption_txt)

    # Sub-metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Exposure", f"{snap.exposure_us // 1000:.0f} ms")
    c2.metric("Gain", f"×{snap.analogue_gain:.1f}")
    c3.metric("ROI Mean", f"{snap.roi_mean_ir:.0f} LSB")
