import cv2


def format_mmss_ms(seconds: float) -> str:
    """Format seconds to MM:SS.ms used by reasoning and UI outputs."""
    total_ms = int(max(0.0, seconds) * 1000)
    mm = total_ms // 60000
    ss = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{mm:02d}:{ss:02d}.{ms:03d}"


def apply_temporal_anchor(frame, current_sec: float):
    """
    Burn a high-contrast temporal anchor on each frame.
    This visual cue helps downstream VLM reasoning align with video clock.
    """
    if frame is None:
        return frame

    stamp = format_mmss_ms(current_sec)
    text = f"T {stamp}"

    # Contrast-safe text rendering: dark backing + bright foreground.
    cv2.putText(frame, text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
