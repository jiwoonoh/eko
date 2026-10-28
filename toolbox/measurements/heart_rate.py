from .ecg import get_ecg

def heart_rate(ecg_path: str) -> float | None:
    """Return lead II heart rate (bpm) from median RR, or None."""
    rr_ms = get_ecg(ecg_path).median_rr_ms
    return round(60 / (rr_ms / 1000), 2) if rr_ms is not None else None
