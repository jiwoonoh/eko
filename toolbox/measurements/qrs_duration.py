from .ecg import get_ecg, intervals, median

def qrs_duration(ecg_path: str) -> float | None:
    """Return median lead II Q-to-S peak span (ms), keeping 80-240, or None."""
    ecg = get_ecg(ecg_path)
    values = intervals(ecg.waves, "ECG_Q_Peaks", "ECG_S_Peaks",
                       ecg.sampling_rate, require_increasing=True)
    value = median(values, 80, 240)
    return round(value, 2) if value is not None else None
