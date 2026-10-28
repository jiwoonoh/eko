from .ecg import get_ecg, intervals, median

def pr_interval(ecg_path: str) -> float | None:
    """Return median lead II P-onset to QRS-onset (ms), keeping 80-400, or None."""
    ecg = get_ecg(ecg_path)
    values = intervals(ecg.waves, "ECG_P_Onsets", "ECG_R_Onsets", ecg.sampling_rate)
    value = median(values, 80, 400)
    return round(value, 2) if value is not None else None
