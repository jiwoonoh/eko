import math
import numpy as np
from .ecg import get_ecg, intervals, median

def qtc_interval(ecg_path: str) -> float | None:
    """Return median lead II Bazett QTc (ms), keeping 300-600, or None."""
    ecg = get_ecg(ecg_path)
    qt_ms = intervals(ecg.waves, "ECG_Q_Peaks", "ECG_T_Offsets", ecg.sampling_rate)
    rr_sec = np.diff(ecg.r_peaks) / ecg.sampling_rate
    values = [qt / math.sqrt(rr) for qt, rr in zip(qt_ms, rr_sec) if 0 < rr < 2]
    value = median(values, 300, 600)
    return round(value, 2) if value is not None else None
