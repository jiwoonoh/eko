"""Combined report containing only the four ECG measurements."""

from .heart_rate import heart_rate
from .pr_interval import pr_interval
from .qrs_duration import qrs_duration
from .qtc_interval import qtc_interval


def ecg_analysis(ecg_path: str) -> dict[str, float | None]:
    """Return heart rate (bpm), PR interval, QRS duration, and QTc interval (ms).

    Unavailable values are None; loading and processing errors propagate.
    """
    return {
        "heart_rate": heart_rate(ecg_path),
        "pr_interval": pr_interval(ecg_path),
        "qrs_duration": qrs_duration(ecg_path),
        "qtc_interval": qtc_interval(ecg_path),
    }
