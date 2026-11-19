import math


def qtc_interval(ecg_path: str) -> float | None:
    """Return median Bazett QTc (ms), keeping 300-600, or None."""
    from .preprocessing import prepare_beats

    return from_beats(prepare_beats(ecg_path))


def from_beats(beats) -> float | None:
    from .preprocessing import median

    qt_ms = beats.intervals("ECG_Q_Peaks", "ECG_T_Offsets")
    values = [qt / math.sqrt(rr) for qt, rr in zip(qt_ms, beats.rr_seconds) if 0 < rr < 2]
    value = median(values, 300, 600)
    return round(value, 2) if value is not None else None
