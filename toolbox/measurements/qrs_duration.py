def qrs_duration(ecg_path: str) -> float | None:
    """Return median Q-to-S peak span (ms), keeping 80-240, or None."""
    from .preprocessing import prepare_beats

    return from_beats(prepare_beats(ecg_path))


def from_beats(beats) -> float | None:
    from .preprocessing import median

    values = beats.intervals("ECG_Q_Peaks", "ECG_S_Peaks", require_increasing=True)
    value = median(values, 80, 240)
    return round(value, 2) if value is not None else None
