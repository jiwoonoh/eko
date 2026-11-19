def pr_interval(ecg_path: str) -> float | None:
    """Return median P-onset to QRS-onset (ms), keeping 80-400, or None."""
    from .preprocessing import prepare_beats

    return from_beats(prepare_beats(ecg_path))


def from_beats(beats) -> float | None:
    from .preprocessing import median

    values = beats.intervals("ECG_P_Onsets", "ECG_R_Onsets")
    value = median(values, 80, 400)
    return round(value, 2) if value is not None else None
