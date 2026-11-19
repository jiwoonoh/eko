def heart_rate(ecg_path: str) -> float | None:
    """Return heart rate (bpm) from median RR, or None."""
    from .preprocessing import prepare_beats

    return from_beats(prepare_beats(ecg_path))


def from_beats(beats) -> float | None:
    from .preprocessing import median

    rr_ms = beats.rr_seconds * 1000
    median_rr = median(rr_ms[rr_ms < 2000], 300, 2000)
    return round(60 / (median_rr / 1000), 2) if median_rr is not None else None
