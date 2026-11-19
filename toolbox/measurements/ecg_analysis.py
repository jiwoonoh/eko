import warnings

from .heart_rate import from_beats as heart_rate
from .pr_interval import from_beats as pr_interval
from .qrs_duration import from_beats as qrs_duration
from .qtc_interval import from_beats as qtc_interval

PANEL = {
    "heart_rate": (heart_rate, 2),
    "pr_interval": (pr_interval, 0),
    "qrs_duration": (qrs_duration, 0),
    "qtc_interval": (qtc_interval, 2),
}


def ecg_analysis(ecg_path: str) -> dict[str, str | None]:
    """Use ECG-Agent's display precision: HR/QTc 2 decimals, PR/QRS whole ms."""
    try:
        from .preprocessing import prepare_beats

        beats = prepare_beats(ecg_path)
        results = {}
        for name, (measure, digits) in PANEL.items():
            value = measure(beats)
            results[name] = f"{value:.{digits}f}" if value is not None else None
        return results
    except Exception as error:
        # ECG-Agent's CSV interface represents failed analyses as missing values.
        warnings.warn(f"ECG measurement failed ({type(error).__name__}): {error}",
                      RuntimeWarning, stacklevel=2)
        return dict.fromkeys(PANEL)
