"""ECG-Agent's single-lead NeuroKit beat detection and delineation."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from joblib import Memory
import neurokit2 as nk
import numpy as np

from ..leads import LEADS
from ..record import load_record

cache_dir = Path.cwd() / ".eko-cache"
cache_dir.mkdir(mode=0o700, exist_ok=True)
cache = Memory(cache_dir / "measurements", verbose=0)

# ECG-Agent requires seven R-peaks for measurement.
MIN_PEAKS = 7


@cache.cache
def _delineate(cleaned, r_peaks, sampling_rate, neurokit_version):
    return nk.ecg_delineate(cleaned, r_peaks, sampling_rate=sampling_rate, method="dwt")[1]


@dataclass
class Beats:
    cleaned: np.ndarray
    sampling_rate: int
    r_peaks: np.ndarray

    @property
    def rr_seconds(self) -> np.ndarray:
        return np.diff(self.r_peaks) / self.sampling_rate

    @cached_property
    def waves(self) -> dict:
        return _delineate(self.cleaned, self.r_peaks, self.sampling_rate, nk.__version__)

    def intervals(self, start: str, end: str, *, require_increasing=False) -> np.ndarray:
        """Drop missing landmark pairs and count both endpoints, as ECG-Agent does."""
        starts = np.asarray(self.waves.get(start, []), dtype=float)
        ends = np.asarray(self.waves.get(end, []), dtype=float)
        count = min(starts.size, ends.size)
        starts, ends = starts[:count], ends[:count]
        valid = ~np.isnan(starts) & ~np.isnan(ends)
        if require_increasing:
            valid &= ends > starts
        return (ends[valid] - starts[valid] + 1) / self.sampling_rate * 1000


def median(values: np.ndarray, minimum: float, maximum: float) -> float | None:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values) & (values >= minimum) & (values <= maximum)]
    return float(np.median(valid)) if valid.size else None


def _load_measurement_lead(ecg_path: str) -> tuple[np.ndarray, int, str]:
    """Load the measurement waveform, defaulting to lead II for standard MAT files."""
    data, rate = load_record(ecg_path)
    signal = np.asarray(data["feats"], dtype=float)
    if rate != 500:
        raise ValueError("ECG-Agent's measurement tool requires 500 Hz ECGs")
    lead = np.asarray(data.get("measurement_lead", "II")).item()
    return signal[LEADS.index(lead)], 500, lead


@cache.cache
def _detect_beats(ecg_path: str, file_version: tuple, neurokit_version: str) -> Beats:
    signal, sampling_rate, lead = _load_measurement_lead(ecg_path)
    if np.all(signal == 0) or np.all(np.isnan(signal)):
        raise ValueError(f"Lead {lead} contains only zeros or NaNs")

    cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate, method="neurokit")
    info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)[1]
    r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=float)
    if r_peaks.size < MIN_PEAKS:
        raise ValueError(
            f"At least {MIN_PEAKS} R-peaks are required; found {r_peaks.size}")

    # Joblib hashes array strides. Match the layout of beats reloaded from disk.
    return Beats(np.ascontiguousarray(cleaned), sampling_rate, r_peaks)


def prepare_beats(ecg_path: str | Path) -> Beats:
    """Share detected beats across measurements of the same record."""
    path = Path(ecg_path).expanduser().resolve(strict=True)
    stat = path.stat()
    file_version = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    return _detect_beats(str(path), file_version, nk.__version__)


def clear_cache() -> None:
    cache.clear(warn=False)
