"""Read waveforms in ECG-Agent's MAT format."""

from scipy.io import loadmat


def load_record(path):
    """Return MAT fields and sampling rate. ECG-Agent defaults to 500 Hz."""
    fields = loadmat(path)
    rate = fields.get("curr_sample_rate")
    return fields, float(rate.item()) if rate is not None else 500
