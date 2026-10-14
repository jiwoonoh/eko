"""Load a verified ECG record without changing the source file."""

from __future__ import annotations

import hashlib
from io import BytesIO
import math
from pathlib import Path

from contracts import Record, STANDARD_LEADS, canonical_record_id

MAX_RECORD_BYTES = 64 * 1024 * 1024


def _strings(value, np) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        return tuple(text for item in array.flat for text in _strings(item, np))
    if array.dtype.kind not in {"U", "S"}:
        raise ValueError("lead names and waveform units must be strings")
    def text(item):
        return item.decode() if isinstance(item, bytes) else str(item)

    if array.ndim == 2 and array.shape[1] > 1 and array.dtype.itemsize in {1, 4}:
        values = ["".join(text(x) for x in row) for row in array]
    else:
        values = array.ravel().tolist()
    return tuple(text(x).strip() for x in values)


def _leads(names: tuple[str, ...]) -> tuple[str, ...]:
    standard = {lead.casefold(): lead for lead in STANDARD_LEADS}
    try:
        normalized = tuple(standard[name.casefold()] for name in names)
    except KeyError as error:
        raise ValueError(f"unknown ECG lead: {error.args[0]}") from None
    if len(normalized) != 12 or len(set(normalized)) != 12:
        raise ValueError("record must declare all 12 distinct standard ECG leads")
    return normalized


def load_record(path, lead_order=None):
    """Return metadata and a channels-first array parsed from the same hashed bytes.

    A MAT file without lead names needs an explicitly confirmed complete order.
    The upstream PTB-XL exporter omits lead names and waveform units.
    """
    import numpy as np
    from scipy.io import loadmat

    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("rb") as source:
        payload = source.read(MAX_RECORD_BYTES + 1)
    if len(payload) > MAX_RECORD_BYTES:
        raise ValueError("ECG MAT file exceeds the 64 MiB limit")
    data = loadmat(BytesIO(payload))
    if "feats" not in data:
        raise ValueError("ECG MAT file must contain channels-first 'feats'")
    signal = np.asarray(data["feats"])
    if signal.ndim != 2 or signal.shape[0] != 12 or signal.shape[1] < 1:
        raise ValueError("ECG 'feats' must have shape (12, samples)")
    if signal.dtype.kind not in {"i", "u", "f"} or not np.isfinite(signal).all():
        raise ValueError("ECG 'feats' must contain finite real numeric samples")
    signal = signal.astype(np.float64, copy=False)

    if "curr_sample_rate" not in data:
        raise ValueError("ECG MAT file must contain 'curr_sample_rate'")
    frequency = np.asarray(data["curr_sample_rate"])
    if frequency.size != 1 or frequency.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("'curr_sample_rate' must be one numeric scalar")
    sampling_rate = float(frequency.ravel()[0])
    if not math.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("'curr_sample_rate' must be positive and finite")

    declared = [_leads(_strings(data[key], np)) for key in ("lead_names", "sig_name") if key in data]
    if declared and any(names != declared[0] for names in declared):
        raise ValueError("ECG lead metadata disagrees")
    if declared:
        names = declared[0]
        if lead_order is not None and _leads(tuple(lead_order)) != names:
            raise ValueError("confirmed lead order disagrees with ECG metadata")
    else:
        if lead_order is None:
            raise ValueError("missing lead metadata; explicitly confirm all 12 ECG leads in row order")
        names = _leads(tuple(lead_order))

    unit_fields = [_strings(data[key], np) for key in ("waveform_units", "units") if key in data]
    normalized_units = []
    for values in unit_fields:
        if len(values) == 1:
            values = values * 12
        if len(values) != 12 or any(not value for value in values):
            raise ValueError("waveform units must contain one value or 12 nonempty values")
        normalized_units.append(values)
    if normalized_units and any(values != normalized_units[0] for values in normalized_units):
        raise ValueError("ECG waveform unit metadata disagrees")
    units = normalized_units[0] if normalized_units else ("unknown",) * 12
    record = Record(
        path=str(resolved),
        record_id=canonical_record_id(resolved.name),
        sha256=hashlib.sha256(payload).hexdigest(),
        sampling_rate_hz=sampling_rate,
        lead_names=names,
        waveform_units=units,
    )
    return record, signal
