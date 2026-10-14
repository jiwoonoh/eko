"""Numeric ECG estimators adapted from the pinned ECG-Agent implementation.

Original: Hyunseung Chung, MIT; see THIRD_PARTY_NOTICES.md.
Q/S peak span is not onset-to-offset QRS duration.
"""

from __future__ import annotations

import math
import time

from contracts import Record, UPSTREAM_REVISION
from records import load_record

VERSION = "ecg-measurements-v1"
FIELDS = ("heart_rate_bpm", "rr_interval_ms", "pr_interval_ms", "q_s_peak_span_ms", "qtc_bazett_ms")
UNITS = {name: "bpm" if name == "heart_rate_bpm" else "ms" for name in FIELDS}
METHODS = {
    "heart_rate_bpm": "60000 / median retained RR interval (ms)",
    "rr_interval_ms": "median adjacent R-peak interval",
    "pr_interval_ms": "median P-onset to R-onset span, inclusive samples",
    "q_s_peak_span_ms": "median Q-peak to S-peak span, inclusive samples; not QRS onset-to-offset duration",
    "qtc_bazett_ms": "median Bazett QTc; Q-peak to T-offset QT, inclusive samples; preceding RR from the same beat",
}


def empty_result(record: Record, reason: str, message: str, status="failed") -> dict:
    return {
        "record_id": record.record_id,
        "record_hash": record.sha256,
        "version": VERSION,
        "status": status,
        "findings": dict.fromkeys(FIELDS),
        "units": dict(UNITS),
        "methods": dict(METHODS),
        "missingness": dict.fromkeys(FIELDS, reason),
        "details": {"reason": reason, "message": message, "upstream_revision": UPSTREAM_REVISION},
        "elapsed": 0.0,
        "source": "live",
    }



def measure_record(record: Record) -> dict:
    """Measurement backend is supplied by the next implementation stage."""
    return empty_result(record, "backend_pending", "Live measurement backend is not installed in this core stage")
