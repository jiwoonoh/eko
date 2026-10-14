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


def _landmark(values, index):
    if index >= len(values):
        return None
    if values[index] is None:
        return None
    value = float(values[index])
    return value if math.isfinite(value) else None


def _span(waves, first, last, index, sampling_rate):
    start = _landmark(waves.get(first, ()), index)
    end = _landmark(waves.get(last, ()), index)
    if start is None or end is None:
        return None, "missing_landmarks"
    if end <= start:
        return None, "invalid_landmark_order"
    return (end - start + 1) * 1000 / sampling_rate, None


def _estimate(peaks, waves, sampling_rate):
    """Keep beat indexes when landmarks are absent, including QT/RR pairing."""
    import numpy as np

    estimates = {name: [] for name in FIELDS if name != "heart_rate_bpm"}
    bounds = {"rr_interval_ms": (300, 2000), "pr_interval_ms": (80, 400),
              "q_s_peak_span_ms": (80, 240), "qtc_bazett_ms": (300, 600)}

    def add(name, beat, value, reason=None):
        if value is not None and not math.isfinite(value):
            value, reason = None, "nonfinite_estimate"
        elif value is not None and (value < bounds[name][0] or value > bounds[name][1]):
            reason = "outside_upstream_filter"
        estimates[name].append({"beat_index": beat, "value": value, "included": reason is None, "reason": reason})

    for index, peak in enumerate(peaks):
        if index > 0:
            rr = (peak - peaks[index - 1]) * 1000 / sampling_rate
            add("rr_interval_ms", index, rr, "rr_not_below_2000_ms" if rr >= 2000 else None)
        pr, reason = _span(waves, "ECG_P_Onsets", "ECG_R_Onsets", index, sampling_rate)
        add("pr_interval_ms", index, pr, reason)
        span, reason = _span(waves, "ECG_Q_Peaks", "ECG_S_Peaks", index, sampling_rate)
        add("q_s_peak_span_ms", index, span, reason)
        qt, reason = _span(waves, "ECG_Q_Peaks", "ECG_T_Offsets", index, sampling_rate)
        qtc = None
        if qt is not None:
            if index == 0:
                reason = "no_preceding_rr"
            else:
                rr_seconds = (peak - peaks[index - 1]) / sampling_rate
                if rr_seconds <= 0 or rr_seconds >= 2:
                    reason = "invalid_preceding_rr"
                else:
                    qtc = qt / math.sqrt(rr_seconds)
        add("qtc_bazett_ms", index, qtc, reason)
        estimates["qtc_bazett_ms"][-1].update(
            qt_interval_ms=qt,
            preceding_rr_ms=None if index == 0 else (peak - peaks[index - 1]) * 1000 / sampling_rate,
        )

    findings = dict.fromkeys(FIELDS)
    missingness = {}
    filtering = {}
    for name, values in estimates.items():
        retained = [item["value"] for item in values if item["included"]]
        exclusions = {}
        for item in values:
            if not item["included"]:
                reason = item["reason"]
                exclusions[reason] = exclusions.get(reason, 0) + 1
        filtering[name] = {"retained": len(retained), "excluded": len(values) - len(retained), "reasons": exclusions,
                           "inclusive_range": list(bounds[name])}
        if retained:
            findings[name] = round(float(np.median(retained)), 2)
        else:
            missingness[name] = "no retained estimates; see filtering reasons"
    if findings["rr_interval_ms"] is not None:
        retained_rr = [item["value"] for item in estimates["rr_interval_ms"] if item["included"]]
        findings["heart_rate_bpm"] = round(60000 / float(np.median(retained_rr)), 2)
    else:
        missingness["heart_rate_bpm"] = "no retained RR estimates"
    return findings, missingness, estimates, filtering


def measure_record(record: Record) -> dict:
    """Measure only the selected bytes, frequency, and lead layout."""
    started = time.monotonic()
    current, signal = load_record(record.path, lead_order=record.lead_names)
    if current != record:
        raise ValueError("selected ECG record changed; attach it again before analysis")

    import numpy as np
    import neurokit2 as nk

    result = empty_result(record, "not_measured", "")
    result["details"] = {"upstream_revision": UPSTREAM_REVISION, "neurokit_version": nk.__version__,
                         "sampling_rate_hz": record.sampling_rate_hz, "lead_used": "II",
                         "waveform_units": record.waveform_units[record.lead_names.index("II")]}
    lead = signal[record.lead_names.index("II")]
    if np.ptp(lead) == 0:
        result.update(status="failed", missingness=dict.fromkeys(FIELDS, "flat_signal"))
        result["details"].update(reason="flat_signal", message="Lead II contains no variation")
    else:
        try:
            cleaned = nk.ecg_clean(lead, sampling_rate=record.sampling_rate_hz, method="neurokit")
            _, detected = nk.ecg_peaks(cleaned, sampling_rate=record.sampling_rate_hz)
            peaks = np.asarray(detected["ECG_R_Peaks"], dtype=float)
            if peaks.ndim != 1 or not np.isfinite(peaks).all() or np.any(np.diff(peaks) <= 0):
                raise ValueError("R peaks must be finite and strictly increasing")
            result["details"]["r_peak_count"] = len(peaks)
            if len(peaks) < 7:
                result.update(status="partial", missingness=dict.fromkeys(FIELDS, "fewer_than_7_r_peaks"))
                result["details"].update(reason="fewer_than_7_r_peaks", message="Upstream estimator requires at least seven R peaks")
            else:
                _, waves = nk.ecg_delineate(cleaned, detected, sampling_rate=record.sampling_rate_hz, method="dwt")
                findings, missingness, estimates, filtering = _estimate(peaks, waves, record.sampling_rate_hz)
                available = sum(value is not None for value in findings.values())
                result.update(findings=findings, missingness=missingness,
                              status="success" if available == len(FIELDS) else "partial" if available else "failed")
                result["details"].update(per_beat=estimates, filtering=filtering)
        except Exception as error:
            result.update(status="failed", missingness=dict.fromkeys(FIELDS, "signal_processing_failed"))
            result["details"].update(reason="signal_processing_failed", message=str(error))
    result["elapsed"] = time.monotonic() - started
    return result
