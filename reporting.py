"""Render bounded answers from evidence references, never generated medical prose."""

from __future__ import annotations

import json
import math

from contracts import Record, ToolObservation


LABELS = {
    "heart_rate_bpm": "Heart rate",
    "rr_interval_ms": "RR interval",
    "pr_interval_ms": "PR interval",
    "q_s_peak_span_ms": "Q-to-S peak span (not full QRS onset-to-offset duration)",
    "qtc_bazett_ms": "QTc estimate (Bazett)",
}
LIMITS = (
    "These measurements do not establish rhythm, rule out atrial fibrillation, "
    "or assess normality. Classification is unavailable. "
    "Filter thresholds are estimator settings, not validated clinical reference ranges."
)
EXPLANATIONS = {
    "capabilities": "I can report ECG measurements, their methods, and missingness from registered tools. Classification is unavailable.",
    "measurement_limits": LIMITS,
    "heart_rate_bpm": "Heart rate is estimated as 60000 divided by the median retained RR interval in ms.",
    "rr_interval_ms": "RR is the median retained interval between adjacent detected R peaks. A median does not describe beat-to-beat variability or establish rhythm regularity.",
    "pr_interval_ms": "PR is an estimated P-onset to R-onset span, with inclusive sample counting. It does not establish P-wave morphology or sinus rhythm.",
    "q_s_peak_span_ms": "Q-to-S peak span measures detected Q peak to S peak, with inclusive sample counting. It is not an onset-to-offset QRS duration.",
    "qtc_bazett_ms": "QTc is a Bazett estimate using Q-peak to T-offset QT and the preceding RR interval from the same beat. It is not a validated clinical onset-to-offset QT measurement.",
}
UNAVAILABLE = {
    "classification": "Classification is unavailable: no verified classifier is enabled. I cannot assign a rhythm label to this ECG.",
    "diagnosis": "Diagnosis is unavailable in this measurement runtime.",
    "normality": "Normality assessment is unavailable: the reported estimates have no independently validated reference-range interpretation.",
    "treatment": "Treatment recommendations are unavailable in this measurement runtime.",
    "unregistered_tool": "That operation is unavailable through the registered tools.",
}
REPORT_INSTRUCTIONS = (
    "Final answers must be a JSON object with exactly three keys: "
    '{"findings": [{"evidence_id": "actual observation ID", "finding": "actual finding key"}], '
    '"explanations": [], "unavailable": []}. '
    "Do not include prose, Markdown fences, numeric values, units, labels, diagnoses, or extra keys. "
    "The controller inserts actual values and fixed explanations. "
    "Use only references from tool observations for this selected record. "
    "Select the latest relevant observation for a requested finding; reuse existing evidence unless asked to rerun. "
    "explanations accepts only record, " + ", ".join(EXPLANATIONS) + ". "
    "unavailable accepts only " + ", ".join(UNAVAILABLE) + ". "
    "For classification or rhythm questions, return findings=[], explanations=[], "
    'unavailable=["classification"]. For greetings, use explanations=["capabilities"]. '
    "Never infer normality, rhythm regularity, sinus rhythm, AF absence, P-wave morphology, "
    "or clinical reference ranges from measurement summaries. "
    "Earlier assistant messages are controller-rendered; your final output must follow this JSON contract."
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate report key")
        result[key] = value
    return result


def _usable(observation: ToolObservation, record: Record, tool_names: set[str]) -> bool:
    return (observation.record_id == record.record_id
            and observation.record_hash == record.sha256
            and observation.status in ("success", "partial")
            and observation.source not in ("replay", "runner")
            and observation.tool in tool_names)


def _codes(value, allowed):
    if not isinstance(value, list) or any(not isinstance(item, str) or item not in allowed for item in value):
        raise ValueError("unsupported report explanation or capability")
    if len(value) != len(set(value)):
        raise ValueError("duplicate report explanation or capability")
    return value


def render_report(content: str, record: Record, evidence: tuple[ToolObservation, ...] | list[ToolObservation],
                  *, tool_names: set[str] | None = None) -> str:
    """Reject all free text; resolve each numeric reference to current-record evidence."""
    allowed_tools = {"call_measurement_tool"} if tool_names is None else tool_names
    try:
        report = json.loads(content, object_pairs_hook=_unique_object)
    except (TypeError, ValueError) as error:
        raise ValueError("final response must follow the report JSON contract") from error
    if not isinstance(report, dict) or set(report) != {"findings", "explanations", "unavailable"}:
        raise ValueError("report must contain exactly findings, explanations, and unavailable")
    refs = report["findings"]
    if not isinstance(refs, list):
        raise ValueError("report findings must be evidence references")
    explanations = _codes(report["explanations"], set(EXPLANATIONS) | {"record"})
    unavailable = _codes(report["unavailable"], UNAVAILABLE)
    if not refs and not explanations and not unavailable:
        raise ValueError("report has no content")
    by_id = {}
    for observation in evidence:
        if observation.evidence_id in by_id:
            raise ValueError("ambiguous evidence ID")
        by_id[observation.evidence_id] = observation
    lines = []
    seen = set()
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"evidence_id", "finding"}:
            raise ValueError("finding must reference only evidence_id and finding")
        identity, name = ref["evidence_id"], ref["finding"]
        if not isinstance(identity, str) or not isinstance(name, str):
            raise ValueError("finding reference fields must be strings")
        if (identity, name) in seen:
            raise ValueError("duplicate finding reference")
        seen.add((identity, name))
        observation = by_id.get(identity)
        if observation is None or not _usable(observation, record, allowed_tools) or name not in observation.findings:
            raise ValueError("finding has no usable evidence for the selected record")
        value = observation.findings[name]
        label = LABELS.get(name, name)
        if value is None:
            reason = observation.missingness.get(name)
            if not reason:
                raise ValueError("missing finding has no explanation")
            text = f"unavailable ({reason})"
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("finding must be a finite number or null")
            unit = observation.units.get(name)
            if not unit:
                raise ValueError("finding has no unit")
            text = f"{value} {unit}"
        lines.append(f"{label}: {text} [evidence {identity}]")
    for code in explanations:
        if code == "record":
            lines.append(f"Selected record: {record.record_id}; sampling frequency: {record.sampling_rate_hz:g} Hz; "
                         f"lead order: {', '.join(record.lead_names)}; waveform units: {', '.join(record.waveform_units)}.")
            latest = next((obs for obs in reversed(evidence) if _usable(obs, record, allowed_tools)
                           and obs.tool == "call_measurement_tool"), None)
            if latest is not None and latest.details.get("lead_used"):
                lines.append(f"Measurement lead: {latest.details['lead_used']} [evidence {latest.evidence_id}].")
        else:
            lines.append(EXPLANATIONS[code])
    lines.extend(UNAVAILABLE[code] for code in unavailable)
    if refs and "measurement_limits" not in explanations:
        lines.append(LIMITS)
    return "\n\n".join(lines)


def safe_report(record: Record, evidence: tuple[ToolObservation, ...] | list[ToolObservation],
                *, tool_names: set[str] | None = None) -> str:
    """Use trusted observations when rejecting an invalid generated final response."""
    allowed_tools = {"call_measurement_tool"} if tool_names is None else tool_names
    latest = {}
    for observation in evidence:
        if _usable(observation, record, allowed_tools):
            for name in observation.findings:
                latest[name] = {"evidence_id": observation.evidence_id, "finding": name}
    report = {"findings": list(latest.values()), "explanations": [], "unavailable": ["classification"]}
    return render_report(json.dumps(report), record, evidence, tool_names=allowed_tools)
