"""Render bounded answers from evidence references, never generated medical prose."""

from __future__ import annotations

import json
import math

from classifier import TOOL_NAME as CLASSIFIER_TOOL
from classifier import label_for_key, load_labels, score_key, validate_output
from contracts import Record, ToolObservation


LABELS = {
    "heart_rate_bpm": "Heart rate",
    "rr_interval_ms": "RR interval",
    "pr_interval_ms": "PR interval",
    "q_s_peak_span_ms": "Q-to-S peak span (not full QRS onset-to-offset duration)",
    "qtc_bazett_ms": "QTc estimate (Bazett)",
}
MEASUREMENT_LIMITS = (
    "These measurements do not establish rhythm, rule out atrial fibrillation, "
    "or assess normality. "
    "Filter thresholds are estimator settings, not validated clinical reference ranges."
)
LIMITS = MEASUREMENT_LIMITS + " Classification is unavailable."
CLASSIFIER_LIMITS = (
    "ECGFounder outputs independent, uncalibrated model scores. Scores need not sum to one. "
    "A score does not establish a clinical diagnosis or rule out a condition. "
    "No validated decision threshold or clinical positivity is applied."
)
EXPLANATIONS = {
    "capabilities": "I can report ECG measurements, their methods, and missingness from registered tools. Classification is unavailable.",
    "measurement_limits": LIMITS,
    "heart_rate_bpm": "Heart rate is estimated as 60000 divided by the median retained RR interval in ms.",
    "rr_interval_ms": "RR is the median retained interval between adjacent detected R peaks. A median does not describe beat-to-beat variability or establish rhythm regularity.",
    "pr_interval_ms": "PR is an estimated P-onset to R-onset span, with inclusive sample counting. It does not establish P-wave morphology or sinus rhythm.",
    "q_s_peak_span_ms": "Q-to-S peak span measures detected Q peak to S peak, with inclusive sample counting. It is not an onset-to-offset QRS duration.",
    "qtc_bazett_ms": "QTc is a Bazett estimate using Q-peak to T-offset QT and the preceding RR interval from the same beat. It is not a validated clinical onset-to-offset QT measurement.",
    "classifier_scores": "ECGFounder supplies 150 independent sigmoid scores. NORMAL SINUS RHYTHM, NORMAL ECG, SINUS RHYTHM, and ATRIAL FIBRILLATION are separate labels in its pinned label manifest.",
    "classifier_limits": CLASSIFIER_LIMITS,
}
UNAVAILABLE = {
    "classification": "Classification is unavailable: no verified classifier is enabled. I cannot assign a rhythm label to this ECG.",
    "diagnosis": "Diagnosis is unavailable in this measurement runtime.",
    "normality": "Normality assessment is unavailable: the reported estimates have no independently validated reference-range interpretation.",
    "treatment": "Treatment recommendations are unavailable in this measurement runtime.",
    "unregistered_tool": "That operation is unavailable through the registered tools.",
}
REPORT_TOOL = "submit_report"


def report_schema(record: Record, evidence, tool_names: set[str]) -> dict:
    """Describe terminal report arguments using only current usable evidence."""
    usable = [item for item in evidence if _usable(item, record, tool_names)]
    properties = {"evidence_id": {"type": "string"}, "finding": {"type": "string"}}
    findings = {"type": "array", "items": {
        "type": "object", "properties": properties,
        "required": ["evidence_id", "finding"], "additionalProperties": False}}
    if usable:
        properties["evidence_id"]["enum"] = [item.evidence_id for item in usable]
        properties["finding"]["enum"] = sorted({key for item in usable for key in item.findings})
    else:
        findings["maxItems"] = 0
    explanations = ["record"] + [key for key in EXPLANATIONS
                                  if CLASSIFIER_TOOL in tool_names or not key.startswith("classifier_")]
    meanings = "; ".join(f"{code}: " + (
        "Selected record metadata and measurement lead." if code == "record"
        else _explanation(code, CLASSIFIER_TOOL in tool_names)) for code in explanations)
    return {"type": "function", "function": {
        "name": REPORT_TOOL,
        "description": "Finish this turn with evidence references and fixed explanations. Use after required scientific calls. The controller supplies values and label names; this action executes no scientific tool.",
        "parameters": {"type": "object", "properties": {
            "findings": {**findings, "description": "References for every requested finding, including null measurements. Use each requested finding's latest relevant evidence."},
            "explanations": {"type": "array", "items": {"type": "string", "enum": explanations},
                             "description": "Select the fixed explanations requested by the user. " + meanings},
            "unavailable": {"type": "array", "items": {"type": "string", "enum": list(UNAVAILABLE)}},
        }, "required": ["findings", "explanations", "unavailable"], "additionalProperties": False}}}


def report_instructions(tool_names: set[str] | None = None) -> str:
    """Describe the report contract and only the classifier capability enabled by the host."""
    enabled = CLASSIFIER_TOOL in (tool_names or set())
    codes = [key for key in EXPLANATIONS if enabled or not key.startswith("classifier_")]
    instructions = (
        "Finish by calling submit_report alone with exactly three argument keys: "
        '{"findings": [{"evidence_id": "actual observation ID", "finding": "actual finding key"}], '
        '"explanations": [], "unavailable": []}. '
        "Do not include prose, Markdown fences, numeric values, units, labels, diagnoses, or extra keys. "
        "The controller inserts actual values and fixed explanations. "
        "Use only references from tool observations for this selected record. "
        "Select the latest relevant observation for a requested finding; reuse existing evidence unless asked to rerun. "
        "explanations accepts only record, " + ", ".join(codes) + ". "
        "unavailable accepts only " + ", ".join(UNAVAILABLE) + ". "
        'For greetings, use explanations=["capabilities"]. '
        "Never infer normality, rhythm regularity, sinus rhythm, AF absence, P-wave morphology, "
        "or clinical reference ranges from measurement summaries. "
        "Assistant history contains validated report JSON; finish using submit_report rather than writing text. "
    )
    if not enabled:
        return instructions + (
            "For classification or rhythm questions, return findings=[], explanations=[], "
            'unavailable=["classification"].'
        )
    label_keys = {label: score_key(index) for index, label in enumerate(load_labels())}
    catalog = "; ".join(f"{key}={label}" for label, key in label_keys.items())
    aliases = "; ".join(f"{alias} means {label} ({label_keys[label]})" for alias, label in {
        "AFIB": "ATRIAL FIBRILLATION", "NORM": "NORMAL ECG",
        "SR": "SINUS RHYTHM", "NSR": "NORMAL SINUS RHYTHM",
    }.items())
    return instructions + (
        "The registered call_classification_tool supplies ECGFounder scores for this record. "
        "Call it for classification or rhythm-score questions when no usable classifier result exists. "
        "Reference its actual evidence ID and score keys; never supply a diagnosis or label text yourself. "
        "For a broad classification question, select up to five highest scores. "
        "For a named class, select that class's score even when it is small. "
        f"Exact label aliases: {aliases}. "
        "These are distinct outputs. Use the requested label's exact key. "
        "Do not merge them or infer disease absence from a low score. "
        'Use unavailable=["classification"] only when no usable result is available for this record. '
        f"Pinned label catalog: {catalog}"
    )


REPORT_INSTRUCTIONS = report_instructions()


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


def parse_report(content: str) -> dict:
    """Parse exact report fields and codes; evidence references are resolved when rendering."""
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
    normalized = []
    seen = set()
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"evidence_id", "finding"}:
            raise ValueError("finding must reference only evidence_id and finding")
        identity, name = ref["evidence_id"], ref["finding"]
        if not isinstance(identity, str) or not identity or not isinstance(name, str) or not name:
            raise ValueError("finding reference fields must be nonempty strings")
        if (identity, name) in seen:
            raise ValueError("duplicate finding reference")
        seen.add((identity, name))
        normalized.append({"evidence_id": identity, "finding": name})
    return {"findings": normalized, "explanations": list(explanations), "unavailable": list(unavailable)}


def _latest_classifier(record: Record, evidence, allowed_tools: set[str]) -> ToolObservation | None:
    for observation in reversed(evidence):
        if observation.tool != CLASSIFIER_TOOL or not _usable(observation, record, allowed_tools):
            continue
        try:
            validate_output(observation.to_dict())
        except ValueError:
            continue
        return observation
    return None


def _explanation(code: str, classifier_enabled: bool) -> str:
    if classifier_enabled and code == "capabilities":
        return (
            "I can report ECG measurements, methods, missingness, and ECGFounder model scores "
            "from the registered tools. Clinical diagnosis and treatment remain unavailable."
        )
    if classifier_enabled and code == "measurement_limits":
        return MEASUREMENT_LIMITS
    return EXPLANATIONS[code]


def render_report(content: str, record: Record, evidence: tuple[ToolObservation, ...] | list[ToolObservation],
                  *, tool_names: set[str] | None = None) -> str:
    """Resolve reports to trusted observations and pinned classifier labels."""
    allowed_tools = {"call_measurement_tool"} if tool_names is None else tool_names
    classifier_enabled = CLASSIFIER_TOOL in allowed_tools
    report = parse_report(content)
    if not classifier_enabled and any(code.startswith("classifier_") for code in report["explanations"]):
        raise ValueError("classifier explanations require an enabled classifier")
    by_id = {}
    for observation in evidence:
        if observation.evidence_id in by_id:
            raise ValueError("ambiguous evidence ID")
        by_id[observation.evidence_id] = observation
    lines = []
    validated_classifiers = set()
    measurement_refs = classifier_refs = False
    for ref in report["findings"]:
        identity, name = ref["evidence_id"], ref["finding"]
        observation = by_id.get(identity)
        if observation is None or not _usable(observation, record, allowed_tools) or name not in observation.findings:
            raise ValueError("finding has no usable evidence for the selected record")
        if observation.tool == CLASSIFIER_TOOL:
            if identity not in validated_classifiers:
                validate_output(observation.to_dict())
                validated_classifiers.add(identity)
            label = label_for_key(name)
            value = observation.findings[name]
            lines.append(f"Model score for {label}: {value} score [evidence {identity}]")
            classifier_refs = True
            continue
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
            if not isinstance(unit, str) or not unit:
                raise ValueError("finding has no unit")
            text = f"{value} {unit}"
        lines.append(f"{label}: {text} [evidence {identity}]")
        measurement_refs = True
    for code in report["explanations"]:
        if code == "record":
            lines.append(f"Selected record: {record.record_id}; sampling frequency: {record.sampling_rate_hz:g} Hz; "
                         f"lead order: {', '.join(record.lead_names)}; waveform units: {', '.join(record.waveform_units)}.")
            latest = next((obs for obs in reversed(evidence) if _usable(obs, record, allowed_tools)
                           and obs.tool == "call_measurement_tool"), None)
            if latest is not None and latest.details.get("lead_used"):
                lines.append(f"Measurement lead: {latest.details['lead_used']} [evidence {latest.evidence_id}].")
        else:
            lines.append(_explanation(code, classifier_enabled))
    for code in report["unavailable"]:
        if classifier_enabled and code == "classification":
            if _latest_classifier(record, evidence, allowed_tools) is not None:
                raise ValueError("classifier scores are already available for the selected record")
            lines.append("Classifier scores are unavailable for this record: no usable classifier result was produced.")
        elif classifier_enabled and code == "diagnosis":
            lines.append("Clinical diagnosis is unavailable in this research runtime.")
        else:
            lines.append(UNAVAILABLE[code])
    if measurement_refs and "measurement_limits" not in report["explanations"]:
        lines.append(_explanation("measurement_limits", classifier_enabled))
    if classifier_refs and "classifier_limits" not in report["explanations"]:
        lines.append(CLASSIFIER_LIMITS)
    return "\n\n".join(lines)


def fallback_report(record: Record, evidence: tuple[ToolObservation, ...] | list[ToolObservation],
                    *, tool_names: set[str] | None = None) -> dict:
    """Select latest measurements and at most five classifier scores for a safe fallback."""
    allowed_tools = {"call_measurement_tool"} if tool_names is None else tool_names
    latest = {}
    for observation in evidence:
        if _usable(observation, record, allowed_tools) and observation.tool != CLASSIFIER_TOOL:
            for name in observation.findings:
                latest[observation.tool, name] = {"evidence_id": observation.evidence_id, "finding": name}
    refs = list(latest.values())
    classifier = _latest_classifier(record, evidence, allowed_tools)
    if classifier is not None:
        ranked = sorted(range(len(load_labels())),
                        key=lambda index: (-classifier.findings[score_key(index)], index))
        refs.extend({"evidence_id": classifier.evidence_id, "finding": score_key(index)} for index in ranked[:5])
    return {"findings": refs, "explanations": [],
            "unavailable": [] if classifier is not None else ["classification"]}


def safe_report(record: Record, evidence: tuple[ToolObservation, ...] | list[ToolObservation],
                *, tool_names: set[str] | None = None) -> str:
    """Render the exact references selected for an invalid generated final response."""
    report = fallback_report(record, evidence, tool_names=tool_names)
    return render_report(json.dumps(report), record, evidence, tool_names=tool_names)
