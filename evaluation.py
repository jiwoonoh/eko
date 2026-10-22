"""Recorded-output adapters and checks for explicit measurement claims.

Replay evidence is not live execution. Claim checks do not assess unrestricted
prose, diagnostic accuracy, or the validity of the measurement algorithm.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from dataclasses import asdict, dataclass, field
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Iterator, Sequence

from contracts import UPSTREAM_REVISION, ToolObservation, TurnResult, canonical_record_id


MEASUREMENTS = {
    "Heart_Rate": ("heart_rate_bpm", "bpm"),
    "PR_Interval_ms": ("pr_interval_ms", "ms"),
    "QRS_Duration_ms": ("q_s_peak_span_ms", "ms"),
    "QTc_ms": ("qtc_bazett_ms", "ms"),
}
LEGACY_FINDINGS = {
    "heart_rate": "Heart_Rate",
    "pr_interval": "PR_Interval_ms",
    "qrs_duration": "QRS_Duration_ms",
    "qtc_interval": "QTc_ms",
}
TOOL_KINDS = {
    "call_measurement_tool": "measurement",
    "call_classification_tool": "classification",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, str) and value.strip().lower() in {"", "null", "none", "nan"}:
        return None
    if isinstance(value, bool):
        raise ValueError("a measurement must be a number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid measurement: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError("non-finite measurement")
    return result


def _decode(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Published outputs sometimes encode Python lists with single quotes.
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


@dataclass(frozen=True)
class ReplayOutput:
    record_id: str
    source_sha256: str
    source_line: int
    tool: str
    status: str
    findings: dict[str, float | None] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    missingness: dict[str, str] = field(default_factory=dict)
    labels: tuple[dict[str, Any], ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def as_observation(self, call_id: str, record_hash: str = "") -> ToolObservation:
        """The source file hash identifies evidence; it is not a waveform hash."""
        return ToolObservation(
            record_id=self.record_id,
            record_hash=record_hash,
            call_id=call_id,
            evidence_id=f"replay:{self.source_sha256}:{self.source_line}:{call_id}",
            tool=self.tool,
            version=f"recorded:{self.source_sha256}",
            status=self.status,
            findings=self.findings,
            units=self.units,
            missingness=self.missingness,
            details={
                **self.details,
                "source_sha256": self.source_sha256,
                "source_line": self.source_line,
                "labels": self.labels,
                "waveform_hash_available": bool(record_hash),
            },
            source="replay",
        )


def _replay_output(
    record_id: str, tool: str, value: Any, source_sha256: str, source_line: int
) -> ReplayOutput:
    value = _decode(value)
    base = dict(record_id=record_id, source_sha256=source_sha256, source_line=source_line, tool=tool)
    if isinstance(value, dict) and value.get("error"):
        return ReplayOutput(**base, status="failed", details={"error": str(value["error"])})
    if isinstance(value, str) and (value.startswith("[Error:") or value == "Classification Error"):
        return ReplayOutput(**base, status="failed", details={"error": value})

    if tool == "call_measurement_tool":
        if not isinstance(value, dict):
            raise ValueError("measurement output must be an object")
        allowed = set(MEASUREMENTS) | set(LEGACY_FINDINGS)
        if set(value) - allowed:
            raise ValueError("unknown measurement fields")
        normalized = {}
        for key, raw in value.items():
            column = LEGACY_FINDINGS.get(key, key)
            if column in normalized:
                raise ValueError("duplicate measurement aliases")
            normalized[column] = raw
        findings = {name: _number(normalized.get(column)) for column, (name, _) in MEASUREMENTS.items()}
        findings["rr_interval_ms"] = None
        units = {name: unit for name, unit in MEASUREMENTS.values()}
        units["rr_interval_ms"] = "ms"
        missingness = {name: "unavailable_in_source" for name, number in findings.items() if number is None}
        measured = [findings[name] for name, _ in MEASUREMENTS.values()]
        status = "partial" if any(number is not None for number in measured) else "failed"
        return ReplayOutput(
            **base, status=status, findings=findings, units=units, missingness=missingness,
            details={
                "source_fields": {name: column for column, (name, _) in MEASUREMENTS.items()},
                "limitations": "QRS source field represents upstream Q-to-S peak span; source has no missingness causes or RR interval.",
            },
        )

    if tool != "call_classification_tool":
        raise ValueError(f"unknown recorded tool: {tool}")
    if isinstance(value, dict) and set(value) == {"classification"}:
        value = value["classification"]
    if value is None or value == "":
        return ReplayOutput(**base, status="failed", details={"error": "unavailable_in_source"})
    if isinstance(value, str) and re.fullmatch(r"None\s*>\s*\d+(?:\.\d+)?%", value):
        return ReplayOutput(**base, status="success", details={"selection": value})
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, list):
        raise ValueError("classification output must be a list of label probabilities")
    labels = []
    seen = set()
    for item in value:
        if isinstance(item, str) and re.fullmatch(r"None\s*>\s*\d+(?:\.\d+)?%", item):
            if len(value) != 1:
                raise ValueError("empty selection marker mixed with labels")
            return ReplayOutput(**base, status="success", details={"selection": item})
        match = re.fullmatch(
            r"([A-Za-z0-9_/]+(?:\([A-Za-z0-9_/]+\))?)\s*\(([0-9]+(?:\.[0-9]+)?)%\)",
            str(item).strip(),
        )
        if match is None:
            raise ValueError(f"invalid recorded label: {item!r}")
        label, probability = match[1], float(match[2]) / 100
        if label in seen or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("duplicate label or invalid classification probability")
        seen.add(label)
        labels.append({"label": label, "probability": probability})
    # An empty list may be an old missing-record join; preserve the ambiguity.
    status = "success" if labels else "partial"
    return ReplayOutput(**base, status=status, labels=tuple(labels), details={"empty_selection_cause": "unknown_in_source"} if not labels else {})


def load_tool_csv(path: str | Path, kind: str = "measurement", dataset: str = "ptbxl") -> dict[str, ReplayOutput]:
    """Load small recorded fixtures; reject duplicate normalized record aliases."""
    if kind not in {"measurement", "classification"}:
        raise ValueError("kind must be measurement or classification")
    source_sha = file_sha256(path)
    expected = {"ecg_file_path", *(MEASUREMENTS if kind == "measurement" else ["top_classes"])}
    result = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames) or set(reader.fieldnames) != expected:
            raise ValueError("unexpected tool CSV schema")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"malformed CSV row at line {reader.line_num}")
            record_id = canonical_record_id(row["ecg_file_path"], dataset)
            if record_id in result:
                raise ValueError(f"duplicate record alias: {record_id}")
            tool = "call_measurement_tool" if kind == "measurement" else "call_classification_tool"
            value = {column: row[column] for column in MEASUREMENTS} if kind == "measurement" else row["top_classes"]
            result[record_id] = _replay_output(record_id, tool, value, source_sha, reader.line_num)
    return result


@dataclass(frozen=True)
class RecordedTurn:
    turn_id: str
    question: str
    actions: tuple[str, ...]
    observations: tuple[ToolObservation, ...]
    answer: str
    reference_actions: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RecordedCase:
    case_id: str
    record_id: str
    source_sha256: str
    source_line: int
    schema: str
    replicate: int
    turns: tuple[RecordedTurn, ...]
    legacy_alignment: tuple[dict[str, Any], ...] = ()
    history: str = "without_gt"
    source_metadata: dict[str, Any] = field(default_factory=dict)


def _dialogue_groups(dialogue: Any) -> list[tuple[str, list[dict[str, Any]]]]:
    if not isinstance(dialogue, list):
        raise ValueError("dialogue must be a list")
    groups = []
    for turn in dialogue:
        if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"}:
            raise ValueError("unsupported recorded dialogue role")
        if turn["role"] == "user":
            question = turn.get("content")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("recorded user question is empty")
            groups.append((question, []))
        else:
            if not groups:
                raise ValueError("assistant turn has no preceding user question")
            if not isinstance(turn.get("action"), str):
                raise ValueError("recorded assistant turn has no action")
            groups[-1][1].append(turn)
    return groups


def _record_alias(value: Any) -> Any:
    value = _decode(value)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("each recorded case must select exactly one ECG")
        return value[0]
    return value


def iter_transcripts(path: str | Path, dataset: str = "ptbxl", history: str = "without_gt") -> Iterator[RecordedCase]:
    """Stream both upstream JSONL formats without treating repeated ECGs as one case."""
    if history != "without_gt":
        raise ValueError("with_gt replay/evaluation is not implemented")
    source_sha = file_sha256(path)
    replicates: Counter = Counter()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("transcript row must be an object")
                if raw.get("history", raw.get("inference_mode", history)) != history:
                    raise ValueError("recorded history mode differs from requested mode")
                if "generated_dialogue" in raw:
                    if "dialogue" in raw or "ecg_file" not in raw:
                        raise ValueError("ambiguous or incomplete current transcript schema")
                    schema = "upstream_inference"
                    record_id = canonical_record_id(_record_alias(raw["ecg_file"]), dataset)
                    groups = _dialogue_groups(raw["generated_dialogue"])
                    reference = _dialogue_groups(raw["ground_truth_dialogue"]) if "ground_truth_dialogue" in raw else None
                    alignment = raw.get("turns", [])
                    if not isinstance(alignment, list) or any(not isinstance(turn, dict) for turn in alignment):
                        raise ValueError("invalid legacy alignment")
                elif {"ecg_id", "dialogue", "batch_idx"}.issubset(raw):
                    schema = "upstream_published"
                    record_id = canonical_record_id(raw["ecg_id"], dataset)
                    groups = _dialogue_groups(raw["dialogue"])
                    reference, alignment = None, []
                else:
                    raise ValueError("unknown transcript schema")
                if reference is not None and [group[0] for group in groups] != [group[0] for group in reference]:
                    raise ValueError("reference user turns do not match generated user turns")
                if not groups:
                    raise ValueError("transcript contains no user turns")
                # Hash the ordered user sequence, not only its ECG or final question.
                question_sha = hashlib.sha256(json.dumps([g[0] for g in groups], ensure_ascii=False).encode()).hexdigest()
                replicate_key = (record_id, question_sha)
                replicates[replicate_key] += 1
                case_id = f"{source_sha}:{line_number}"
                turns = []
                for index, (question, assistant) in enumerate(groups, 1):
                    turn_id = f"{case_id}/user:{index}"
                    observations = []
                    for step, turn in enumerate(assistant, 1):
                        action = turn["action"]
                        if "tool_output" not in turn:
                            continue
                        if action not in TOOL_KINDS:
                            raise ValueError("tool result attached to an unknown tool action")
                        output = _replay_output(record_id, action, turn["tool_output"], source_sha, line_number)
                        observations.append(output.as_observation(f"{turn_id}/step:{step}"))
                    answer = "\n".join(turn["content"] for turn in assistant if isinstance(turn.get("content"), str))
                    refs = tuple(turn["action"] for turn in reference[index - 1][1]) if reference is not None else None
                    turns.append(RecordedTurn(turn_id, question, tuple(t["action"] for t in assistant), tuple(observations), answer, refs))
                metadata = {key: raw[key] for key in ("sample_id", "batch_idx", "source_category") if key in raw}
                metadata["history_declared_in_source"] = "history" in raw or "inference_mode" in raw
                yield RecordedCase(
                    case_id, record_id, source_sha, line_number, schema,
                    replicates[replicate_key], tuple(turns), tuple(alignment), history, metadata,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"transcript line {line_number}: {error}") from error


def legacy_nap(case: RecordedCase) -> dict[str, Any]:
    """Recompute legacy positional agreement only when explicit labels exist."""
    eligible = [turn for turn in case.legacy_alignment if turn.get("gt_action") is not None]
    if not eligible:
        return {"status": "unavailable", "reason": "no_saved_reference_actions", "total": 0, "correct": 0, "accuracy": None}
    correct = sum(
        isinstance(turn.get("gen_action"), str)
        and str(turn["gt_action"]).strip() == turn["gen_action"].strip()
        for turn in eligible
    )
    return {"status": "legacy_positional", "total": len(eligible), "correct": correct, "accuracy": correct / len(eligible)}


@dataclass(frozen=True)
class MeasurementClaim:
    evidence_id: str
    finding: str
    value: float
    unit: str
    decimals: int | None = None


def evaluate_claims(
    claims: Iterable[MeasurementClaim], evidence: Sequence[ToolObservation],
    *, record_id: str | None = None, requested_findings: Sequence[str] = ()
) -> dict[str, Any]:
    """Check explicit numeric claims. No text extraction or clinical judgment."""
    by_id = {observation.evidence_id: observation for observation in evidence}
    if len(by_id) != len(evidence):
        raise ValueError("duplicate evidence identifier")
    verdicts = []
    supported_fields = set()
    for claim in claims:
        if claim.decimals is not None and (isinstance(claim.decimals, bool) or not isinstance(claim.decimals, int) or not 0 <= claim.decimals <= 6):
            raise ValueError("claim decimals must be an integer from 0 to 6")
        actual = _number(claim.value)
        if actual is None:
            raise ValueError("a numeric claim cannot have a missing value")
        observation = by_id.get(claim.evidence_id)
        verdict = {"evidence_id": claim.evidence_id, "finding": claim.finding}
        if observation is None:
            verdict.update(status="unverifiable", reason="unknown_evidence")
        elif record_id is not None and observation.record_id != record_id:
            verdict.update(status="contradicted", reason="wrong_record")
        elif observation.status not in {"success", "partial"}:
            verdict.update(status="unverifiable", reason="tool_did_not_produce_usable_evidence")
        elif observation.findings.get(claim.finding) is None:
            verdict.update(status="unverifiable", reason="measurement_unavailable")
        elif claim.unit != observation.units.get(claim.finding):
            verdict.update(status="contradicted", reason="unit_mismatch")
        else:
            expected = _number(observation.findings[claim.finding])
            if claim.decimals is not None:
                expected = round(expected, claim.decimals)
            if math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9):
                verdict.update(status="supported", reason="matches_tool_value")
                supported_fields.add(claim.finding)
            else:
                verdict.update(status="contradicted", reason="value_mismatch")
        verdicts.append(verdict)
    counts = Counter(verdict["status"] for verdict in verdicts)
    available = {
        name for observation in evidence
        if observation.status in {"success", "partial"} and (record_id is None or observation.record_id == record_id)
        for name, value in observation.findings.items() if value is not None
    }
    requested_available = set(requested_findings) & available
    checkable = counts["supported"] + counts["contradicted"]
    return {
        "scope": "explicit_numeric_claims_only", "claims": verdicts,
        "supported": counts["supported"], "contradicted": counts["contradicted"], "unverifiable": counts["unverifiable"],
        "checkable": checkable, "support_rate": counts["supported"] / checkable if checkable else None,
        "requested_available": len(requested_available), "requested_supported": len(requested_available & supported_fields),
        "coverage": len(requested_available & supported_fields) / len(requested_available) if requested_available else None,
    }


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    record_id: str
    questions: tuple[str, ...]
    requested_findings: tuple[str, ...] = ()
    history: str = "without_gt"
    requested_findings_by_turn: tuple[tuple[str, ...], ...] = ()
    expected_tools_by_turn: tuple[tuple[str, ...], ...] = ()
    required_explanations_by_turn: tuple[tuple[str, ...], ...] = ()


def evaluate_task(result: TurnResult, record, *, requested_findings=(), expected_tools=None,
                  required_explanations=(), previous_evidence_ids=()) -> dict[str, Any]:
    """Check request fulfillment, including nulls, freshness, and tool scheduling.

    Expectations belong to the evaluator and never enter the model conversation.
    A replacement report on a failed turn cannot count as task completion.
    """
    evidence = {item.evidence_id: item for item in result.evidence
                if item.record_id == record.record_id and item.record_hash == record.sha256
                and item.status in {"success", "partial"} and item.source not in {"runner", "replay"}}
    latest = {key: item.evidence_id for item in evidence.values() for key in item.findings}
    report = result.report or {}
    reported, stale = set(), set()
    for reference in report.get("findings", []):
        key, identity = reference["finding"], reference["evidence_id"]
        observation = evidence.get(identity)
        if observation is None or key not in observation.findings:
            continue
        if observation.findings[key] is None and not observation.missingness.get(key):
            continue
        reported.add(key)
        if identity != latest[key] and key in requested_findings:
            stale.add(key)
    missing = sorted(set(requested_findings) - reported)
    explanations = sorted(set(required_explanations) - set(report.get("explanations", [])))
    observed_tools = [item.tool for item in result.evidence if item.evidence_id not in previous_evidence_ids]
    scheduling = expected_tools is None or (
        Counter(observed_tools) == Counter(expected_tools) and result.tool_calls == len(expected_tools))
    return {
        "complete": result.status == "success" and bool(report) and not missing and not stale
                    and not explanations and scheduling,
        "requested_findings": list(requested_findings), "reported_findings": sorted(reported),
        "missing_findings": missing, "stale_findings": sorted(stale),
        "missing_explanations": explanations, "expected_tools": expected_tools,
        "observed_tools": observed_tools, "tool_schedule_met": scheduling,
    }


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _session_configuration(session: Any) -> dict[str, Any]:
    """Collect explicit reproducibility fields without inspecting credentials."""
    model = session.model
    configuration = {
        "provider": f"{type(model).__module__}.{type(model).__qualname__}",
        "model": getattr(model, "model", None),
        "provider_settings": {
            name: getattr(model, name) for name in ("max_tokens", "timeout") if hasattr(model, name)
        },
        "limits": {
            name: getattr(session, name)
            for name in ("max_responses", "max_tool_calls", "tool_timeout", "request_timeout")
        },
        "prompt_sha256": _json_sha256([message for message in session.history if message.get("role") == "system"]),
        "tools_sha256": _json_sha256(session.tools),
        "classifier": getattr(getattr(session, "worker", None), "classifier_provenance", None),
    }
    provider_module = sys.modules.get(type(model).__module__)
    provider_source = getattr(provider_module, "__file__", None)
    configuration["provider_source_sha256"] = file_sha256(provider_source) if provider_source and Path(provider_source).is_file() else None
    json.dumps(configuration, allow_nan=False)
    return configuration


def _error_kind(turn: dict[str, Any]) -> str | None:
    kinds = {"provider", "tool", "runner", "judge"}
    explicit = turn.get("error_kind")
    if explicit is not None:
        if explicit not in kinds:
            raise ValueError("unknown error kind")
        return explicit
    error = turn.get("error")
    if not error:
        return None
    if not isinstance(error, str):
        raise ValueError("turn error must be text")
    prefix = error.partition(":")[0]
    return prefix if prefix in kinds else "runner"


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("numpy", "scipy", "neurokit2", "torch"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_manifest(
    cases: Iterable[EvaluationCase], session_factory: Callable[[EvaluationCase], Any],
    claims_for_result: Callable[[EvaluationCase, int, TurnResult], Iterable[MeasurementClaim]] | None = None,
) -> dict[str, Any]:
    """Run a fixed sorted manifest through the interactive runtime's own loop."""
    from eko import run_turn

    ordered = sorted(cases, key=lambda case: case.case_id)
    if len({case.case_id for case in ordered}) != len(ordered):
        raise ValueError("duplicate manifest case identifier")
    for case in ordered:
        if case.history != "without_gt":
            raise ValueError("with_gt evaluation is not implemented")
        if not case.case_id or not case.questions or any(not isinstance(question, str) or not question.strip() for question in case.questions):
            raise ValueError("case needs a stable identifier and nonempty questions")
        if case.requested_findings_by_turn and len(case.requested_findings_by_turn) != len(case.questions):
            raise ValueError("per-turn requested findings must match the question count")
        for expectations in (case.expected_tools_by_turn, case.required_explanations_by_turn):
            if expectations and len(expectations) != len(case.questions):
                raise ValueError("per-turn expectations must match the question count")
    turns = []
    manifest = []
    for case in ordered:
        session = session_factory(case)
        try:
            if session.record.record_id != case.record_id:
                raise ValueError("session record does not match manifest")
            record = session.record.public()
            configuration = _session_configuration(session)
            manifest.append({
                "case_id": case.case_id, "history": case.history,
                "questions_sha256": _json_sha256(case.questions),
                "requested_findings": case.requested_findings,
                "requested_findings_by_turn": case.requested_findings_by_turn,
                "expected_tools_by_turn": case.expected_tools_by_turn,
                "required_explanations_by_turn": case.required_explanations_by_turn,
                "record": record, "configuration": configuration,
            })
            evidence_steps = {}
            response_trace = []
            answer_rejections = []
            provider_attempts = []
            original_log = session.log

            def collect_models(event_type: str, **fields: Any) -> None:
                if event_type == "model_response":
                    response_trace.append({"model": fields.get("model"), "usage": fields.get("usage", {})})
                if event_type == "answer_rejected":
                    answer_rejections.append(fields.get("reason"))
                if event_type in {"model_response", "model_error"}:
                    provider_attempts.extend({key: item[key] for key in ("attempt", "model", "status", "usage", "usage_invalid")
                                              if key in item} for item in fields.get("provider_attempts", []))
                original_log(event_type, **fields)

            session.log = collect_models
            for index, question in enumerate(case.questions, 1):
                response_trace.clear()
                answer_rejections.clear()
                provider_attempts.clear()
                previous_ids = set(evidence_steps)
                result = run_turn(session, question)
                row = result.to_dict()
                row.update(
                    case_id=case.case_id, record_id=case.record_id, record=record,
                    configuration=configuration, model_responses_trace=list(response_trace),
                    user_turn=index, question=question, runtime_turn_id=result.turn_id,
                    turn_id=f"{case.case_id}/user:{index}",
                    answer_rejections=len(answer_rejections),
                    provider_attempts=list(provider_attempts),
                )
                new_steps = 0
                for observation in row["evidence"]:
                    evidence_id = observation["evidence_id"]
                    reused = evidence_id in evidence_steps
                    if not reused:
                        new_steps += 1
                        evidence_steps[evidence_id] = f"{row['turn_id']}/tool:{new_steps}"
                    observation["step_id"] = evidence_steps[evidence_id]
                    observation["reused"] = reused
                claims = tuple(claims_for_result(case, index, result)) if claims_for_result else ()
                requested = case.requested_findings_by_turn[index - 1] if case.requested_findings_by_turn else case.requested_findings
                row["claim_checks"] = evaluate_claims(claims, result.evidence, record_id=case.record_id, requested_findings=requested)
                if (case.requested_findings or case.requested_findings_by_turn or
                        case.expected_tools_by_turn or case.required_explanations_by_turn):
                    row["task_checks"] = evaluate_task(
                        result, session.record, requested_findings=requested,
                        expected_tools=case.expected_tools_by_turn[index - 1] if case.expected_tools_by_turn else None,
                        required_explanations=case.required_explanations_by_turn[index - 1] if case.required_explanations_by_turn else (),
                        previous_evidence_ids=previous_ids)
                row["error_kind"] = _error_kind(row)
                turns.append(row)
        finally:
            session.close()
    from traces import code_hashes
    return {
        "metadata": {
            "format_version": 1, "upstream_revision": UPSTREAM_REVISION,
            "python_version": sys.version.split()[0], "package_versions": _package_versions(),
            "source_sha256": {"evaluation.py": file_sha256(__file__), **code_hashes()},
            "manifest_sha256": _json_sha256(manifest),
        },
        "manifest": manifest, "summary": summarize(turns), "turns": turns,
    }


def summarize(turns: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Failures remain in completion counts; only checkable claims get support rates."""
    counts: Counter = Counter()
    ids = set()
    evidence_ids = set()
    total_elapsed = 0.0
    for turn in turns:
        identity = turn.get("turn_id")
        if not identity or identity in ids:
            raise ValueError("missing or duplicate evaluated turn identifier")
        ids.add(identity)
        status = turn.get("status")
        if status not in {"success", "failed", "timeout", "interrupted", "limit"}:
            raise ValueError("unknown turn completion status")
        counts["expected_turns"] += 1
        counts["completed_answers"] += status == "success"
        counts[f"status_{status}"] += 1
        counts["provider_attempt_failures"] += sum(item.get("status") == "failed" for item in turn.get("provider_attempts", []))
        if "task_checks" in turn:
            counts["task_scored_turns"] += 1
            complete = status == "success" and turn["task_checks"].get("complete") is True
            counts["completed_tasks"] += complete
        rejected = turn.get("answer_rejections", 0)
        if isinstance(rejected, bool) or not isinstance(rejected, int) or rejected < 0:
            raise ValueError("answer rejections must be a nonnegative integer")
        counts["answer_rejections"] += rejected
        kind = _error_kind(turn)
        if kind in {"provider", "tool", "runner", "judge"}:
            counts[f"{kind}_errors"] += 1
        if "elapsed" in turn:
            elapsed = _number(turn["elapsed"])
            if elapsed is None or elapsed < 0:
                raise ValueError("elapsed time must be a nonnegative finite number")
            total_elapsed += elapsed
            counts["timed_turns"] += 1
        for key in ("tool_calls", "model_responses"):
            value = turn.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
            counts[key] += value
        observations = turn.get("evidence", [])
        for observation in observations:
            if not isinstance(observation, dict):
                raise ValueError("observation must be an object")
            evidence_id = observation.get("evidence_id")
            if not evidence_id:
                raise ValueError("observation has no evidence identifier")
            identity = (turn.get("case_id"), evidence_id)
            marked_reused = observation.get("reused", False)
            if not isinstance(marked_reused, bool):
                raise ValueError("observation reused flag must be boolean")
            reused = marked_reused or identity in evidence_ids
            evidence_ids.add(identity)
            if reused:
                counts["reused_observations"] += 1
                continue
            counts["failed_tool_observations"] += observation.get("status") not in {"success", "partial"}
            source = observation.get("source")
            if source in {"live", "replay", "runner"}:
                counts[f"{source}_observations"] += 1
        checks = turn.get("claim_checks", {})
        if not isinstance(checks, dict):
            raise ValueError("claim checks must be an object")
        validated = {}
        for name in ("supported", "contradicted", "unverifiable", "requested_available", "requested_supported"):
            value = checks.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("invalid claim count")
            validated[name] = value
        if status != "success" or kind in {"provider", "judge"}:
            counts["excluded_claim_turns"] += 1
            continue
        counts.update(validated)
    checkable = counts["supported"] + counts["contradicted"]
    return {
        **{name: counts[name] for name in ("expected_turns", "completed_answers", "provider_errors", "tool_errors", "runner_errors", "judge_errors", "failed_tool_observations", "live_observations", "replay_observations", "runner_observations", "reused_observations", "excluded_claim_turns", "supported", "contradicted", "unverifiable", "requested_available", "requested_supported")},
        "statuses": {name.removeprefix("status_"): count for name, count in counts.items() if name.startswith("status_")},
        "completion_rate": counts["completed_answers"] / counts["expected_turns"] if counts["expected_turns"] else None,
        "task_scored_turns": counts["task_scored_turns"], "completed_tasks": counts["completed_tasks"],
        "provider_attempt_failures": counts["provider_attempt_failures"],
        "task_completion_rate": counts["completed_tasks"] / counts["task_scored_turns"] if counts["task_scored_turns"] else None,
        "answer_rejections": counts["answer_rejections"],
        "answer_rejection_rate": counts["answer_rejections"] / counts["expected_turns"] if counts["expected_turns"] else None,
        "timed_turns": counts["timed_turns"], "total_elapsed_seconds": total_elapsed,
        "mean_elapsed_seconds": total_elapsed / counts["timed_turns"] if counts["timed_turns"] else None,
        "attempted_tool_calls": counts["tool_calls"], "model_responses": counts["model_responses"],
        "claim_scope": "explicit_numeric_claims_only", "checkable_claims": checkable,
        "claim_support_rate": counts["supported"] / checkable if checkable else None,
        "requested_available_coverage": counts["requested_supported"] / counts["requested_available"] if counts["requested_available"] else None,
        "clinical_accuracy": None, "prose_faithfulness": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    csv_parser = commands.add_parser("csv", help="inspect a recorded tool CSV")
    csv_parser.add_argument("path", type=Path)
    csv_parser.add_argument("--kind", choices=("measurement", "classification"), default="measurement")
    transcript_parser = commands.add_parser("transcripts", help="inspect recorded JSONL without calling a model")
    transcript_parser.add_argument("path", type=Path)
    transcript_parser.add_argument("--history", choices=("without_gt", "with_gt"), default="without_gt")
    report_parser = commands.add_parser("report", help="summarize canonical evaluated-turn JSONL")
    report_parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "csv":
            outputs = load_tool_csv(args.path, args.kind)
            report = {"source": "replay", "records": len(outputs), "source_sha256": file_sha256(args.path), "statuses": dict(Counter(output.status for output in outputs.values())), "missing_values": dict(Counter(name for output in outputs.values() for name in output.missingness))}
        elif args.command == "transcripts":
            counts: Counter = Counter()
            for case in iter_transcripts(args.path, history=args.history):
                counts["source_rows"] += 1
                counts[case.schema] += 1
                counts["user_turns"] += len(case.turns)
                counts["tool_results"] += sum(len(turn.observations) for turn in case.turns)
                counts["additional_replicates"] += case.replicate > 1
                counts["rows_with_nap_labels"] += legacy_nap(case)["status"] != "unavailable"
            report = {"source": "replay", "source_sha256": file_sha256(args.path), **dict(counts)}
        else:
            with args.path.open(encoding="utf-8") as source:
                report = summarize(json.loads(line) for line in source if line.strip())
    except (OSError, ValueError) as error:
        parser.exit(2, f"evaluation: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
