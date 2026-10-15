"""A bounded ECG dialogue loop with named tools and persistent evidence."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import time
import uuid
from typing import Any, Callable

from contracts import Decision, Record, ToolCall, ToolObservation, TurnResult
from reporting import REPORT_INSTRUCTIONS, render_report, safe_report


def measurement_schema(record_id: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "call_measurement_tool",
            "description": (
                "Measure the selected ECG's heart rate, RR and PR intervals, "
                "Q-to-S peak span, and QTc. The Q-to-S peak span is the time "
                "between detected Q and S peaks; it is not a QRS onset-to-offset interval."
            ),
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string", "enum": [record_id]}},
                "required": ["record_id"],
                "additionalProperties": False,
            },
        },
    }


def _record_key(record: Record) -> str:
    return json.dumps(record.public(), sort_keys=True)


class Session:
    """Own one selected record, model history, and tool-produced evidence."""

    def __init__(self, model, worker, record: Record,
                 log: Callable[..., None] | None = None,
                 tools: list[dict[str, Any]] | None = None, *,
                 max_responses: int = 8, max_tool_calls: int = 4,
                 tool_timeout: float = 30.0, request_timeout: float = 60.0) -> None:
        if max_responses < 1 or max_tool_calls < 1:
            raise ValueError("response and tool-call limits must be positive")
        if not all(math.isfinite(value) and value > 0
                   for value in (tool_timeout, request_timeout)):
            raise ValueError("timeouts must be positive finite seconds")
        self.model = model
        self.worker = worker
        self.record = record
        self.log = log or (lambda _event_type, **_fields: None)
        self.max_responses = max_responses
        self.max_tool_calls = max_tool_calls
        self.tool_timeout = tool_timeout
        self.request_timeout = request_timeout
        self._custom_tools = deepcopy(tools)
        self._active = False
        self.history: list[dict[str, Any]] = []
        self.evidence: list[ToolObservation] = []
        self._reset()

    def _reset(self) -> None:
        self._identity = _record_key(self.record)
        self.tools = deepcopy(self._custom_tools) if self._custom_tools is not None else [
            measurement_schema(self.record.record_id)]
        names = [tool["function"]["name"] for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        self._schemas = {tool["function"]["name"]: tool["function"]["parameters"]
                         for tool in self.tools}
        prompt = (
            "You are an ECG measurement assistant. Use only the registered tools. "
            "The selected record is fixed; pass its record_id, never file paths. "
            "Tool messages contain observations from the runner; never invent Tool_Output. "
            "Base numeric findings on those observations, preserve units and missingness, "
            "Describe Q-to-S peak span as a peak-span estimate, never as full QRS duration. "
            "Explain failed or unavailable analysis. Reuse evidence for follow-ups "
            "about this record. Classification and diagnostic interpretation are unavailable. "
            "Do not provide diagnosis or treatment advice.\n"
            f"{REPORT_INSTRUCTIONS}\n"
            f"Selected record: {json.dumps(self.record.public(), sort_keys=True)}"
        )
        self.history = [{"role": "system", "content": prompt}]
        self.evidence.clear()

    def set_record(self, record: Record) -> None:
        if self._active:
            raise RuntimeError("cannot change records during a turn")
        changed = _record_key(record) != self._identity
        self.record = record
        if changed:
            self._reset()
            self.log("record_selected", record=record.public())

    def close(self) -> None:
        try:
            self.model.close()
        finally:
            self.worker.close()


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("tool output contains a nonfinite number")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


def _arguments(session: Session, call: ToolCall) -> dict[str, Any]:
    if call.name not in session._schemas:
        raise ValueError(f"unavailable tool: {call.name}")
    args = call.arguments
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be an object")
    schema = session._schemas[call.name]
    properties = schema.get("properties", {})
    if set(args) - set(properties):
        raise ValueError("tool arguments include unsupported fields")
    if any(name not in args for name in schema.get("required", [])):
        raise ValueError("tool arguments omit required fields")
    if args.get("record_id") != session.record.record_id:
        raise ValueError("tool record_id does not match the selected record")
    if any(name in args for name in ("path", "ecg_path", "file_path", "filepath")):
        raise ValueError("model-supplied file paths are not accepted")
    types = {"string": str, "object": dict, "array": list, "boolean": bool,
             "integer": int, "number": (int, float)}
    for name, value in args.items():
        spec = properties[name]
        kind = spec.get("type")
        if kind in types and (not isinstance(value, types[kind]) or
                              (kind in ("integer", "number") and isinstance(value, bool))):
            raise ValueError(f"invalid type for tool argument {name}")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"invalid value for tool argument {name}")
    _finite(args)
    return deepcopy(args)


def _observation(session: Session, call: ToolCall, raw: dict[str, Any],
                 elapsed: float) -> ToolObservation:
    if not isinstance(raw, dict):
        raise ValueError("tool output must be an object")
    if raw.get("record_id") != session.record.record_id or raw.get("record_hash") != session.record.sha256:
        raise ValueError("tool output belongs to a different record or file hash")
    if raw.get("status") not in ("success", "partial", "failed"):
        raise ValueError("tool output has an invalid status")
    _finite(raw)
    findings = raw.get("findings", {})
    if not isinstance(findings, dict) or any(
            value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)))
            for value in findings.values()):
        raise ValueError("tool findings must contain numbers or null")
    for field in ("units", "methods", "missingness"):
        mapping = raw.get(field, {})
        if not isinstance(mapping, dict) or any(not isinstance(value, str) for value in mapping.values()):
            raise ValueError(f"tool {field} must map names to strings")
    if raw["status"] in ("success", "partial"):
        if not findings:
            raise ValueError("successful tool output has no findings")
        for name, value in findings.items():
            if value is None and not raw.get("missingness", {}).get(name):
                raise ValueError(f"missing finding {name} has no explanation")
            if value is not None and not raw.get("units", {}).get(name):
                raise ValueError(f"finding {name} has no unit")
    details = raw.get("details", {})
    if not isinstance(details, dict):
        raise ValueError("tool details must be an object")
    details = deepcopy(details)
    if raw["status"] == "failed" and not any(details.get(key) for key in ("error", "message", "reason")):
        reason = raw.get("error") or raw.get("message") or raw.get("reason")
        if reason:
            details["error"] = str(reason)
    duration = raw.get("elapsed", elapsed)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        raise ValueError("tool elapsed time must be a nonnegative number")
    version, source = raw.get("version", "unknown"), raw.get("source", "live")
    if not isinstance(version, str) or not version or not isinstance(source, str) or not source:
        raise ValueError("tool source and version must be nonempty strings")
    json.dumps(raw, allow_nan=False)
    return ToolObservation(
        record_id=session.record.record_id, record_hash=session.record.sha256,
        call_id=call.id, evidence_id=uuid.uuid4().hex, tool=call.name,
        version=version, status=raw["status"], findings=deepcopy(findings),
        units=deepcopy(raw.get("units", {})), methods=deepcopy(raw.get("methods", {})),
        missingness=deepcopy(raw.get("missingness", {})), details=deepcopy(details),
        elapsed=float(duration), source=source,
    )


def _assistant(decision: Decision) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": None}
    if decision.tool_calls:
        message["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {
                "name": call.name, "arguments": json.dumps(call.arguments, allow_nan=False)}}
            for call in decision.tool_calls]
    return message


def run_turn(session: Session, question: str) -> TurnResult:
    """Complete one user turn; every native tool call receives a matching result."""
    started = time.monotonic()
    turn_id = uuid.uuid4().hex
    responses = calls = 0
    pending: list[ToolCall] = []
    failed_observations: list[ToolObservation] = []
    unresolved: dict[str, str] = {}

    def finish(status: str, *, answer: str = "", error: str | None = None) -> TurnResult:
        result = TurnResult(turn_id, status, answer,
                            tuple(session.evidence) + tuple(failed_observations), error,
                            responses, calls, time.monotonic() - started)
        session.log("turn_end", **result.to_dict())
        return result

    def reject(call: ToolCall, reason: str) -> None:
        observation = ToolObservation(
            record_id=session.record.record_id, record_hash=session.record.sha256,
            call_id=call.id, evidence_id=uuid.uuid4().hex, tool=call.name,
            version="runner", status="failed", details={"error": reason}, source="runner")
        failed_observations.append(observation)
        session.history.append({"role": "tool", "tool_call_id": call.id,
                                "name": call.name, "content": json.dumps(observation.to_dict())})
        session.log("tool_error", turn_id=turn_id, observation=observation.to_dict(), error=reason)

    if session._active:
        raise RuntimeError("a session can run only one turn at a time")
    if _record_key(session.record) != session._identity:
        return finish("failed", error="runner: selected record changed; use set_record to reset history")
    if not isinstance(question, str) or not question.strip():
        return finish("failed", error="runner: question must be nonempty text")
    session._active = True
    session.history.append({"role": "user", "content": question})
    session.log("turn_start", turn_id=turn_id, question=question)
    try:
        while responses < session.max_responses:
            session.log("model_request", turn_id=turn_id,
                        requested_model=getattr(session.model, "model", None),
                        messages=deepcopy(session.history), tools=session.tools)
            if hasattr(session.model, "set_deadline"):
                session.model.set_deadline(time.monotonic() + session.request_timeout)
            try:
                decision = session.model.complete(deepcopy(session.history), deepcopy(session.tools))
            except TimeoutError as error:
                session.log("model_error", turn_id=turn_id, error=str(error),
                            provider_attempts=getattr(session.model, "last_attempts", []))
                return finish("timeout", error=f"provider: {error}")
            except Exception as error:
                session.log("model_error", turn_id=turn_id, error=str(error),
                            provider_attempts=getattr(session.model, "last_attempts", []))
                return finish("failed", error=f"provider: {error}")
            finally:
                if hasattr(session.model, "set_deadline"):
                    session.model.set_deadline(None)
            responses += 1
            if not isinstance(decision, Decision) or not isinstance(decision.content, str) or not isinstance(
                    decision.tool_calls, (tuple, list)) or not isinstance(decision.usage, dict):
                return finish("failed", error="provider: invalid decision")
            if any(not isinstance(call, ToolCall) or not isinstance(call.id, str) or
                   not call.id or not isinstance(call.name, str) or not call.name
                   for call in decision.tool_calls):
                return finish("failed", error="provider: invalid tool call")
            if len({call.id for call in decision.tool_calls}) != len(decision.tool_calls):
                return finish("failed", error="provider: duplicate tool-call IDs")
            try:
                assistant = _assistant(decision)
            except (ValueError, TypeError):
                return finish("failed", error="provider: tool arguments are not finite JSON")
            session.log("model_response", turn_id=turn_id, model=decision.model, usage=decision.usage,
                        provider_attempts=getattr(session.model, "last_attempts", []),
                        content=decision.content, tool_calls=[
                            {"id": call.id, "name": call.name, "arguments": call.arguments}
                            for call in decision.tool_calls])
            if not decision.tool_calls:
                if not decision.content.strip():
                    return finish("failed", error="provider: empty model response")
                try:
                    answer = render_report(decision.content, session.record, session.evidence,
                                           tool_names=set(session._schemas))
                except ValueError as error:
                    session.log("answer_rejected", turn_id=turn_id, reason=str(error))
                    answer = safe_report(session.record, session.evidence,
                                         tool_names=set(session._schemas))
                    session.history.append({"role": "assistant", "content": answer})
                    errors = [*unresolved.values(), f"runner: unsupported final response blocked: {error}"]
                    return finish("failed", answer=answer, error="; ".join(errors))
                session.history.append({"role": "assistant", "content": answer})
                error = "; ".join(unresolved.values()) if unresolved else None
                return finish("failed" if unresolved else "success", answer=answer,
                              error=error)
            session.history.append(assistant)
            pending = list(decision.tool_calls)
            exceeded = False
            while pending:
                call = pending[0]
                if calls >= session.max_tool_calls:
                    reject(call, "tool-call limit reached")
                    pending.pop(0)
                    exceeded = True
                    continue
                calls += 1
                session.log("tool_call", turn_id=turn_id, call_id=call.id, tool=call.name,
                            arguments=call.arguments)
                try:
                    args = _arguments(session, call)
                    tool_started = time.monotonic()
                    raw = session.worker.call(call.name, args, session.record, session.tool_timeout)
                    observation = _observation(session, call, raw, time.monotonic() - tool_started)
                except TimeoutError as error:
                    for unfinished in pending:
                        reject(unfinished, str(error) or "tool timed out")
                    pending.clear()
                    session.worker.close()
                    return finish("timeout", error=f"tool: {error}")
                except Exception as error:
                    reject(call, str(error))
                    unresolved[call.name] = f"tool: {error}"
                    pending.pop(0)
                    continue
                session.history.append({"role": "tool", "tool_call_id": call.id,
                                        "name": call.name,
                                        "content": json.dumps(observation.to_dict(), allow_nan=False)})
                session.log("tool_result", turn_id=turn_id, observation=observation.to_dict())
                if observation.status in ("success", "partial"):
                    session.evidence.append(observation)
                    unresolved.pop(call.name, None)
                else:
                    failed_observations.append(observation)
                    reason = (observation.details.get("error") or observation.details.get("message")
                              or observation.details.get("reason") or "analysis failed")
                    unresolved[call.name] = f"tool: {reason}"
                pending.pop(0)
            if exceeded:
                return finish("limit", error="runner: tool-call limit reached")
        return finish("limit", error="runner: model-response limit reached")
    except KeyboardInterrupt:
        session.model.interrupt()
        session.worker.close()
        for call in pending:
            reject(call, "turn interrupted")
        return finish("interrupted", error="runner: turn interrupted")
    finally:
        session._active = False
