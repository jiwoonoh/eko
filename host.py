"""A basic CLI for the ECG research runtime."""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
from pathlib import Path
import sys

from contracts import STANDARD_LEADS
from traces import EventLog


def lead_order(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if value == "standard":
        return STANDARD_LEADS
    return tuple(part.strip() for part in value.split(","))


def show_result(result, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), allow_nan=False))
        return
    if result.answer:
        print(result.answer)
    if result.error:
        print(f"{result.status}: {result.error}", file=sys.stderr)
    for observation in result.evidence:
        if observation.status not in ("success", "partial"):
            continue
        print(f"\nEvidence {observation.evidence_id} ({observation.status})")
        findings = observation.findings.items()
        if observation.tool == "call_classification_tool":
            from classifier import label_for_key
            findings = sorted(findings, key=lambda item: (-item[1], item[0]))[:5]
            print("  Five highest classifier scores; all 150 are available in JSON output and the log.")
        for name, value in findings:
            if value is None:
                print(f"  {name}: unavailable ({observation.missingness.get(name, 'unknown reason')})")
            else:
                label = label_for_key(name) if observation.tool == "call_classification_tool" else name
                print(f"  {label}: {value:g} {observation.units.get(name, '')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="omit for an interactive conversation")
    parser.add_argument("--model", required=True, help="explicit OpenRouter tool-capable model ID")
    parser.add_argument("--ecg", type=Path, required=True, help="known full 12-lead PTB-XL MAT record")
    parser.add_argument("--lead-order", help="standard, or twelve comma-separated lead names; confirms layout when MAT metadata is absent")
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--json", action="store_true", help="emit a TurnResult JSON object per question")
    parser.add_argument("--max-responses", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=4)
    parser.add_argument("--tool-timeout", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--classifier", choices=("ecgfounder",), help="enable the verified optional classifier")
    parser.add_argument("--checkpoint", type=Path, help="local ECGFounder checkpoint; never sent to the language model")
    args = parser.parse_args(argv)
    if bool(args.classifier) != bool(args.checkpoint):
        parser.error("--classifier and --checkpoint must be provided together")
    for name in ("max_responses", "max_tool_calls", "tool_timeout", "request_timeout", "max_tokens"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")

    try:
        # Optional numerical dependencies are loaded only when selecting a record.
        from records import load_record
        record, signal = load_record(args.ecg, lead_order(args.lead_order))
        del signal
    except (ImportError, ValueError, OSError) as error:
        parser.error(str(error))

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and sys.stdin.isatty():
        try:
            api_key = getpass.getpass("OpenRouter API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 130
    if not api_key:
        parser.error("set OPENROUTER_API_KEY; an interactive terminal can also prompt for it")

    from eko import Session, run_turn
    from providers import OpenRouter
    from worker import ToolWorker

    config = {
        "model": args.model,
        "history": "without_gt",
        "max_responses": args.max_responses,
        "max_tool_calls": args.max_tool_calls,
        "tool_timeout": args.tool_timeout,
        "request_timeout": args.request_timeout,
        "max_tokens": args.max_tokens,
        "classifier": args.classifier,
    }
    with EventLog(args.log_dir, record, config) as log:
        worker = ToolWorker(checkpoint=args.checkpoint)
        if args.classifier:
            try:
                ready = worker.prepare_classifier(record, timeout=args.tool_timeout)
            except (TimeoutError, KeyboardInterrupt) as error:
                worker.close()
                ready = False
                worker.classifier_error = str(error) or "classifier preparation interrupted"
                if isinstance(error, KeyboardInterrupt):
                    return 130
            log("classifier_readiness", ready=ready, provenance=worker.classifier_provenance,
                error=worker.classifier_error)
            if not ready:
                print(f"Classification unavailable: {worker.classifier_error}", file=sys.stderr)
        model = OpenRouter(args.model, api_key=api_key, max_tokens=args.max_tokens,
                           timeout=args.request_timeout)
        session = Session(model, worker, record, log=log,
                          max_responses=args.max_responses,
                          max_tool_calls=args.max_tool_calls,
                          tool_timeout=args.tool_timeout,
                          request_timeout=args.request_timeout)
        print(f"Record: {record.record_id}; log: {log.path}", file=sys.stderr)
        try:
            if args.question is not None:
                result = run_turn(session, args.question)
                show_result(result, as_json=args.json)
                return 0 if result.status == "success" else (130 if result.status == "interrupted" else 1)
            capability = "Measurements and classifier scores are available." if worker.classifier_ready else "Classification is unavailable."
            print(f"Ask about this ECG. {capability} Ctrl+D exits.", file=sys.stderr)
            while True:
                try:
                    question = input("You: ")
                except EOFError:
                    return 0
                except KeyboardInterrupt:
                    print(file=sys.stderr)
                    continue
                if not question.strip():
                    continue
                result = run_turn(session, question)
                show_result(result, as_json=args.json)
        finally:
            session.close()
            log("run_finished")


if __name__ == "__main__":
    raise SystemExit(main())
