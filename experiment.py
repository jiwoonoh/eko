"""Run a frozen reporting experiment; each label identifies one source revision."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import uuid

from benchmark import load_manifest, prepare, run_dialogue


def load_config(path: Path) -> dict:
    """Validate explicit experimental controls, without reading any credentials."""
    config = json.loads(path.read_text())
    required = {"schema_version", "model", "checkpoint", "manifest", "repeats", "limits"}
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("unsupported experiment configuration")
    for name in ("model", "checkpoint", "manifest"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"experiment needs an explicit {name}")
    if type(config["repeats"]) is not int or not 1 <= config["repeats"] <= 10:
        raise ValueError("experiment repeats must be an integer from 1 to 10")
    limits = config["limits"]
    if not isinstance(limits, dict) or set(limits) != {
            "max_responses", "max_tool_calls", "tool_timeout", "request_timeout", "max_tokens"}:
        raise ValueError("experiment needs all five runtime limits")
    for name, value in limits.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid experiment limit: {name}")
        if name.startswith("max_") and type(value) is not int:
            raise ValueError(f"experiment limit must be an integer: {name}")
    return config


def load_key(path: Path) -> None:
    """Load only OPENROUTER_API_KEY as data; never execute the key file."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    if path.is_file():
        with path.open() as source:
            content = source.read(8193)
        if len(content) > 8192:
            raise ValueError("key file exceeds its size limit")
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            name, separator, value = line.partition("=")
            if separator and name.strip() == "OPENROUTER_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value:
                    os.environ["OPENROUTER_API_KEY"] = value
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY is unavailable")


def summarize(reports: list[dict]) -> dict:
    """Keep full task fulfillment separate from renderer acceptance and API failures."""
    totals = {name: sum(report["summary"][name] for report in reports) for name in (
        "expected_turns", "completed_answers", "completed_tasks", "answer_rejections",
        "provider_errors", "tool_errors", "attempted_tool_calls", "model_responses")}
    totals["task_completion_rate"] = totals["completed_tasks"] / totals["expected_turns"]
    usage = [item.get("usage", {}) for report in reports for turn in report["turns"]
             for item in (turn.get("provider_attempts") or turn["model_responses_trace"])]
    totals["reported_cost_usd"] = sum(float(item.get("cost") or 0) for item in usage)
    totals["prompt_tokens"] = sum(item.get("prompt_tokens", 0) for item in usage)
    totals["elapsed_seconds"] = sum(report["summary"]["total_elapsed_seconds"] for report in reports)
    totals["tasks_by_repeat"] = [report["summary"]["completed_tasks"] for report in reports]
    return totals


def run(config: dict, label: str, output_dir: Path) -> Path:
    """Run the identical frozen dialogue suite sequentially and save every result."""
    if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", label) is None:
        raise ValueError("experiment label must be a short filename component")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = output_dir / f"{stamp}-{label}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(config, sort_keys=True, allow_nan=False).encode()
    manifest = load_manifest(config["manifest"])
    paths = prepare(manifest)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    reports = []
    for index in range(1, config["repeats"] + 1):
        print(json.dumps({"experiment": label, "repeat": index, "status": "started"}), flush=True)
        report = run_dialogue(manifest, paths, config["checkpoint"], model=config["model"],
                              timeout=config["limits"]["tool_timeout"], settings=config["limits"],
                              log_dir=directory / "events" / str(index))
        if "summary" not in report:
            raise ValueError("experiment did not execute; check local key configuration")
        report["experiment"] = {"label": label, "repeat": index, "revision": revision,
                                "config_sha256": hashlib.sha256(encoded).hexdigest(),
                                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        with (directory / f"repeat-{index}.json").open("x") as target:
            json.dump(report, target, indent=2, allow_nan=False)
            target.write("\n")
        reports.append(report)
        print(json.dumps({"repeat": index, **summarize([report])}), flush=True)
    summary = {"schema_version": 1, "label": label, "revision": revision,
               "config": config, "config_sha256": hashlib.sha256(encoded).hexdigest(),
               "results": summarize(reports), "recorded_at": datetime.now(timezone.utc).isoformat()}
    with (directory / "summary.json").open("x") as target:
        json.dump(summary, target, indent=2, allow_nan=False)
        target.write("\n")
    return directory


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--key-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ablations"))
    parser.add_argument("--repeats", type=int, help="explicit override, recorded in the saved configuration")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.repeats is not None:
            if not 1 <= args.repeats <= 10:
                raise ValueError("repeats must be from 1 to 10")
            config["repeats"] = args.repeats
        load_key(args.key_file)
        print(run(config, args.label, args.output_dir))
    except (OSError, ValueError) as error:
        message = str(error)
        key = os.environ.get("OPENROUTER_API_KEY")
        if key:
            message = message.replace(key, "[redacted]")
        parser.exit(1, f"experiment: {message}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
