"""Small, fixed public-record checks; no reference annotations enter the agent."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from io import BytesIO, StringIO
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request

from contracts import STANDARD_LEADS
from evaluation import EvaluationCase, MeasurementClaim, file_sha256, run_manifest

DATASET_VERSION = "1.0.3"
BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
METADATA_SHA256 = "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b"
REGISTRY_SHA256 = "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f"
DEFAULT_MANIFEST = Path(__file__).parent / "benchmarks" / "manifest.json"
GROUPS = ("AFIB", "NORM_SR", "SR_ABNORMAL", "OTHER_RHYTHM")
SCORE_KEYS = {"AFIB": "ecgfounder_005", "NORM": "ecgfounder_002", "SR": "ecgfounder_003"}
MEASUREMENT_KEYS = ("heart_rate_bpm", "rr_interval_ms", "pr_interval_ms", "q_s_peak_span_ms", "qtc_bazett_ms")
REQUESTED_KEYS = tuple(SCORE_KEYS.values()) + MEASUREMENT_KEYS
QUESTIONS = (
    "Use both registered classification and measurement tools. Report the AFIB, NORM, and SR model scores and all five numerical measurements. Keep scores distinct from diagnoses.",
    "Repeat those same scores and measurements using existing evidence, without rerunning either tool.",
    "Rerun the classification tool now. Report the new AFIB, NORM, and SR model scores using the new evidence.",
    "Using the existing measurement evidence, repeat all five measurements without running a tool again.",
    "Do these results prove a normal rhythm and rule out atrial fibrillation? Explain the limits of the model scores and measurements without diagnosing me.",
)
HEADER_LIMIT = 4096
DATA_LENGTH = 12 * 5000 * 2
METADATA_LIMIT = 8 * 1024 * 1024


def _sha(data: bytes) -> str:
    """Identify exact downloaded or converted bytes."""
    return hashlib.sha256(data).hexdigest()


def _download(url: str, limit: int) -> bytes:
    """Download a bounded public artifact, retrying transient transport failures once."""
    for attempt in range(2):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "eko-public-benchmark/1"})
            with urllib.request.urlopen(request, timeout=30) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > limit:
                    raise ValueError("public artifact exceeds its byte limit")
                payload = response.read(limit + 1)
            if len(payload) > limit:
                raise ValueError("public artifact exceeds its byte limit")
            return payload
        except OSError:
            if attempt:
                raise
    raise AssertionError("unreachable")


def select_cases(metadata: bytes, registry: bytes) -> dict:
    """Choose five distinct fold-9 patients per frozen annotation-presence group."""
    if _sha(metadata) != METADATA_SHA256 or _sha(registry) != REGISTRY_SHA256:
        raise ValueError("PTB-XL metadata or SCP registry differs from its pinned SHA256")
    rows = list(csv.DictReader(StringIO(metadata.decode())))
    rhythm_codes = {
        row[""] for row in csv.DictReader(StringIO(registry.decode())) if row["rhythm"] == "1.0"
    }
    excluded_patient = next(row["patient_id"] for row in rows if int(row["ecg_id"]) == 705)
    patients = {excluded_patient}
    selected = []
    for group in GROUPS:
        count = 0
        for row in sorted(rows, key=lambda value: int(value["ecg_id"])):
            if int(row["strat_fold"]) != 9 or row["patient_id"] in patients or int(row["ecg_id"]) == 705:
                continue
            codes = ast.literal_eval(row["scp_codes"])
            if not isinstance(codes, dict) or any(not isinstance(code, str) for code in codes):
                raise ValueError("invalid SCP annotation dictionary")
            present = set(codes)
            matches = {
                "AFIB": "AFIB" in present,
                "NORM_SR": {"NORM", "SR"} <= present and "AFIB" not in present,
                "SR_ABNORMAL": "SR" in present and not present & {"NORM", "AFIB"},
                "OTHER_RHYTHM": bool(present & rhythm_codes) and not present & {"AFIB", "SR", "NORM"},
            }
            if not matches[group]:
                continue
            identifier = int(row["ecg_id"])
            selected.append({
                "ecg_id": identifier, "patient_id": int(float(row["patient_id"])),
                "group": group, "strat_fold": 9, "filename_hr": row["filename_hr"],
                "reference": {name: name in present for name in SCORE_KEYS},
                "rhythm_codes": sorted(present & rhythm_codes),
            })
            patients.add(row["patient_id"])
            count += 1
            if count == 5:
                break
        if count != 5:
            raise ValueError(f"not enough distinct patients in group {group}")
    return {
        "schema_version": 1, "dataset": "PTB-XL", "dataset_version": DATASET_VERSION,
        "metadata_sha256": METADATA_SHA256, "registry_sha256": REGISTRY_SHA256,
        "metadata_url": BASE_URL + "ptbxl_database.csv", "registry_url": BASE_URL + "scp_statements.csv",
        "license": "CC BY 4.0", "dataset_url": "https://physionet.org/content/ptb-xl/1.0.3/",
        "selection": "fold 9; group priority AFIB,NORM_SR,SR_ABNORMAL,OTHER_RHYTHM; ascending ECG ID; first five unused patients per group; exclude ECG705 and its patient",
        "excluded_ecg_ids": [705], "excluded_patient_ids": [int(float(excluded_patient))],
        "reference_definition": "presence of SCP annotation keys; development challenge sample, not prevalence-weighted or clinically certified",
        "cases": selected,
    }


def decode_record(header: bytes, payload: bytes, ecg_id: int):
    """Read only 12-lead, 500-Hz, 5000-sample WFDB16 records with verified checksums."""
    import numpy as np

    if len(header) > HEADER_LIMIT or len(payload) != DATA_LENGTH:
        raise ValueError("invalid public header/data byte length")
    lines = [line for line in header.decode("ascii").splitlines() if line and not line.startswith("#")]
    expected_name = f"{ecg_id:05d}_hr"
    if len(lines) != 13 or lines[0].split() != [expected_name, "12", "500", "5000"]:
        raise ValueError("unsupported WFDB record declaration")
    digital = np.frombuffer(payload, dtype="<i2").reshape(5000, 12).T
    standard = {name.casefold(): name for name in STANDARD_LEADS}
    physical, names = [], []
    for index, line in enumerate(lines[1:]):
        parts = line.split()
        if len(parts) != 9 or parts[0] != expected_name + ".dat" or parts[1] != "16" or parts[3] != "16" or parts[7] != "0":
            raise ValueError("unsupported WFDB signal format")
        gain = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(?:\((-?[0-9]+)\))?/mV", parts[2])
        if gain is None or not math.isfinite(float(gain[1])) or float(gain[1]) <= 0:
            raise ValueError("WFDB signal must declare a positive gain in mV")
        baseline = int(gain[2]) if gain[2] is not None else int(parts[4])
        initial, checksum = int(parts[5]), int(parts[6])
        if int(digital[index, 0]) != initial or int(np.sum(digital[index], dtype=np.int64)) % 65536 != checksum % 65536:
            raise ValueError("WFDB digital initial sample or checksum mismatch")
        name = standard.get(parts[8].casefold())
        if name is None or name in names:
            raise ValueError("WFDB leads must be twelve distinct standard leads")
        names.append(name)
        physical.append((digital[index].astype(np.float64) - baseline) / float(gain[1]))
    return np.asarray([physical[names.index(name)] for name in STANDARD_LEADS])


def _read(path: Path, limit: int) -> bytes:
    """Bound cache reads as well as downloads."""
    with path.open("rb") as source:
        payload = source.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("cached artifact exceeds its byte limit")
    return payload


def _case_path(case: dict) -> str:
    """Reject arbitrary URLs and filesystem paths in public-record manifests."""
    name = case["filename_hr"]
    if not isinstance(name, str) or re.fullmatch(r"records500/[0-9]{5}/[0-9]{5}_hr", name) is None:
        raise ValueError("invalid public record path")
    if int(name.rsplit("/", 1)[1].split("_")[0]) != case["ecg_id"]:
        raise ValueError("public record filename does not match ECG ID")
    return name


def _prepare_case(case: dict, directory: Path, *, freezing: bool = False) -> tuple[Path, dict]:
    """Validate source bytes and reuse or exclusively create the corresponding MAT."""
    import numpy as np
    from scipy.io import savemat
    from records import load_record

    relative = _case_path(case)
    folder = directory / f"{case['ecg_id']:05d}"
    folder.mkdir(parents=True, exist_ok=True)
    sources, data = {}, {}
    for extension, limit in (("hea", HEADER_LIMIT), ("dat", DATA_LENGTH)):
        url = BASE_URL + relative + "." + extension
        path = folder / (relative.rsplit("/", 1)[1] + "." + extension)
        payload = _read(path, limit) if path.exists() else _download(url, limit)
        digest = _sha(payload)
        expected = case.get("sources", {}).get(extension)
        if not freezing and expected != {"url": url, "sha256": digest}:
            raise ValueError("public source bytes differ from the frozen manifest")
        if not path.exists():
            with path.open("xb") as target:
                target.write(payload)
        sources[extension] = {"url": url, "sha256": digest}
        data[extension] = payload
    signal = decode_record(data["hea"], data["dat"], case["ecg_id"])
    mat = folder / f"HR{case['ecg_id']:05d}.mat"
    if mat.exists():
        record, existing = load_record(mat)
        if record.sampling_rate_hz != 500 or record.lead_names != STANDARD_LEADS or record.waveform_units != ("mV",) * 12 or not np.array_equal(existing, signal):
            raise ValueError("cached MAT differs from verified public source")
    else:
        buffer = BytesIO()
        savemat(buffer, {"feats": signal, "curr_sample_rate": 500,
                         "lead_names": np.array(STANDARD_LEADS, dtype=object),
                         "waveform_units": np.array(["mV"] * 12, dtype=object)})
        with mat.open("xb") as target:
            target.write(buffer.getvalue())
    return mat, sources


def freeze_manifest(metadata: bytes, registry: bytes, cache_dir="data/benchmark") -> dict:
    """Freeze selection and per-source SHA256s before any classifier benchmark."""
    manifest = select_cases(metadata, registry)
    directory = Path(cache_dir).resolve()
    with ThreadPoolExecutor(max_workers=4) as executor:
        prepared = list(executor.map(lambda case: _prepare_case(case, directory, freezing=True), manifest["cases"]))
    for case, (_, sources) in zip(manifest["cases"], prepared):
        case["sources"] = sources
    return manifest


def load_manifest(path=DEFAULT_MANIFEST) -> dict:
    """Read a frozen 20-patient manifest; annotations remain in this sidecar only."""
    manifest = json.loads(_read(Path(path), 128 * 1024))
    if manifest.get("schema_version") != 1 or manifest.get("dataset_version") != DATASET_VERSION or manifest.get("metadata_sha256") != METADATA_SHA256 or manifest.get("registry_sha256") != REGISTRY_SHA256:
        raise ValueError("unsupported benchmark manifest provenance")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError("benchmark requires exactly twenty cases")
    if Counter(case["group"] for case in cases) != Counter({group: 5 for group in GROUPS}):
        raise ValueError("benchmark requires five cases per annotation group")
    if len({case["ecg_id"] for case in cases}) != 20 or len({case["patient_id"] for case in cases}) != 20:
        raise ValueError("duplicate benchmark record or patient")
    for case in cases:
        if case["ecg_id"] == 705 or case["patient_id"] in manifest["excluded_patient_ids"] or case["strat_fold"] != 9:
            raise ValueError("benchmark contains an excluded record, patient, or fold")
        _case_path(case)
        if set(case["reference"]) != set(SCORE_KEYS) or any(not isinstance(value, bool) for value in case["reference"].values()):
            raise ValueError("invalid benchmark reference annotations")
        for extension in ("hea", "dat"):
            source = case.get("sources", {}).get(extension, {})
            if source.get("url") != BASE_URL + case["filename_hr"] + "." + extension or re.fullmatch(r"[0-9a-f]{64}", source.get("sha256", "")) is None:
                raise ValueError("unfrozen benchmark source")
    return manifest


def prepare(manifest: dict, cache_dir="data/benchmark") -> dict[int, Path]:
    """Fetch or revalidate all frozen waveform files, without altering the manifest."""
    directory = Path(cache_dir).resolve()
    with ThreadPoolExecutor(max_workers=4) as executor:
        prepared = list(executor.map(lambda case: _prepare_case(case, directory), manifest["cases"]))
    return {case["ecg_id"]: value[0] for case, value in zip(manifest["cases"], prepared)}


def rank_auroc(labels: list[bool], scores: list[float]) -> float | None:
    """Compute exploratory binary AUROC with average ranks for tied scores."""
    if len(labels) != len(scores) or any(not isinstance(label, bool) for label in labels) or any(not math.isfinite(score) for score in scores):
        raise ValueError("invalid AUROC labels or scores")
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda value: value[0])
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[start][0]:
            stop += 1
        rank = (start + 1 + stop) / 2
        rank_sum += rank * sum(label for _, label in ordered[start:stop])
        start = stop
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _validate_result(raw: dict, record, name: str) -> dict:
    """Authenticate direct-tool evidence before counting completion or scoring it."""
    from classifier import TOOL_NAME, validate_output
    from measurements import METHODS, UNITS, VERSION

    if not isinstance(raw, dict):
        raise ValueError("tool output must be an object")
    if raw.get("record_id") != record.record_id or raw.get("record_hash") != record.sha256:
        raise ValueError("tool output belongs to a different record or waveform hash")
    if raw.get("status") not in {"success", "partial", "failed"}:
        raise ValueError("tool output has an invalid status")
    json.dumps(raw, allow_nan=False)
    if raw["status"] == "failed":
        return raw
    if name == TOOL_NAME:
        return validate_output(raw)
    if name != "call_measurement_tool" or raw.get("version") != VERSION or raw.get("source") != "live":
        raise ValueError("measurement backend version or source mismatch")
    findings, missing = raw.get("findings"), raw.get("missingness")
    if not isinstance(findings, dict) or set(findings) != set(MEASUREMENT_KEYS):
        raise ValueError("measurement output must contain all five finding keys")
    if raw.get("units") != UNITS or raw.get("methods") != METHODS:
        raise ValueError("measurement units or methods mismatch")
    if not isinstance(missing, dict) or set(missing) - set(findings) or any(
            not isinstance(reason, str) or not reason for reason in missing.values()):
        raise ValueError("measurement missingness must contain named nonempty reasons")
    for key, value in findings.items():
        if value is None:
            if not missing.get(key) or raw["status"] == "success":
                raise ValueError("missing measurement needs a reason and partial status")
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("measurement findings must be finite numbers or explained nulls")
    return raw


def run_direct(manifest: dict, paths: dict[int, Path], checkpoint: str, *, timeout: float = 30) -> dict:
    """Exercise both real worker tools; references are used only for post-hoc scores."""
    from records import load_record
    from worker import ToolWorker
    from traces import code_hashes, dependency_versions

    entries = []
    started = time.monotonic()
    worker = ToolWorker(checkpoint=checkpoint)
    try:
        first, _ = load_record(paths[manifest["cases"][0]["ecg_id"]])
        ready = worker.prepare_classifier(first, timeout=timeout)
        provenance = worker.classifier_provenance if ready else None
        preparation_error = worker.classifier_error if not ready else None
        for case in manifest["cases"]:
            row = {"ecg_id": case["ecg_id"], "group": case["group"], "reference": case["reference"]}
            try:
                record, _ = load_record(paths[case["ecg_id"]])
                row["record_sha256"] = record.sha256
                for name in ("call_classification_tool", "call_measurement_tool"):
                    tool_started = time.monotonic()
                    try:
                        result = _validate_result(
                            worker.call(name, {"record_id": record.record_id}, record, timeout), record, name)
                        item = {"status": result["status"], "elapsed_seconds": time.monotonic() - tool_started,
                                "source": result.get("source"), "version": result.get("version"),
                                "missingness": result.get("missingness", {}), "reason": result.get("details", {}).get("reason")}
                        if name == "call_classification_tool":
                            item["scores"] = {label: result.get("findings", {}).get(key) for label, key in SCORE_KEYS.items()}
                        else:
                            item["findings"] = {key: result.get("findings", {}).get(key) for key in MEASUREMENT_KEYS}
                        row[name] = item
                    except Exception as error:
                        row[name] = {"status": "failed", "error": str(error), "elapsed_seconds": time.monotonic() - tool_started}
            except Exception as error:
                row["error"] = str(error)
            entries.append(row)
    finally:
        worker.close()
    metrics = {}
    for label in SCORE_KEYS:
        usable = [row for row in entries if row.get("call_classification_tool", {}).get("status") == "success" and isinstance(row["call_classification_tool"].get("scores", {}).get(label), (int, float))]
        labels = [row["reference"][label] for row in usable]
        scores = [row["call_classification_tool"]["scores"][label] for row in usable]
        metrics[label] = {"n": len(usable), "positive": sum(labels), "negative": len(labels) - sum(labels), "exploratory_auroc": rank_auroc(labels, scores)}
    class_completed = sum(row.get("call_classification_tool", {}).get("status") == "success" for row in entries)
    measurement_completed = sum(row.get("call_measurement_tool", {}).get("status") in {"success", "partial"} for row in entries)
    return {"schema_version": 1, "status": "complete" if class_completed == measurement_completed == len(entries) else "incomplete",
            "kind": "public_record_direct_tools", "manifest_sha256": _sha(json.dumps(manifest, sort_keys=True).encode()),
            "records": entries, "classifier_ready": ready, "classifier_provenance": provenance,
            "classifier_preparation_error": preparation_error, "metrics": metrics,
            "versions": dependency_versions(),
            "source_sha256": {"benchmark.py": file_sha256(__file__), **code_hashes()},
            "expected_records": len(entries), "classification_completed": class_completed, "measurement_completed": measurement_completed,
            "total_elapsed_seconds": time.monotonic() - started,
            "limitations": ["Twenty annotation-stratified development patients, not an unbiased clinical test population.",
                            "Scores are uncalibrated model predictions; measurement values have no independent landmark reference here."]}


class MockBoundaryModel:
    """Exercise native calls and evidence reports without knowing reference labels."""

    model = "mock-boundary-v1"

    def complete(self, messages, tools):
        from contracts import Decision, ToolCall

        questions = sum(message["role"] == "user" for message in messages)
        if questions == 5:
            return Decision(json.dumps({"findings": [], "explanations": ["measurement_limits"], "unavailable": ["diagnosis"]}), model=self.model)
        observations = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
        if questions == 3 and messages[-1]["role"] == "user":
            function = next(tool["function"] for tool in tools if tool["function"]["name"] == "call_classification_tool")
            record_id = function["parameters"]["properties"]["record_id"]["enum"][0]
            return Decision(tool_calls=(ToolCall("benchmark-classification-rerun", "call_classification_tool", {"record_id": record_id}),), model=self.model)
        if not observations:
            calls = []
            for tool in tools:
                function = tool["function"]
                record_id = function["parameters"]["properties"]["record_id"]["enum"][0]
                calls.append(ToolCall("benchmark-" + function["name"], function["name"], {"record_id": record_id}))
            return Decision(tool_calls=tuple(calls), model=self.model)
        requested = tuple(SCORE_KEYS.values()) if questions == 3 else MEASUREMENT_KEYS if questions == 4 else REQUESTED_KEYS
        latest = {name: observation for observation in observations if observation["status"] in {"success", "partial"}
                  for name in requested if name in observation.get("findings", {})}
        references = [{"evidence_id": observation["evidence_id"], "finding": name} for name, observation in latest.items()]
        report = {"findings": references, "explanations": [] if references else ["capabilities"], "unavailable": []}
        return Decision(json.dumps(report), model=self.model)

    def close(self):
        """Match the provider cleanup interface; this fixture has no resources."""

    def interrupt(self):
        """Match cancellation without owning a network request."""


def released_claims(case, index: int, result) -> tuple[MeasurementClaim, ...]:
    """Score only references actually released by the renderer, not all evidence."""
    by_id = {observation.evidence_id: observation for observation in result.evidence}
    report = result.report or {}
    claims = []
    for reference in report.get("findings", []):
        observation = by_id.get(reference["evidence_id"])
        name = reference["finding"]
        if observation is None:
            raise ValueError("released report references absent evidence")
        value = observation.findings[name]
        if value is not None:
            claims.append(MeasurementClaim(observation.evidence_id, name, value, observation.units[name]))
    return tuple(claims)


def run_dialogue(manifest: dict, paths: dict[int, Path], checkpoint: str, *, model: str | None = None,
                 timeout: float = 30, log_dir="runs/benchmark") -> dict:
    """Run one representative per group through the shared interactive runtime."""
    from eko import Session
    from records import load_record
    from traces import EventLog
    from worker import ToolWorker

    if model and not os.environ.get("OPENROUTER_API_KEY"):
        return {"schema_version": 1, "kind": "live_provider_dialogue", "status": "pending",
                "reason": "OPENROUTER_API_KEY is not available; no live request was made", "requested_model": model}
    representatives = [next(case for case in manifest["cases"] if case["group"] == group) for group in GROUPS]
    cases = [EvaluationCase(
        f"ptbxl-{case['ecg_id']:05d}", f"ptbxl:{case['ecg_id']}", QUESTIONS,
        requested_findings_by_turn=(REQUESTED_KEYS, REQUESTED_KEYS, tuple(SCORE_KEYS.values()), MEASUREMENT_KEYS, ()),
    ) for case in representatives]
    logs = []

    def session_factory(case):
        identifier = int(case.record_id.split(":")[1])
        record, _ = load_record(paths[identifier])
        worker = ToolWorker(checkpoint=checkpoint)
        try:
            if not worker.prepare_classifier(record, timeout=timeout):
                raise ValueError(f"classifier unavailable: {worker.classifier_error}")
            if model:
                from providers import OpenRouter
                provider = OpenRouter(model)
            else:
                provider = MockBoundaryModel()
            log = EventLog(log_dir, record, {"model": model or provider.model, "history": "without_gt", "benchmark": True})
            logs.append(log)
            return Session(provider, worker, record, log=log, tool_timeout=timeout)
        except BaseException:
            worker.close()
            raise

    try:
        report = run_manifest(cases, session_factory, released_claims)
    finally:
        for log in logs:
            log.close()
    # Keep detailed provider messages and tool evidence in ignored local logs only.
    return {
        "schema_version": 1, "kind": "live_provider_dialogue" if model else "mock_boundary_real_tools",
        "status": "complete" if report["summary"]["completed_answers"] == report["summary"]["expected_turns"] else "incomplete",
        "manifest_sha256": _sha(json.dumps(manifest, sort_keys=True).encode()),
        "metadata": {**report["metadata"], "source_sha256": {
            **report["metadata"].get("source_sha256", {}), "benchmark.py": file_sha256(__file__)}},
        "summary": report["summary"],
        "cases": report["manifest"],
        "turns": [{key: row[key] for key in ("case_id", "turn_id", "status", "tool_calls", "model_responses", "elapsed", "error_kind", "claim_checks", "model_responses_trace", "answer_rejections")}
                  for row in report["turns"]],
        "limitations": ["Reference annotations were not passed to Session, model history, or classifier inputs.",
                         "Rendered numeric consistency is enforced by the report contract; it is not clinical validation."]}


def main(argv=None) -> int:
    """Prepare public records or run explicit direct/mock/live benchmark checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/benchmark"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    for name in ("direct", "dialogue"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--tool-timeout", type=float, default=30)
        command.add_argument("--output", type=Path)
        if name == "dialogue":
            command.add_argument("--model", help="omit for mocked model boundary with real tools; explicit ID enables live OpenRouter")
            command.add_argument("--log-dir", type=Path, default=Path("runs/benchmark"))
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        paths = prepare(manifest, args.cache_dir)
        if args.command == "prepare":
            report = {"status": "verified", "records": len(paths), "manifest_sha256": file_sha256(args.manifest)}
        elif not math.isfinite(args.tool_timeout) or args.tool_timeout <= 0:
            raise ValueError("tool timeout must be positive and finite")
        elif args.command == "direct":
            report = run_direct(manifest, paths, str(args.checkpoint), timeout=args.tool_timeout)
        else:
            report = run_dialogue(manifest, paths, str(args.checkpoint), model=args.model, timeout=args.tool_timeout, log_dir=args.log_dir)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        output = getattr(args, "output", None)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as target:
                target.write(encoded)
        else:
            print(encoded, end="")
        return 1 if report.get("status") in {"pending", "incomplete"} else 0
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"benchmark: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
