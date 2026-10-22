"""Pinned ECGFounder CPU inference and an explicit, verified download command.

Importing this module does not import scientific dependencies or load weights.
The adapter follows the official PTB-XL 150-label evaluator, not its separate
downstream LVEF dataset. Scores are uncalibrated, independent sigmoid outputs.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import math
import os
from pathlib import Path
import re
import tempfile
import time
import urllib.request

from contracts import Record, STANDARD_LEADS
from ecgfounder_labels import LABEL_BYTES

TOOL_NAME = "call_classification_tool"
VERSION = "ecgfounder-150-v1"
SOURCE_REVISION = "04edac702b61c91face519774ddcc0cd712fef23"
MODEL_REVISION = "d9b1793951b2342f5f7e84f1ac03cd37f8a08724"
CHECKPOINT_NAME = "12_lead_ECGFounder.pth"
CHECKPOINT_BYTES = 369942585
CHECKPOINT_SHA256 = "ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997"
CHECKPOINT_URL = f"https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/{MODEL_REVISION}/{CHECKPOINT_NAME}"
LABELS_SHA256 = "7c087a0d383c7ea9ae9b818ad00a551b029334edfa5f1fbb5036bfbaa0df0c39"
PREPROCESSING_VERSION = "ecgfounder-ptbxl-global-zscore-v1"
SCORE_METHOD = "sigmoid_logit; uncalibrated multilabel model score"
TASK_COUNT = 150


def load_labels() -> tuple[str, ...]:
    payload = LABEL_BYTES
    if hashlib.sha256(payload).hexdigest() != LABELS_SHA256:
        raise ValueError("ECGFounder label manifest checksum mismatch")
    labels = tuple(payload.decode("utf-8").splitlines())
    if len(labels) != TASK_COUNT or len(set(labels)) != TASK_COUNT or any(not label for label in labels):
        raise ValueError("ECGFounder requires exactly 150 unique labels")
    return labels


def score_key(index: int) -> str:
    if type(index) is not int or not 0 <= index < TASK_COUNT:
        raise ValueError("ECGFounder label index must be an integer from 0 to 149")
    return f"ecgfounder_{index:03d}"


def label_for_key(key: str) -> str:
    if not isinstance(key, str) or re.fullmatch(r"ecgfounder_\d{3}", key) is None:
        raise ValueError("unknown ECGFounder finding key")
    index = int(key.rsplit("_", 1)[1])
    if score_key(index) != key:
        raise ValueError("unknown ECGFounder finding key")
    return load_labels()[index]


def provenance() -> dict:
    return {
        "backend": "ecgfounder",
        "source_revision": SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "label_map_sha256": LABELS_SHA256,
        "preprocessing_version": PREPROCESSING_VERSION,
        "device": "cpu",
        "task_count": TASK_COUNT,
        "output_transform": "sigmoid",
    }


def validate_output(raw: dict) -> dict:
    """Validate complete scores and pinned provenance before exposing evidence."""
    load_labels()
    if not isinstance(raw, dict) or raw.get("version") != VERSION:
        raise ValueError("invalid ECGFounder output version")
    if raw.get("status") != "success" or raw.get("source") != "live":
        raise ValueError("ECGFounder evidence must be a successful live result")
    keys = {score_key(index) for index in range(TASK_COUNT)}
    for field in ("findings", "units", "methods"):
        if not isinstance(raw.get(field), dict) or set(raw[field]) != keys:
            raise ValueError(f"ECGFounder {field} must contain all 150 label keys")
    for key, value in raw["findings"].items():
        if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError(f"invalid ECGFounder score: {key}")
    if any(unit != "score" for unit in raw["units"].values()):
        raise ValueError("ECGFounder scores must use score units")
    if any(method != SCORE_METHOD for method in raw["methods"].values()):
        raise ValueError("ECGFounder score method mismatch")
    if raw.get("missingness") != {}:
        raise ValueError("ECGFounder successful output cannot have missing scores")
    details = raw.get("details")
    if not isinstance(details, dict) or any(details.get(key) != value for key, value in provenance().items()):
        raise ValueError("ECGFounder model provenance mismatch")
    return raw


def preprocess(record: Record):
    """Revalidate the attached MAT and return a canonical (12, 5000) float32 array."""
    import numpy as np
    from records import load_record

    current, waveform = load_record(record.path, lead_order=record.lead_names)
    if current != record:
        raise ValueError("ECG record changed after attachment")
    if record.sampling_rate_hz != 500:
        raise ValueError("ECGFounder requires a verified 500 Hz recording")
    if waveform.shape != (12, 5000):
        raise ValueError("ECGFounder requires exactly 12 leads and 5000 samples (10 seconds)")
    if record.waveform_units != ("mV",) * 12:
        raise ValueError("ECGFounder requires waveform units explicitly verified as mV")
    order = [record.lead_names.index(lead) for lead in STANDARD_LEADS]
    waveform = waveform[order]
    mean = float(np.mean(waveform))
    deviation = float(np.std(waveform))
    if not math.isfinite(mean) or not math.isfinite(deviation) or deviation <= 0:
        raise ValueError("ECGFounder cannot classify a flat or invalid waveform")
    normalized = (waveform - mean) / (deviation + 1e-8)
    normalized = np.asarray(normalized, dtype=np.float32, order="C")
    if not np.isfinite(normalized).all():
        raise ValueError("ECGFounder preprocessing produced nonfinite samples")
    return normalized


def _checkpoint_bytes(path: str | Path) -> bytes:
    with Path(path).expanduser().open("rb") as source:
        payload = source.read(CHECKPOINT_BYTES + 1)
    if len(payload) != CHECKPOINT_BYTES or hashlib.sha256(payload).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("ECGFounder checkpoint size or SHA256 mismatch; file was not changed")
    return payload


def _load_model(payload: bytes):
    import numpy as np
    from numpy.core.multiarray import scalar
    import torch
    from ecgfounder import Net1D

    # The pinned checkpoint contains one NumPy float64 val_auroc metadata value.
    # Permit only its scalar/dtype constructors while retaining weights-only load.
    with torch.serialization.safe_globals([scalar, np.dtype, type(np.dtype("float64"))]):
        checkpoint = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("ECGFounder checkpoint must contain state_dict")
    model = Net1D(in_channels=12, base_filters=64, ratio=1,
                  filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                  m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16,
                  stride=2, groups_width=16, verbose=False,
                  use_bn=False, use_do=False, n_classes=TASK_COUNT)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to("cpu")
    model.eval()
    model.requires_grad_(False)
    return model


def _scores(model, batch) -> list[float]:
    import torch

    with torch.inference_mode():
        logits = model(batch)
        if tuple(logits.shape) != (1, TASK_COUNT) or not bool(torch.isfinite(logits).all()):
            raise ValueError("ECGFounder must return 150 finite logits")
        scores = torch.sigmoid(logits)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("ECGFounder returned nonfinite scores")
    return [float(value) for value in scores[0].tolist()]


class Classifier:
    def __init__(self, checkpoint: str | Path):
        self.ready = False
        load_labels()
        payload = _checkpoint_bytes(checkpoint)
        self.model = _load_model(payload)
        del payload
        import torch

        _scores(self.model, torch.zeros((1, 12, 5000), dtype=torch.float32, device="cpu"))
        self.ready = True

    def provenance(self) -> dict:
        return provenance()

    def classify(self, record: Record) -> dict:
        if not self.ready:
            raise ValueError("ECGFounder classifier is not ready")
        started = time.monotonic()
        waveform = preprocess(record)
        import torch

        values = _scores(self.model, torch.from_numpy(waveform).unsqueeze(0))
        findings = {score_key(index): value for index, value in enumerate(values)}
        result = {
            "record_id": record.record_id,
            "record_hash": record.sha256,
            "version": VERSION,
            "status": "success",
            "source": "live",
            "findings": findings,
            "units": {key: "score" for key in findings},
            "methods": {key: SCORE_METHOD for key in findings},
            "missingness": {},
            "details": {**provenance(), "input_shape": [12, 5000],
                        "sampling_rate_hz": 500, "lead_order": list(STANDARD_LEADS),
                        "waveform_units": "mV", "filtering": "none",
                        "normalization": "global z-score; epsilon=1e-8"},
            "elapsed": time.monotonic() - started,
        }
        return validate_output(result)


def _file_matches(path: Path) -> bool:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            count += len(chunk)
            if count > CHECKPOINT_BYTES:
                return False
            digest.update(chunk)
    return count == CHECKPOINT_BYTES and digest.hexdigest() == CHECKPOINT_SHA256


def download(directory: str | Path) -> Path:
    """Download only the pinned public checkpoint; never overwrite any existing file."""
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / CHECKPOINT_NAME
    if target.exists():
        if _file_matches(target):
            return target
        raise ValueError(f"Existing checkpoint failed verification and was preserved: {target}")
    temporary = None
    try:
        request = urllib.request.Request(CHECKPOINT_URL, headers={"Accept-Encoding": "identity"})
        digest = hashlib.sha256()
        count = 0
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != CHECKPOINT_BYTES:
                raise ValueError("ECGFounder download length does not match pinned checkpoint")
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".ecgfounder-", delete=False) as output:
                temporary = Path(output.name)
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > CHECKPOINT_BYTES:
                        raise ValueError("ECGFounder download exceeds pinned checkpoint size")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        if count != CHECKPOINT_BYTES or digest.hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("ECGFounder download SHA256 or size mismatch")
        try:
            os.link(temporary, target)
        except FileExistsError:
            if not _file_matches(target):
                raise ValueError(f"Checkpoint appeared during download and was preserved: {target}") from None
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def check(checkpoint: str | Path, ecg: str | Path, timeout: float = 30) -> dict:
    """Measure readiness, repeated CPU inference, and worker memory without an API key."""
    import platform
    import resource
    from records import load_record
    from traces import dependency_versions
    from worker import ToolWorker

    record, _ = load_record(ecg)
    started = time.monotonic()
    with ToolWorker(checkpoint=checkpoint) as worker:
        if not worker.prepare_classifier(record, timeout):
            raise ValueError(worker.classifier_error)
        readiness = time.monotonic() - started
        pid = worker.process.pid
        first = worker.call(TOOL_NAME, {"record_id": record.record_id}, record, timeout)
        measured = worker.call("call_measurement_tool", {"record_id": record.record_id}, record, timeout)
        second = worker.call(TOOL_NAME, {"record_id": record.record_id}, record, timeout)
        validate_output(first)
        validate_output(second)
        if measured.get("status") not in ("success", "partial"):
            raise ValueError("measurement tool failed during shared-worker check")
        if first["findings"] != second["findings"] or worker.process.pid != pid:
            raise ValueError("classifier repeat or worker reuse check failed")
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_bytes = peak if platform.system() == "Darwin" else peak * 1024
    return {"schema_version": 1, "kind": "public_record_cpu_inference",
            "record_id": record.record_id, "record_sha256": record.sha256,
            "provenance": provenance(), "versions": dependency_versions(),
            "environment": {"platform": platform.system(), "architecture": platform.machine()},
            "readiness_seconds": readiness, "first_inference_seconds": first["elapsed"],
            "repeat_inference_seconds": second["elapsed"], "child_peak_rss_mib": peak_bytes / (1024 ** 2),
            "complete_score_count": len(first["findings"]), "deterministic_repeat": True,
            "same_worker_for_both_tools": True,
            "focus_scores": {label_for_key(score_key(index)): first["findings"][score_key(index)] for index in (1, 2, 3, 5, 36)},
            "measurements": measured["findings"],
            "limitations": ["Uncalibrated scores; no diagnostic threshold or clinical validation.",
                            "Memory is the maximum RSS among this process's terminated children.",
                            "No language-model API call."]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("download", help="download and verify the pinned public 12-lead checkpoint")
    command.add_argument("--directory", default="checkpoints/ecgfounder")
    command = commands.add_parser("check", help="verify CPU inference and worker reuse on a supplied record")
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--ecg", type=Path, required=True)
    command.add_argument("--tool-timeout", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            import json
            print(json.dumps(check(args.checkpoint, args.ecg, args.tool_timeout), indent=2, allow_nan=False))
            return 0
        path = download(args.directory)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{path}\nSHA256 {CHECKPOINT_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
