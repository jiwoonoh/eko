"""Append-only local events with unique run IDs and explicit provenance."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import os
import platform
from pathlib import Path
import threading
import uuid

from contracts import Record, UPSTREAM_REVISION


def dependency_versions() -> dict[str, str | None]:
    versions = {"python": platform.python_version()}
    for name in ("eko", "numpy", "scipy", "neurokit2", "pandas", "matplotlib", "scikit-learn", "requests", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _json_value(value):
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ("contracts.py", "eko.py", "host.py", "records.py", "measurements.py",
             "worker.py", "traces.py", "reporting.py", "classifier.py", "ecgfounder.py",
             "ecgfounder_labels.py", "providers/openrouter.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names if (root / name).is_file()}


class EventLog:
    def __init__(self, directory: str | Path, record: Record, config: dict):
        folder = Path(directory).expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.run_id = uuid.uuid4().hex
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.path = folder / f"{stamp}-{self.run_id}.jsonl"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._file = os.fdopen(fd, "w", encoding="utf-8")
        self._lock = threading.Lock()
        self._sequence = 0
        self("run_started", record=record.public(), config=config,
             source_revision=UPSTREAM_REVISION, versions=dependency_versions(),
             code_sha256=code_hashes())

    def __call__(self, event_type: str, **fields) -> None:
        with self._lock:
            if self._file.closed:
                raise RuntimeError("event log is closed")
            if set(fields) & {"schema_version", "run_id", "sequence", "timestamp", "type"}:
                raise ValueError("event fields cannot replace provenance")
            event = {
                "schema_version": 1,
                "run_id": self.run_id,
                "sequence": self._sequence + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": event_type,
                **fields,
            }
            line = json.dumps(event, default=_json_value, allow_nan=False)
            self._file.write(line + "\n")
            self._file.flush()
            os.fsync(self._file.fileno())
            self._sequence += 1

    def close(self) -> None:
        with self._lock:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
