"""Small data contracts shared by the runner, tools, and evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

UPSTREAM_REVISION = "0ab07899ec0a39b087bebd560b280cad484eb0d0"
STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def canonical_record_id(value: str | int, dataset: str = "ptbxl") -> str:
    """Resolve PTB-XL aliases without renaming files or guessing other datasets."""
    text = str(value)
    if ":" in text:
        prefix, text = text.split(":", 1)
        if prefix != dataset:
            raise ValueError("record belongs to a different dataset")
    name = Path(text).name
    match = re.fullmatch(r"(?:HR)?(\d+)(?:\.mat)?", name, re.IGNORECASE)
    if match is None or int(match[1]) < 1:
        raise ValueError(f"invalid {dataset} record identifier: {name}")
    return f"{dataset}:{int(match[1])}"


@dataclass(frozen=True)
class Record:
    path: str
    record_id: str
    sha256: str
    sampling_rate_hz: float
    lead_names: tuple[str, ...]
    waveform_units: tuple[str, ...]

    def public(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if key != "path"}


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Decision:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolObservation:
    record_id: str
    record_hash: str
    call_id: str
    evidence_id: str
    tool: str
    version: str
    status: str
    findings: dict[str, float | None] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    methods: dict[str, str] = field(default_factory=dict)
    missingness: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0
    source: str = "live"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TurnResult:
    turn_id: str
    status: str
    answer: str = ""
    evidence: tuple[ToolObservation, ...] = ()
    error: str | None = None
    model_responses: int = 0
    tool_calls: int = 0
    elapsed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
