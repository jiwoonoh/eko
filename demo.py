"""Fetch and verify one public PTB-XL 1.0.3 ECG for a local demonstration.

This converter supports only the pinned 00705_hr format-16 record. It is not
a general WFDB reader. Data remains under the ignored output directory.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import urllib.request

from contracts import STANDARD_LEADS, UPSTREAM_REVISION

DATASET_VERSION = "1.0.3"
DATASET_URL = "https://physionet.org/content/ptb-xl/1.0.3/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode"
BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/records500/00000/"
HEADER_NAME = "00705_hr.hea"
DATA_NAME = "00705_hr.dat"
MAT_NAME = "HR00705.mat"
HEADER_URL = BASE_URL + HEADER_NAME
DATA_URL = BASE_URL + DATA_NAME
HEADER_SHA256 = "f8a6b942147e32a34f9c11998f213103f9175bc32017f53a22f7b53ba035c464"
DATA_SHA256 = "481ae4a893fcc113ebacdad1cf80ced881586d0714af3e9ac5e301e880ebc39a"
INITIAL = (-55, -30, 25, 42, -40, -2, 10, 5, -80, -235, -215, -270)
CHECKSUMS = (46899, 7231, 25876, 5250, 43079, 16170, 15250, 53133, 64830, 56888, 58309, 47755)
PUBLISHED_URL = f"https://raw.githubusercontent.com/gustmd0121/ECG-Agent/{UPSTREAM_REVISION}/src/experiments/overall_results/td_ecg_llama_final/results_rank0_without_gt.jsonl"
PUBLISHED_LINE_SHA256 = "03bbe7ca98bbd0ca308ee7c5256d0518422afb94bc2e4ac97f2ff03988c1bc51"
PUBLISHED_QUESTION_SHA256 = "e9023abf32a8ce36ef23c55761e96ae62438617f5f18d7ccfb62fcedc3274804"
STORED_MEASUREMENTS = {"heart_rate": 75.95, "pr_interval": 116, "qrs_duration": 93, "qtc_interval": 422.28}
PUBLISHED_LINE_LIMIT = 64 * 1024
SOURCE_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
HEADER_LIMIT = 4096
DATA_LENGTH = 12 * 5000 * 2
MAT_LIMIT = 2 * 1024 * 1024
MANIFEST_LIMIT = 16 * 1024


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str, limit: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "eko-ecg-demo/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > limit:
            raise ValueError("demo download exceeds its byte limit")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("demo download exceeds its byte limit")
    return data


def _published_case() -> dict:
    request = urllib.request.Request(PUBLISHED_URL, headers={"User-Agent": "eko-ecg-demo/1.0",
                                                            "Range": f"bytes=0-{PUBLISHED_LINE_LIMIT - 1}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        line = response.readline(PUBLISHED_LINE_LIMIT + 1)
    if len(line) > PUBLISHED_LINE_LIMIT or not line.endswith(b"\n"):
        raise ValueError("published example first line exceeds its byte limit")
    if _digest(line) != PUBLISHED_LINE_SHA256:
        raise ValueError("published example first-line SHA256 differs from the pinned artifact")
    example = json.loads(line)
    if example.get("ecg_id") != "00705" or example.get("batch_idx") != 283:
        raise ValueError("published example does not identify ECG 00705, batch 283")
    dialogue = example.get("dialogue", [])
    if len(dialogue) < 5 or dialogue[3].get("role") != "user" or dialogue[4].get("action") != "call_measurement_tool":
        raise ValueError("published example does not contain the expected measurement turn")
    question = dialogue[3].get("content", "")
    if not isinstance(question, str) or _digest(question.encode()) != PUBLISHED_QUESTION_SHA256:
        raise ValueError("published measurement question differs from the pinned artifact")
    stored = dialogue[4].get("tool_output")
    if stored != STORED_MEASUREMENTS:
        raise ValueError("published stored measurement values differ from the pinned artifact")
    return {"source_url": PUBLISHED_URL, "source_revision": UPSTREAM_REVISION,
            "first_line_sha256": _digest(line), "line_number": 1, "ecg_id": "00705", "batch_idx": 283,
            "measurement_user_turn": 2, "dialogue_index": 3, "question": question,
            "question_sha256": _digest(question.encode()), "stored_tool_output": stored,
            "tool_action": "call_measurement_tool",
            "context": "Use the measurement question alone; no prior classification analysis is supplied"}


def convert_record(header: bytes, payload: bytes):
    """Validate this known header and its digital checksums, then return mV."""
    import numpy as np

    if len(header) > HEADER_LIMIT:
        raise ValueError("demo header exceeds its byte limit")
    lines = header.decode("ascii").splitlines()
    if len(lines) != 13 or lines[0].split() != ["00705_hr", "12", "500", "5000"]:
        raise ValueError("demo header must declare 00705_hr, 12 leads, 500 Hz, 5000 samples")
    for index, line in enumerate(lines[1:]):
        expected = [DATA_NAME, "16", "1000.0(0)/mV", "16", "0", str(INITIAL[index]),
                    str(CHECKSUMS[index]), "0", SOURCE_LEADS[index]]
        if line.split() != expected:
            raise ValueError(f"demo header metadata differs for lead {SOURCE_LEADS[index]}")
    if len(payload) != DATA_LENGTH:
        raise ValueError("demo data must contain exactly 120000 format-16 bytes")
    digital = np.frombuffer(payload, dtype="<i2").reshape(5000, 12).T
    if tuple(int(value) for value in digital[:, 0]) != INITIAL:
        raise ValueError("demo digital initial samples do not match the header")
    checksums = tuple(int(value) for value in np.sum(digital, axis=1, dtype=np.int64) % 65536)
    if checksums != CHECKSUMS:
        raise ValueError("demo per-lead digital checksums do not match the header")
    # The pinned header gives baseline 0 and ADC gain 1000 digital units/mV.
    return digital.astype(np.float64) / 1000.0


def _read_bounded(path: Path, limit: int) -> bytes:
    with path.open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"demo file exceeds its byte limit: {path.name}")
    return data


def _verify_source(header: bytes, payload: bytes):
    if _digest(header) != HEADER_SHA256 or _digest(payload) != DATA_SHA256:
        raise ValueError("demo source SHA256 differs from the pinned public record")


def _verify_existing(directory: Path) -> Path:
    import numpy as np
    from records import load_record

    manifest = json.loads(_read_bounded(directory / "source.json", MANIFEST_LIMIT))
    if manifest.get("dataset_version") != DATASET_VERSION or manifest.get("dataset_url") != DATASET_URL:
        raise ValueError("existing demo manifest does not identify PTB-XL 1.0.3")
    expected = {"header": {"url": HEADER_URL, "sha256": HEADER_SHA256},
                "digital_data": {"url": DATA_URL, "sha256": DATA_SHA256}}
    if manifest.get("sources") != expected or manifest.get("license_url") != LICENSE_URL:
        raise ValueError("existing demo provenance differs from the pinned source")
    published = manifest.get("published_example", {})
    if (published.get("source_url") != PUBLISHED_URL or published.get("source_revision") != UPSTREAM_REVISION
            or published.get("first_line_sha256") != PUBLISHED_LINE_SHA256 or published.get("ecg_id") != "00705"
            or published.get("batch_idx") != 283 or published.get("stored_tool_output") != STORED_MEASUREMENTS
            or _digest(str(published.get("question", "")).encode()) != PUBLISHED_QUESTION_SHA256):
        raise ValueError("existing published example provenance differs from the pinned artifact")
    header = _read_bounded(directory / HEADER_NAME, HEADER_LIMIT)
    payload = _read_bounded(directory / DATA_NAME, DATA_LENGTH)
    _verify_source(header, payload)
    signal = convert_record(header, payload)
    path = directory / MAT_NAME
    mat_bytes = _read_bounded(path, MAT_LIMIT)
    if _digest(mat_bytes) != manifest.get("mat_sha256"):
        raise ValueError("existing demo MAT hash differs from its provenance")
    record, existing_signal = load_record(path)
    if record.sampling_rate_hz != 500 or record.lead_names != STANDARD_LEADS or record.waveform_units != ("mV",) * 12:
        raise ValueError("existing demo MAT metadata differs from the pinned record")
    if not np.array_equal(existing_signal, signal):
        raise ValueError("existing demo MAT samples differ from the pinned record")
    return path


def prepare_demo(output_dir="data/demo") -> Path:
    """Create an exclusive local bundle, or return an already verified bundle."""
    import numpy as np
    from scipy.io import savemat

    directory = Path(output_dir).expanduser().resolve()
    names = (HEADER_NAME, DATA_NAME, MAT_NAME, "source.json")
    existing = [(directory / name).exists() or (directory / name).is_symlink() for name in names]
    if any(existing):
        if not all(existing):
            raise FileExistsError("demo directory contains an incomplete bundle; choose an empty output directory")
        return _verify_existing(directory)

    published = _published_case()
    header = _download(HEADER_URL, HEADER_LIMIT)
    payload = _download(DATA_URL, DATA_LENGTH)
    _verify_source(header, payload)
    signal = convert_record(header, payload)
    buffer = BytesIO()
    savemat(buffer, {"feats": signal, "curr_sample_rate": 500,
                     "lead_names": np.array(STANDARD_LEADS, dtype=object),
                     "waveform_units": np.array(["mV"] * 12, dtype=object)})
    mat_bytes = buffer.getvalue()
    manifest = {
        "converter": "eko-pinned-ptbxl-demo-v2",
        "dataset": "PTB-XL, a large publicly available electrocardiography dataset",
        "dataset_version": DATASET_VERSION,
        "dataset_url": DATASET_URL,
        "doi": "10.13026/kfzx-aw45",
        "authors": ["Patrick Wagner", "Nils Strodthoff", "Ralf-Dieter Bousseljot", "Wojciech Samek", "Tobias Schaeffter"],
        "license": "Creative Commons Attribution 4.0 International",
        "license_url": LICENSE_URL,
        "source_record": "00705_hr",
        "record_id": "ptbxl:705",
        "published_example": published,
        "sources": {"header": {"url": HEADER_URL, "sha256": _digest(header)},
                    "digital_data": {"url": DATA_URL, "sha256": _digest(payload)}},
        "mat_sha256": _digest(mat_bytes),
        "conversion": "format-16 little-endian signed int16; interleaved 12 leads; physical mV=(digital-0)/1000",
        "sampling_rate_hz": 500,
        "lead_names": list(STANDARD_LEADS),
        "waveform_units": ["mV"] * 12,
        "changes": "Converted the original WFDB header/data pair to a MAT file; no waveform filtering or resampling",
    }
    bundle = {HEADER_NAME: header, DATA_NAME: payload, MAT_NAME: mat_bytes,
              "source.json": (json.dumps(manifest, indent=2) + "\n").encode()}
    directory.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for name, data in bundle.items():
            path = directory / name
            with path.open("xb") as target:
                created.append(path)
                target.write(data)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return directory / MAT_NAME


def main():
    parser = argparse.ArgumentParser(description="Download and verify one public PTB-XL 1.0.3 demo ECG")
    parser.add_argument("--output-dir", default="data/demo")
    arguments = parser.parse_args()
    try:
        path = prepare_demo(arguments.output_dir)
    except ImportError:
        parser.exit(1, "Install the measurement extra before preparing the demo.\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Demo preparation failed: {error}\n")
    print(path)


if __name__ == "__main__":
    main()
