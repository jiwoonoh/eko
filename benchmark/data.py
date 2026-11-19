"""ECG-MTD dialogues and PTB-XL waveforms."""

import json
import os
from pathlib import Path

from toolbox.leads import LEADS, keep_leads, measurement_lead, select_leads

DATASET = "gustmd0121/12-lead-ecg-mtd-dataset"


def read_rows(path):
    return list(iter_rows(path))


def iter_rows(path, *, missing_ok=False):
    path = Path(path)
    if missing_ok and not path.exists():
        return
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def append(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def response_text(text):
    """Remove Qwen reasoning; its opening tag may be prefilled by the template."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "<think>" in text:
        raise ValueError("Response contains an unclosed reasoning block")
    return text.strip()


def reference_responses(dialogue):
    """ECG-Agent response categories by user-turn position."""
    responses = []
    user_index = -1
    query = ""
    for index, turn in enumerate(dialogue):
        if turn["role"] == "user":
            user_index += 1
            query = turn["content"]
            continue
        if turn["role"] != "assistant" or turn.get("action") != "response":
            continue
        previous = dialogue[index - 1] if index else {}
        action = previous.get("action")
        category = None
        if action == "call_classification_tool":
            category = "post_classification"
        elif action == "call_measurement_tool":
            category = "post_measurement"
        elif previous.get("role") == "user" or action == "response_followup":
            category = "direct_response"
        if category and user_index >= 0:
            responses.append({"turn": user_index, "category": category,
                              "query": query, "response": turn["content"]})
    return responses


def load_references(dataset, split):
    """Index judge references by ECG ID and exact user question."""
    from datasets import load_dataset

    references = {}
    for row in load_dataset(dataset, split=split):
        key = str(json.loads(row["metadata"])["ecg_id"])
        if key in references:
            raise ValueError(f"Duplicate reference ECG ID: {key}")
        queries = {}
        for reference in reference_responses(json.loads(row["dialogue"])):
            query = reference["query"]
            if query in queries:
                raise ValueError(f"Duplicate reference question for ECG {key}: {query!r}")
            queries[query] = reference
        references[key] = queries
    return references


def load_cases(dataset=DATASET, split="test", max_samples=None):
    from datasets import load_dataset

    rows = load_dataset(dataset, split=split)
    if max_samples is not None:
        rows = rows.select(range(min(max_samples, len(rows))))
    cases = []
    for row in rows:
        metadata = json.loads(row["metadata"])
        dialogue = json.loads(row["dialogue"])
        cases.append({"case_id": str(metadata["ecg_id"]),
                      "ecg_file": metadata["ecg_lookup_filename_used"],
                      "cefr_level": metadata["cefr_level"],
                      "user_turns": [turn["content"] for turn in dialogue if turn["role"] == "user"],
                      "reference_dialogue": dialogue})
    return cases


def prepare_ecg(ecg_dir, ecg_file, destination, *, leads=None):
    """Stage only the selected waveforms for the agent."""
    leads = select_leads(leads)
    import numpy as np
    from scipy.io import savemat
    from toolbox.record import load_record
    ecg_dir = Path(ecg_dir).resolve()
    # ECG-Agent preprocessing writes HR25.mat; the dataset names HR00025.mat.
    ecg_id = int(Path(ecg_file).stem.removeprefix("HR"))
    source = (ecg_dir / f"HR{ecg_id}.mat").resolve()
    source.relative_to(ecg_dir)
    data, rate = load_record(source)
    signal = np.asarray(data["feats"])
    if rate != 500:
        raise ValueError("ECG-Agent's benchmark tools require 500 Hz ECGs")
    if signal.ndim != 2 or signal.shape[0] != len(LEADS) or not signal.shape[1]:
        raise ValueError("Expected feats with shape (12, samples) in ECG-Agent lead order")
    signal = keep_leads(signal, leads)
    if not np.isfinite(signal).all():
        raise ValueError("Selected ECG leads contain nonfinite values")
    record = {"feats": signal, "curr_sample_rate": int(rate),
              "measurement_lead": measurement_lead(leads)}
    savemat(destination, record)
