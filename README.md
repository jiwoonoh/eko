# Eko: ECG research runtime

A small OpenRouter agent that requests predefined ECG measurements, observes the
actual results, and continues until it answers. The original coding agent remains
in Git history. The `ecg-core` branch replaces generated Python, agent sockets,
child agents, Linux sandboxing, and the terminal UI with a basic CLI and one
persistent scientific-tool worker.

The core has no third-party dependencies. The optional ECG extra installs the
numerical measurement backend. The optional ECGFounder backend now supports verified CPU inference.
The agent can now request measurements and classifier scores through the same worker.
The development benchmark and full function reference follow in the final change.

## Rebuild progress

| Change | State | Evidence |
|---|---|---|
| Coding-agent teardown | Complete | [`ac4aad8`](https://github.com/jiwoonoh/eko/commit/ac4aad8) |
| Live measurements and public demo | Complete | [`37a7557`](https://github.com/jiwoonoh/eko/commit/37a7557) |
| Artifact adapters and shared evaluation | Complete | [`344fc8b`](https://github.com/jiwoonoh/eko/commit/344fc8b) |
| OpenRouter parameter-routing fix | Complete | [`8d4e838`](https://github.com/jiwoonoh/eko/commit/8d4e838) |
| Bounded evidence reports | Complete | [`973537b`](https://github.com/jiwoonoh/eko/commit/973537b) |
| ECGFounder CPU backend | Verified | [Classifier tests](tests/test_classifier.py), [public-record check](evidence/classifier-smoke.json) |
| Agent classifier integration | Verified with mocked provider | [Runtime integration tests](tests/test_classification_runtime.py), [reporting tests](tests/test_reporting.py) |
| Development benchmark and function reference | In progress | Delivered in the final change |

The classifier is ECGFounder's public 150-label model, distinct from ECG-Agent's
unavailable 71-label checkpoint. The complete trained head is loaded with strict
parameter matching. Input is verified 12-lead, 500 Hz, 5000-sample mV data,
normalized globally as in the pinned official PTB-XL evaluator. No bandpass filter
or diagnostic threshold is added. Weights are downloaded explicitly, verified by
SHA-256, and kept outside Git.

```bash
./.venv/bin/python -m pip install -r requirements-classifier.lock
./.venv/bin/python -m pip install --no-deps -e '.[ecg,classifier]'
./.venv/bin/python -m classifier download --directory checkpoints/ecgfounder
./.venv/bin/python -m classifier check \
  --checkpoint checkpoints/ecgfounder/12_lead_ECGFounder.pth \
  --ecg data/demo/HR00705.mat
```

The check uses the actual waveform and CPU model without a language-model API
request. It tests repeatability and reuse of one worker for both scientific tools.
It does not establish clinical validity.

To enable classification in the dialogue CLI:

```bash
./.venv/bin/eko --model YOUR_OPENROUTER_TOOL_CAPABLE_MODEL_ID \
  --ecg data/demo/HR00705.mat --classifier ecgfounder \
  --checkpoint checkpoints/ecgfounder/12_lead_ECGFounder.pth
```

Classifier readiness runs before tool registration. A failed check leaves the
measurement tool available. Model-selected checkpoint paths are rejected.
`TurnResult.report` records the exact finding references released to the user,
separately from all accumulated evidence. Classification scores remain independent
model outputs; no diagnostic thresholds or free-form clinical conclusions are
released. New live OpenRouter classifier checks are pending because no API key is
available to the implementation session.

## Install

The core supports Python 3.10 or newer. Use Python 3.11 for the locked ECG
environment, tested on macOS arm64.

```bash
cd ~/code/eko
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements-ecg.lock
./.venv/bin/python -m pip install -e '.[ecg]'
```

For a core-only development installation, use `pip install -e .`. Numerical tests
skip when the ECG extra is absent. NeuroKit2 0.2.7 imports Requests without
listing it in its dependency metadata, so the ECG extra declares it explicitly.

## Select an ECG and run

Use the same ECG as the first published ECG-Agent transcript example:

```bash
./.venv/bin/python -m demo
./.venv/bin/python host.py \
  --model YOUR_OPENROUTER_TOOL_CAPABLE_MODEL_ID \
  --ecg data/demo/HR00705.mat
```

The helper fetches the public PTB-XL 1.0.3 waveform for **HR00705**, verifies pinned
source hashes and digital checksums, and converts physical mV samples to MAT.
`data/demo/source.json` records attribution, the exact published measurement
question, and their stored tool output. Reruns verify the existing bundle without
overwriting it. The waveform is the same selected record; the initial CLI supports
its measurement question, with classification unavailable.

For this record, live measurements are HR **75.95 bpm**, RR **790 ms**, PR
**116 ms**, Q–S peak span **93 ms**, and corrected QTc **432.53 ms**. The published
QTc is **422.28 ms**; preceding-RR beat alignment deliberately changes that
estimator. The other shared measurement values match the published output.

The official offline dialogue script reads precomputed tool CSVs and does not need
waveform files. Live numerical execution needs the underlying waveform, which the
demo obtains publicly rather than requiring your own data.

Start with a known, full 12-lead PTB-XL-style MAT file: numeric channels-first
`feats`, scalar `curr_sample_rate`, and a verified lead order. File identifiers
such as `HR00025.mat` and `HR25.mat` resolve to the same dataset-qualified record
ID. Files are not renamed.

When MAT lead names are absent, `--lead-order standard` explicitly confirms that
the rows are I, II, III, aVR, aVL, aVF, V1–V6. Do this only for records with that
known layout. A different known order can be supplied as twelve comma-separated
names. Unknown layouts, invalid sampling rates, and nonfinite signals are rejected.
Waveform units stay unknown unless supplied by record metadata.

```bash
./.venv/bin/python host.py \
  --model YOUR_OPENROUTER_TOOL_CAPABLE_MODEL_ID \
  --ecg /path/to/HR00025.mat \
  --lead-order standard
```

In an interactive terminal, a missing API key produces a hidden key prompt. An
existing `OPENROUTER_API_KEY` is also accepted. The key is kept out of worker
processes and logs.

For one question and a machine-readable result:

```bash
./.venv/bin/python host.py \
  --model YOUR_OPENROUTER_TOOL_CAPABLE_MODEL_ID \
  --ecg /path/to/HR00025.mat --lead-order standard --json \
  'Measure the heart rate and intervals, and explain what is unavailable.'
```

Choose a specific model for reproducible experiments. Requested and returned
model IDs, generation usage, input hash, tool versions, and runtime limits are
recorded in the local run log. A provider response that requests a tool is not
considered a final answer.

Tool execution is sequential in the local worker. The client omits the optional
`parallel_tool_calls` parameter, which otherwise excludes free endpoints under
strict parameter routing. Restart the CLI after updating client code.

## Measurements and evidence

`call_measurement_tool` accepts only the selected `record_id`; the model cannot
choose filesystem paths or run Python. Each actual result has an evidence ID,
record hash, execution status, findings, units, method, missingness, and timing.

| Finding | Unit | Method |
|---|---|---|
| `heart_rate_bpm` | bpm | Heart rate from median RR |
| `rr_interval_ms` | ms | Consecutive R peaks |
| `pr_interval_ms` | ms | P onset to R onset |
| `q_s_peak_span_ms` | ms | Q peak to S peak; distinct from onset-to-offset QRS |
| `qtc_bazett_ms` | ms | Q peak to T offset, corrected using preceding RR |

The source numerical estimators and filter thresholds are recorded as an initial
reference. Raw beat estimates and filter exclusions remain available alongside
filtered summaries. Missingness is preserved, including partial delineation.
The backend supplies measurements rather than the source's hardcoded diagnostic
interpretations. Matching source outputs establishes reproducibility; clinical
validity requires independent annotated measurements.

The controller renders answers from a small JSON report contract. The model
selects evidence IDs, finding names, and fixed explanation or unavailable codes;
it cannot supply displayed values, diagnostic labels, or free-form medical prose.
Measurement definitions and follow-ups remain available. Rhythm classification,
normality judgments, and treatment recommendations are explicitly unavailable.
Median RR does not establish regularity, and estimator filter bounds are not
clinical reference ranges.

Invalid final reports return `failed`, with a safe evidence summary and an
`answer_rejected` log event. Original model text stays in `model_response` events
for research review; it does not enter the displayed answer or subsequent model
history. This boundary also strips prose accompanying native tool calls. A
`success` turn means a permitted report completed, not clinical validation.

JSONL events are appended under `runs/` with unique run and call identities. Logs,
ECG files, datasets, API-key files, and checkpoints are ignored by Git. The worker
stays loaded between calls and is restarted after a timeout. Defaults are eight
model responses and four tool executions per question, thirty seconds per tool,
sixty seconds per model request, and one bounded transient retry. Ctrl+C cancels
an active turn; Ctrl+D exits the interactive CLI.

## Evaluation

The evaluator separates recorded-output adapter regression, live measurements,
and agent reporting. Its manifest runner calls the same `run_turn` as the CLI.
Numeric claim checks operate on explicit structured claims; they do not constitute
an unrestricted prose or clinical correctness judge.

```bash
./.venv/bin/python -m evaluation csv /path/to/ecg_analysis_measurements.csv --kind measurement
./.venv/bin/python -m evaluation transcripts /path/to/results.jsonl
./.venv/bin/python -m evaluation report /path/to/turn_results.jsonl
```

CSV blanks mean unavailable, with an unknown source reason. CSV replay is labeled
as replayed evidence. Both published and newer inference transcript formats have
explicit adapters. Source-file/line identity preserves repeated generations and
repeated questions. New results are aligned by user turn and tool step rather
than assistant-message position. Cases without reference actions have no invented
next-action prediction labels. Ground-truth-assisted and autonomous histories are
kept separate. Provider failures count against task completion and are excluded
from fidelity denominators.

## Tests

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

Tests use mocked model responses and controlled signals, without an API key or
classifier checkpoint. The public published sample also has a real-worker
integration check. A live OpenRouter call requires your key in your terminal.

## Source and next milestones

Numerical reference: [ECG-Agent](https://github.com/gustmd0121/ECG-Agent/tree/0ab07899ec0a39b087bebd560b280cad484eb0d0),
pinned to `0ab07899ec0a39b087bebd560b280cad484eb0d0`. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for its MIT notice and [PLAN.md](PLAN.md)
for the implementation scope.

Exact ECG-Agent reproduction still requires its 71-label checkpoint and compatible
fairseq revision. The optional ECGFounder backend uses a separate, pinned 150-label
checkpoint and does not require fairseq. Later
work adds a local dialogue-model adapter, structured evidence-linked claims,
clinical validation, and training experiments. These capabilities are deferred;
the first prototype uses real measurements and OpenRouter named calls.
