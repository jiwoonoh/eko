# Eko: ECG research runtime

This branch turns Eko's coding agent into a bounded ECG tool runner. It measures
waveforms, runs an optional 150-label classifier, and supports follow-up questions
through OpenRouter. Displayed findings come from executed tools and validated
references. The language model cannot write the displayed measurement values or
release an unrestricted diagnosis.

The core has no third-party runtime dependencies. Measurement uses optional
NumPy, SciPy, and NeuroKit2 packages. Classification additionally uses PyTorch.
There is no training pipeline, calibrated diagnostic threshold, or clinical
validation in this release.

## What changed

| Commit | Completed change |
|---|---|
| [`ac4aad8`](https://github.com/jiwoonoh/eko/commit/ac4aad8) | Replaced the coding agent with the bounded ECG runtime. |
| [`37a7557`](https://github.com/jiwoonoh/eko/commit/37a7557) | Added live measurements and the public HR00705 sample. |
| [`344fc8b`](https://github.com/jiwoonoh/eko/commit/344fc8b) | Added artifact adapters and shared evaluation. |
| [`8d4e838`](https://github.com/jiwoonoh/eko/commit/8d4e838) | Fixed OpenRouter parameter routing. |
| [`973537b`](https://github.com/jiwoonoh/eko/commit/973537b) | Enforced evidence-based answer rendering. |
| [`1f96d8f`](https://github.com/jiwoonoh/eko/commit/1f96d8f) | Added and verified the ECGFounder CPU backend. |
| [`8bc21d9`](https://github.com/jiwoonoh/eko/commit/8bc21d9) | Connected classification to named calls and bounded reports. |
| [`895f4fa`](https://github.com/jiwoonoh/eko/commit/895f4fa) | Added the frozen development benchmark, saved execution evidence, and this function reference. |
| This revision | Recorded live OpenRouter execution and reporting failures with the existing guard unchanged. |

Removed: generated Python execution, child agents, model/inbox/world sockets,
Linux sandbox machinery, terminal rendering, Claude and Hugging Face dialogue
backends, feral mode, and automatic context resets. Their dependencies and obsolete
tests were removed with them. These belonged to a general coding agent and are
unnecessary for calling two registered ECG tools. The original source remains in
Git history.

Retained: repeated model decisions, actual tool observations, one conversation,
interruption, bounded execution, and OpenRouter connectivity. Added: record
validation, scientific tools, evidence IDs, append-only event logs, bounded
reporting, and evaluation adapters. [PLAN.md](PLAN.md) records completed and
remaining work.

## Install and run

Python 3.10 or newer is required. The locked scientific environment was checked
with Python 3.11.4 on macOS arm64. Use `python3.11` rather than `/usr/bin/python`.
From a checkout of this branch:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements-classifier.lock
./.venv/bin/python -m pip install --no-deps -e '.[ecg,classifier]'
./.venv/bin/python -m demo
./.venv/bin/python -m classifier download --directory checkpoints/ecgfounder
./.venv/bin/python -m classifier check \
  --checkpoint checkpoints/ecgfounder/12_lead_ECGFounder.pth \
  --ecg data/demo/HR00705.mat
```

`demo` downloads the public PTB-XL 1.0.3 record used in ECG-Agent's first published
transcript, verifies pinned source hashes and digital checksums, and creates a MAT
file with lead and unit metadata. Reruns validate the existing bundle. The original
ECG-Agent offline script reads stored CSV outputs; live tools need the waveform.
You do not need to supply a private record for these checks.

`classifier download` fetches about 370 MB explicitly and verifies its pinned hash.
It preserves an existing invalid file and reports failure. Runtime inference never
downloads weights. `classifier check` runs both tools in one persistent worker and
repeats classification without making a language-model request.

For measurements alone, install `requirements-ecg.lock` and `-e '.[ecg]'`, and omit
classifier options below. A core-only installation is `pip install -e .`; waveform
execution still needs the ECG packages.

```bash
./.venv/bin/eko --model YOUR_OPENROUTER_TOOL_CAPABLE_MODEL_ID \
  --ecg data/demo/HR00705.mat --classifier ecgfounder \
  --checkpoint checkpoints/ecgfounder/12_lead_ECGFounder.pth
```

Use a specific model ID to reproduce an experiment. The interactive CLI prompts
for a missing API key with hidden input; `OPENROUTER_API_KEY` is also accepted.
The key stays in the controller and is excluded from the scientific worker.

For one question, append the question as an argument. Add `--json` to retain all
150 scores, complete observations, and the released report:

```bash
./.venv/bin/eko --model YOUR_OPENROUTER_TOOL_CAPABLE_MODEL_ID \
  --ecg data/demo/HR00705.mat --classifier ecgfounder \
  --checkpoint checkpoints/ecgfounder/12_lead_ECGFounder.pth --json \
  'Report the AFib, sinus rhythm, normal sinus rhythm, and normal ECG scores. Then measure the heart rate and intervals.'
```

Useful follow-ups are “Repeat those measurements using existing evidence,”
“Rerun classification and report the new AFib score,” and “What can these results
not establish?” Numerical results and model scores are supported; an unrestricted
“Is this normal or AFib?” diagnosis is outside the report contract.

A private MAT record needs numeric channels-first `feats`, scalar
`curr_sample_rate`, and verified full 12-lead metadata. If lead names are absent,
`--lead-order standard` confirms I, II, III, aVR, aVL, aVF, V1–V6; use it only when
that layout is known. Twelve comma-separated names can confirm another known
order. Unknown waveform units remain unknown for measurement and prevent
classification. Files are not renamed; supported padded and unpadded identifiers
resolve to the same dataset-qualified record ID.

## Runtime and reporting boundary

One `Session` owns conversation and evidence. `run_turn` asks the provider for a
`Decision`, executes registered calls in a sequential worker, and returns a
`TurnResult`. Native tool arguments contain only `record_id`. Filesystem and
checkpoint paths are controller configuration. Model-generated tool outputs are
never observations.

The default limits are eight model responses and four tool executions per user
turn, thirty seconds per tool, sixty seconds per model request, and one transient
provider retry. A timeout kills the worker; a later call can start another worker.
Ctrl+C interrupts a turn and Ctrl+D exits the interactive CLI. Checkpoint readiness
runs before classification is advertised; failure leaves measurements available.
The OpenRouter client omits `parallel_tool_calls`, whose presence caused the
observed parameter-routing 404 on free endpoints. Endpoint availability still
depends on the selected provider/model.

The earlier classification failure was a reporting failure: without a classifier,
the language model inferred “normal sinus rhythm” and excluded AFib from summary
intervals, then the runner printed that unverified prose. A median RR does not
establish regularity, a PR estimate does not verify P-wave morphology, and filter
bounds are not clinical normal ranges.

The correction requires a JSON report containing evidence references and fixed
explanation codes. The renderer supplies values, units, and trusted classifier
label names. Unknown references, wrong-record evidence, malformed reports, and
free-form conclusions are rejected. A rejected report returns `failed` with a safe
summary and an `answer_rejected` event. Raw model text remains in the local audit
log and is excluded from displayed answers and follow-up history.
`TurnResult.report` contains the validated references released in the answer.
The CLI also prints an evidence appendix: measurements and the five highest
classifier scores; complete scores remain in JSON and logs.

There are four different sources of text and numbers:

| Source | What it supplies |
|---|---|
| Measurement backend | Computed waveform estimates, raw beat values, methods, filtering, and missingness. |
| Classifier | All 150 independent sigmoid scores from the pinned trained model. Scores are uncalibrated and have no diagnostic threshold here. |
| Fixed templates | Tool instructions, measurement definitions, capability explanations, and report wording. ECG values are inserted from evidence. |
| Dataset annotations | Reference labels used only for benchmark selection and scoring, outside model history and classifier inputs. |

A successful turn means that a permitted report completed. It does not establish
clinical correctness. A low AFib score does not rule out AFib; “SINUS RHYTHM,”
“NORMAL SINUS RHYTHM,” and “NORMAL ECG” remain separate labels.

## Scientific methods and provenance

Measurements derive from [ECG-Agent at the reviewed revision](https://github.com/gustmd0121/ECG-Agent/tree/0ab07899ec0a39b087bebd560b280cad484eb0d0).
The backend reads sampling frequency from metadata and measures lead II. It
preserves partial results, metric-specific missingness, raw estimates, and filter
exclusions. It excludes the upstream hardcoded diagnostic interpretations.

| Finding | Unit | Method |
|---|---|---|
| `heart_rate_bpm` | bpm | Heart rate from retained median RR. |
| `rr_interval_ms` | ms | Consecutive R peaks. |
| `pr_interval_ms` | ms | P onset to R onset. |
| `q_s_peak_span_ms` | ms | Q peak to S peak; not onset-to-offset QRS duration. |
| `qtc_bazett_ms` | ms | Q peak to T offset, Bazett corrected with the preceding RR from the same beat identity. |

For HR00705 the computed values are 75.95 bpm, 790 ms RR, 116 ms PR, 93 ms Q–S
peak span, and 432.53 ms QTc. The published QTc is 422.28 ms; the corrected
preceding-RR pairing changes that estimate. Other shared values match the stored
output. Agreement establishes reproducibility; independent annotations are still
needed to validate these estimates.

The classifier is the public **150-label ECGFounder**, a different head from
ECG-Agent's unavailable 71-label weights. The architecture and task order come
from [source revision `04edac7`](https://github.com/PKUDigitalHealth/ECGFounder/tree/04edac702b61c91face519774ddcc0cd712fef23);
the checkpoint comes from [model revision `d9b1793`](https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/d9b1793951b2342f5f7e84f1ac03cd37f8a08724).
Its SHA-256 is
`ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997`.
The ordered label bytes have SHA-256
`7c087a0d383c7ea9ae9b818ad00a551b029334edfa5f1fbb5036bfbaa0df0c39`.

Loading requires complete parameter matching with the original trained head.
Inputs must have twelve verified leads, 500 Hz, exactly 5,000 samples, and mV units.
Rows are reordered to standard lead order, normalized with one global mean and
standard deviation plus epsilon `1e-8`, and converted to float32. This follows the
pinned [150-class PTB-XL evaluator](https://github.com/PKUDigitalHealth/ECGFounder/blob/04edac702b61c91face519774ddcc0cd712fef23/ptbxl_eval.py),
which differs from the repository's separate downstream LVEF preprocessing.
There is no additional filtering or resampling. CPU inference must produce finite
`(1, 150)` outputs. The 150 labels do not provide an exact replacement for every
ECG-Agent label; for example, PTB-XL `INJAS` has no exact counterpart in this task
list.

Observations record record identity/hash, evidence/call IDs, status, units,
methods, tool version, checkpoint/source/label revisions, dependencies, and timing.
JSONL events use unique run files under `runs/`. Data, weights, secrets, and raw
interaction logs are ignored by Git. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for attribution and licenses.

## Capability checks

| Behavior | Implementation | Verification | Reproduce |
|---|---|---|---|
| Repeated native calls and follow-ups | [Runner](eko.py), [provider](providers/openrouter.py) | [Runtime tests](tests/test_runtime.py), [provider tests](tests/test_openrouter.py), [live run](evidence/benchmark-live-cohere.json) | Start the CLI and use the questions above; live reporting limitations are measured below. |
| Record validation and real measurements | [Records](records.py), [measurements](measurements.py) | [Measurement tests](tests/test_measurements.py), [public-record evidence](evidence/classifier-smoke.json) | `python -m demo`, then `python -m classifier check ...` with the options above. |
| Strict classifier loading and preprocessing | [Backend](classifier.py), [architecture](ecgfounder.py) | [Classifier tests](tests/test_classifier.py), [CPU evidence](evidence/classifier-smoke.json) | The classifier check above; no API key needed. |
| Worker reuse and timeout recovery | [Worker](worker.py) | [Worker tests](tests/test_worker.py), [classifier runtime tests](tests/test_classification_runtime.py) | `python -m unittest discover -s tests -p 'test_worker.py' -v`. |
| Bounded, record-bound reports | [Reporting](reporting.py), [contracts](contracts.py) | [Report tests](tests/test_reporting.py), [runtime tests](tests/test_classification_runtime.py) | Run the full test command below; invalid interpretations are explicit fixtures. |
| Unique evidence and versioned logs | [Trace writer](traces.py) | [Log tests](tests/test_cli.py) | CLI runs create distinct files under `runs/`. |
| CSV/transcript compatibility | [Adapters](evaluation.py) | [Adapter tests](tests/test_evaluation.py) | The adapter commands below. |
| Frozen public development cases | [Benchmark](benchmark.py), [manifest](benchmarks/manifest.json) | [Benchmark tests](tests/test_benchmark.py), [direct evidence](evidence/benchmark-direct.json) | The benchmark commands below. |
| Reporting coverage from released references | [Evaluator](evaluation.py), [released claims](benchmark.py) | [Benchmark tests](tests/test_benchmark.py), [dialogue evidence](evidence/benchmark-dialogue.json) | Mocked dialogue with real tools below. |

## What was executed

Saved evidence contains configuration, source hashes, dependency versions, and
sanitized results. Weights, waveforms, and raw conversations are excluded.

| Check | Observed result | Evidence |
|---|---|---|
| HR00705 CPU inference | All 150 finite scores; repeat scores identical; measurements and classifier reused one worker. Readiness 3.37 s, inference 0.046/0.042 s, child peak RSS 999 MiB on macOS arm64. | [CPU run](evidence/classifier-smoke.json) |
| Direct scientific tools | All 20 classifier calls succeeded; measurements returned 17 success and 3 partial results, retaining unavailable metrics. | [Direct run](evidence/benchmark-direct.json) |
| Four-record dialogue | Five turns per record; 20 completed turns, 12 tool executions, 81/81 available requested findings reported, zero rejected reports or execution failures. Model responses were scripted; tools were real. | [Dialogue run](evidence/benchmark-dialogue.json) |
| Automated checks | 182 tests pass with the scientific environment. Core-only execution passes with 31 optional-dependency skips. | [Verification record](evidence/verification.json), [tests](tests/) |
| Live OpenRouter with `cohere/north-mini-code:free` | 9/20 accepted reports; 12 completed tool calls; 11 rejected reports; 49/56 available requested fields reported among accepted turns. Provider-reported inference cost was $0. | [Live run](evidence/benchmark-live-cohere.json), [report audit](evidence/live-report-audit.json) |
| Initial live attempt with `google/gemma-4-31b-it:free` | All 20 turns failed with provider HTTP 429 before any model response or tool execution. Model behavior was not assessed. | [Provider failure](evidence/benchmark-live-gemma.json) |

Timing and memory are measurements from this environment, not resource guarantees.
The memory figure is the maximum RSS of terminated children in the check process.
Dialogue coverage measures the references actually released in `TurnResult.report`;
it does not credit every value in accumulated evidence or the CLI appendix.
Available-value coverage excludes requested null measurements. Clinical accuracy
and unrestricted prose faithfulness are unscored.

The live Cohere run followed the requested tool sequence in all four cases:
both tools, evidence reuse, classifier rerun, measurement reuse, then a limits
question. Scientific execution succeeded, including partial measurement results.
Reporting was unreliable: two replies used unknown finding names and nine failed
the final JSON contract. These eleven replies were replaced by safe summaries
and counted as failures. None of the four cases completed all five turns with
accepted reports covering every requested field. The 87.5% numeric coverage above
applies only to accepted turns and excludes null measurements. It does not mean
87.5% task completion, model accuracy, or clinical accuracy. Zero contradictions
reflects the renderer's enforced evidence boundary. The live audit separately
checks tool scheduling, requested null fields, and evidence reuse.

All seven accepted reports with requested classifier scores selected NORMAL
SINUS RHYTHM (`ecgfounder_001`) in place of SINUS RHYTHM (`ecgfounder_003`). The
renderer labeled those scores correctly, but the requested sinus-rhythm score
was omitted. This accounts for the seven uncovered requested fields. Only two
accepted turns reported every requested field; both were measurement follow-ups.
All four final limits-question replies failed the JSON contract. These results
identify report adherence and class selection as the next implementation work.

## Development benchmark and adapters

[The frozen manifest](benchmarks/manifest.json) selects twenty distinct patients
from PTB-XL 1.0.3 fold 9, five each with AFib, normal ECG plus sinus rhythm,
sinus rhythm without a normal-ECG annotation, and another rhythm annotation.
Selection uses ascending ECG ID and unused patients in that group order. HR00705
and its patient are excluded. Fold 10 remains reserved. Reference annotations
stay in the benchmark sidecar and post-execution scoring.

Group selection uses annotation-key presence, including annotations whose numeric
likelihood is zero/unspecified. Groups are a development challenge sample, not an
exclusive clinical truth table. One AFib-group record also has `NORM`, so there
are six `NORM` positives overall. Exploratory AUROCs are 0.9733 for AFib (5 positive,
15 negative), 0.8452 for normal ECG (6/14), and 0.9800 for sinus rhythm (10/10).
These describe this selected development set; no threshold is fitted and no
clinical performance claim follows.

```bash
./.venv/bin/python -m benchmark prepare
./.venv/bin/python -m benchmark direct \
  --checkpoint checkpoints/ecgfounder/12_lead_ECGFounder.pth \
  --output /tmp/benchmark-direct-new.json
./.venv/bin/python -m benchmark dialogue \
  --checkpoint checkpoints/ecgfounder/12_lead_ECGFounder.pth \
  --output /tmp/benchmark-dialogue-new.json
```

`dialogue` uses a scripted model with real scientific tools unless `--model ID`
is supplied. Adding a model ID uses OpenRouter and requires the environment API
key; it does not prompt. Outputs are created exclusively: choose a fresh filename
for each run. Tool failures, rejected reports, provider failures, completion,
latency, and coverage are reported separately. The four dialogue cases are the
first selected record in each group. They exercise both tools, evidence reuse,
classification reruns, measurement follow-ups, and refusal to infer a diagnosis.

The artifact adapters remain separate from these live-tool checks:

```bash
./.venv/bin/python -m evaluation csv /path/to/ecg_analysis_measurements.csv --kind measurement
./.venv/bin/python -m evaluation transcripts /path/to/results.jsonl
./.venv/bin/python -m evaluation report /path/to/turn_results.jsonl
./.venv/bin/python -m unittest discover -s tests -v
```

Adapters retain repeated generations and questions using source/line identities,
normalize record aliases, reject ambiguous joins, and preserve blank values as
unavailable with unknown source reasons. Published and current transcript formats
have distinct adapters. Execution aligns by sample, user turn, and tool step.
Missing reference actions are not manufactured. CSV replay is labeled separately
from live observations. Ground-truth-assisted history is not implemented in the
manifest runner; autonomous runs are explicitly labeled `without_gt`.

Automated tests use controlled signals, mocked provider responses, and temporary
fixtures. They require neither checkpoint downloads nor an API key. The saved
public-record runs separately establish that the installed CPU backend executed
real waveforms. Mocked-provider tests cannot establish live model behavior.

## Remaining work

The implementation, CPU checks, and development benchmark are complete. Live
OpenRouter requests and native tool execution were verified; reliable final
reporting remains unfinished. The tested free Cohere model produced accepted
reports on nine of twenty turns. Improve report-contract adherence and requested
field coverage, then rerun the frozen benchmark while retaining these baseline
failures. Do not relax evidence validation to raise completion counts. No clinical
validation has been performed. Independent waveform annotations are needed to evaluate
measurement validity. Threshold calibration and an independent evaluation design
must precede diagnostic performance claims. Exact ECG-Agent reproduction remains
blocked by its unavailable 71-label checkpoint and unverified fairseq revision;
this release uses the explicitly different ECGFounder model.

Later work: a local dialogue adapter, structured claim verification beyond numeric
references, structured ECG interpretation with independent references, and SFT/RL.
Training, threshold calibration, treatment guidance, and unrestricted medical prose
remain deferred.

## Function reference

This inventory covers **148 explicitly defined production functions and methods**, including private and nested named helpers. It excludes tests, lambdas, generated dataclass methods, and external-library APIs. `ecgfounder_labels.py` contains verified label bytes and defines no functions.

**Evidence:** “Direct test” targets the callable; “Indirect test” exercises it through a caller. Neither means complete branch coverage. “Actual” links a saved local CPU/public-record run. “Source only” claims no dedicated execution check. Mocked OpenRouter tests do not establish live-provider behavior or clinical accuracy. Source/test links are relative to the repository root.

### `contracts.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`canonical_record_id`](contracts.py#L14) | Normalize supported PTB-XL file names and aliases. | ID/path and dataset -> ptbxl:<integer>. Rejects foreign dataset, malformed name, or ID below one. | [Indirect test](tests/test_evaluation.py#L80) |
| [`Record.public`](contracts.py#L37) | Expose record metadata without its local path. | Record -> dictionary excluding path. Does not validate or read waveform bytes. | [Indirect test](tests/test_runtime.py#L102) |
| [`ToolObservation.to_dict`](contracts.py#L73) | Serialize all observation fields, including evidence identity. | Observation -> recursive dataclass dictionary. Does not itself authenticate the evidence. | [Indirect test](tests/test_classification_runtime.py#L59) |
| [`TurnResult.to_dict`](contracts.py#L89) | Serialize the answer, observations, counters, and released report. | Turn result -> recursive dictionary. Preserves failures and null report; no clinical scoring. | [Direct test](tests/test_evaluation.py#L321) |

### `eko.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`measurement_schema`](eko.py#L16) | Define the measurement tool and fixed record argument. | Record ID -> native function schema. No file-path parameter; additional properties disabled. | [Indirect test](tests/test_runtime.py#L235) |
| [`classification_schema`](eko.py#L36) | Define the optional ECGFounder score tool. | Record ID -> classification function schema. Scores remain separate from diagnosis and calibrated probabilities. | [Indirect test](tests/test_classification_runtime.py#L46) |
| [`_record_key`](eko.py#L51) | Fingerprint public record descriptors for session identity. | Record -> sorted JSON string. Includes hash/layout/units but excludes local path. | [Indirect test](tests/test_runtime.py#L374) |
| [`Session.__init__`](eko.py#L58) | Configure one record's bounded model/tool session. | Model, worker, record, limits -> initialized session. Rejects nonpositive response/tool caps or nonfinite timeouts. | [Indirect test](tests/test_runtime.py#L160) |
| [`Session._reset`](eko.py#L82) | Rebuild schemas and instructions and clear record evidence. | Session state -> fresh system history and empty evidence. Rejects duplicate registered tool names; classifier requires ready worker. | [Indirect test](tests/test_classification_runtime.py#L46) |
| [`Session.set_record`](eko.py#L109) | Select a different record with an explicit history reset. | Record -> changed session or no-op for same identity. Cannot change the record during an active turn. | [Direct test](tests/test_runtime.py#L374) |
| [`Session.close`](eko.py#L118) | Close the provider and always close its worker. | Session -> cleanup side effects. Worker close runs even if provider cleanup raises. | [Indirect test](tests/test_integration.py#L20) |
| [`_finite`](eko.py#L125) | Walk nested values and reject nonfinite floats. | Nested value -> None on success. Raises ValueError for NaN or infinity. | [Indirect test](tests/test_runtime.py#L276) |
| [`_arguments`](eko.py#L136) | Validate a model tool call against registered arguments. | Session and call -> copied argument object. Rejects unknown tool, wrong record, paths, extra/missing fields, type/enum errors. | [Indirect test](tests/test_runtime.py#L235) |
| [`_observation`](eko.py#L166) | Authenticate raw output and assign controller evidence identity. | Call and raw result -> ToolObservation. Rejects wrong record/hash, invalid classifier metadata, nonnumeric findings, unexplained nulls, or malformed timing. | [Indirect test](tests/test_classification_runtime.py#L101) |
| [`_assistant`](eko.py#L220) | Encode native tool calls for provider history. | Decision -> assistant message with JSON arguments. Nonfinite or nonserializable arguments fail serialization. | [Indirect test](tests/test_runtime.py#L124) |
| [`run_turn`](eko.py#L230) | Run repeated model/tool steps and release a validated report. | Session and question -> TurnResult. Reports provider/tool/runner failure, timeout, interrupt, and caps; blocks unsupported final prose. | [Direct test](tests/test_classification_runtime.py#L59) |
| [`run_turn.finish`](eko.py#L240) | Build and log the turn's final result. | Status, answer, error -> TurnResult with released references. Includes failed attempts separately from reusable evidence. | [Indirect test](tests/test_classification_runtime.py#L59) |
| [`run_turn.reject`](eko.py#L247) | Pair a rejected native call with an auditable failure. | Call and reason -> history, failure list, and log updates. Rejected results use source=runner and never become reusable evidence. | [Indirect test](tests/test_runtime.py#L147) |

### `host.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`lead_order`](host.py#L17) | Parse explicit MAT row-order confirmation. | None, standard, or comma list -> tuple or None. Name/count validation occurs in load_record. | [Direct test](tests/test_cli.py#L26) |
| [`show_result`](host.py#L25) | Print a TurnResult as JSON or readable evidence. | Result and JSON option -> stdout/stderr. Text mode shows top five classifier scores; full scores remain in JSON/logs. | [Direct test](tests/test_cli.py#L39) |
| [`main`](host.py#L50) | Load the chosen record and run one-shot or interactive OpenRouter dialogue. | CLI arguments/env/prompted key -> process exit code; linked test checks only --help. Checks options, record, key, and optional classifier; interruption returns 130. | [Indirect test](tests/test_cli.py#L16) |

### `records.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`_strings`](records.py#L15) | Decode MATLAB string, character, and cell-array metadata. | MATLAB value and NumPy -> tuple of stripped strings. Rejects nonstring metadata. | [Indirect test](tests/test_measurements.py#L34) |
| [`_strings.text`](records.py#L21) | Decode one MATLAB byte or text element. | Element -> Python string. Bytes use the default text decoder; malformed bytes raise. | [Indirect test](tests/test_measurements.py#L34) |
| [`_leads`](records.py#L31) | Canonicalize a complete declared twelve-lead layout. | Names -> canonical-cased tuple preserving row order. Rejects unknown, duplicate, missing, or extra leads. | [Indirect test](tests/test_measurements.py#L79) |
| [`load_record`](records.py#L42) | Hash and parse the same bounded MAT bytes. | Path and optional confirmed order -> Record plus float64 12xN array. Rejects >64MiB, invalid samples/rate, ambiguous layout, or conflicting metadata; units may remain unknown. | [Direct test](tests/test_measurements.py#L34) |

### `measurements.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`empty_result`](measurements.py#L27) | Create a complete missing-value measurement observation. | Record, reason, message, status -> raw result. Every field gets explicit missingness; default status is failed. | [Indirect test](tests/test_measurements.py#L97) |
| [`_landmark`](measurements.py#L43) | Read a beat-indexed finite delineation landmark. | Landmark sequence and beat index -> float or None. Absent, null, or nonfinite landmarks return None. | [Indirect test](tests/test_measurements.py#L123) |
| [`_span`](measurements.py#L52) | Measure an inclusive sample span between landmarks. | Wave dictionary, landmark keys, beat, rate -> ms and reason. Missing or reversed endpoints return null with a reason. | [Indirect test](tests/test_measurements.py#L136) |
| [`_estimate`](measurements.py#L62) | Aggregate filtered beat-aligned RR, PR, Q/S, and Bazett QTc estimates. | R peaks, delineations, rate -> findings, missingness, per-beat values, filtering. Keeps preceding-RR pairing and exclusion reasons; computes HR from retained median RR. | [Direct test](tests/test_measurements.py#L123) |
| [`_estimate.add`](measurements.py#L70) | Record a raw per-beat estimate and whether its filter retained it. | Field, beat, value, reason -> append to estimate list. Marks nonfinite or outside-filter values without discarding raw evidence. | [Indirect test](tests/test_measurements.py#L136) |
| [`measure_record`](measurements.py#L126) | Revalidate bytes and measure lead II using NeuroKit. | Record -> numeric observation with methods and missingness. Flat input fails; fewer than seven R peaks is partial; processing exceptions become failed results. | [Direct test](tests/test_measurements.py#L153) |

### `worker.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`ToolWorker.__init__`](worker.py#L33) | Configure persistent subprocess transport and optional local checkpoint. | Optional command/checkpoint -> stopped worker with classifier disabled. An injected command is a host/test option, never an LLM argument. | [Indirect test](tests/test_worker.py#L39) |
| [`ToolWorker._start`](worker.py#L43) | Lazily start an isolated worker process group. | Worker state -> running stdin/stdout/stderr transport. Removes secret-like environment names; process-start errors propagate. | [Indirect test](tests/test_worker.py#L51) |
| [`ToolWorker.close`](worker.py#L56) | Kill and reap the worker group and close its streams. | Worker state -> no active process. Handles absent groups and macOS permission fallback; later calls may restart. | [Direct test](tests/test_worker.py#L39) |
| [`ToolWorker._exchange`](worker.py#L73) | Multiplex one JSONL request, result, and bounded stderr. | Encoded payload and deadline -> decoded response. Hard timeout; rejects EOF, malformed/extra JSON, or oversized stdout; retains bounded stderr tail. | [Indirect test](tests/test_worker.py#L98) |
| [`ToolWorker.prepare_classifier`](worker.py#L120) | Run a child readiness check and authenticate pinned provenance. | Record and timeout -> boolean, provenance/error fields. Missing checkpoint or forged/failed handshake keeps classification disabled. | [Direct test](tests/test_classification_runtime.py#L180) |
| [`ToolWorker.call`](worker.py#L134) | Gate and dispatch the two registered scientific tool names. | Name, record-only arguments, record, timeout -> raw result. Unknown tool/arguments or unready classifier returns failed result. | [Direct test](tests/test_classification_runtime.py#L205) |
| [`ToolWorker._invoke`](worker.py#L143) | Correlate one request/result exchange and attach elapsed time. | Validated name/arguments/record/timeout -> raw result. Rejects request-ID mismatch; kills on timeout/interrupt; transport errors return failure. | [Indirect test](tests/test_classification_runtime.py#L197) |
| [`ToolWorker.__enter__`](worker.py#L172) | Return the worker for a context manager. | Worker -> same instance. No process starts until a request. | [Actual](evidence/classifier-smoke.json) |
| [`ToolWorker.__exit__`](worker.py#L175) | Close the worker when its context exits. | Context exit -> process cleanup. Delegates kill/reap behavior to close. | [Actual](evidence/classifier-smoke.json) |
| [`_failure`](worker.py#L179) | Create a measurement or classifier-specific failed result. | Record, tool, reason, message -> raw failure. Classifier/transport failures use source=runner and no scores. | [Indirect test](tests/test_classification_runtime.py#L205) |
| [`_dispatch`](worker.py#L188) | Validate a child request and invoke measurement or cached classifier. | Request plus persistent cache -> ready/result/failure object. Checks record-only arguments; rejects checkpoint changes; catches dependency/validation errors. | [Direct test](tests/test_classification_runtime.py#L223) |
| [`serve`](worker.py#L223) | Read bounded JSONL requests and keep one classifier cache. | Worker stdin -> one correlated stdout line per request. Scientific stdout goes to stderr; oversized/incomplete request terminates service. | [Indirect test](tests/test_worker.py#L108) |

### `traces.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`dependency_versions`](traces.py#L19) | Read installed package versions without importing scientific packages. | Installed distribution metadata -> version strings or None. Missing distributions are recorded as None. | [Indirect test](tests/test_cli.py#L55) |
| [`_json_value`](traces.py#L29) | Serialize a dataclass passed into an event. | Value -> dataclass dictionary. Non-dataclass unsupported values raise TypeError. | Source only |
| [`code_hashes`](traces.py#L35) | Hash available production files for a run's provenance. | Installed source files -> SHA256 mapping. Missing optional source files are omitted; read errors propagate. | [Indirect test](tests/test_cli.py#L55) |
| [`EventLog.__init__`](traces.py#L45) | Create a unique private JSONL log and write run provenance. | Directory, record, configuration -> open log. Exclusive creation prevents collision/overwrite; filesystem errors propagate. | [Direct test](tests/test_cli.py#L55) |
| [`EventLog.__call__`](traces.py#L59) | Append and fsync one sequence-numbered event under a lock. | Event type and fields -> durable JSONL line. Rejects reserved-field replacement, closed log, or nonfinite JSON. | [Direct test](tests/test_cli.py#L70) |
| [`EventLog.close`](traces.py#L79) | Close the event stream under its write lock. | Open log -> closed stream. Later event writes fail. | [Indirect test](tests/test_cli.py#L55) |
| [`EventLog.__enter__`](traces.py#L83) | Expose the log through a with block. | Log -> same instance. Does not create a second log. | [Indirect test](tests/test_cli.py#L55) |
| [`EventLog.__exit__`](traces.py#L86) | Close the event stream after a with block. | Context exit -> closed stream. Delegates to close even on exceptions. | [Indirect test](tests/test_cli.py#L70) |

### `evaluation.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`file_sha256`](evaluation.py#L44) | Stream a file's SHA256 for recorded evidence provenance. | Path -> digest. Filesystem errors propagate; this is not automatically a waveform hash. | [Indirect test](tests/test_evaluation.py#L63) |
| [`_number`](evaluation.py#L52) | Normalize an explicit measurement or missing marker. | Number/text/null -> finite float or None. Rejects booleans, invalid numeric text, and nonfinite numeric values. | [Indirect test](tests/test_evaluation.py#L95) |
| [`_decode`](evaluation.py#L66) | Read JSON or a literal-only published Python representation. | Value -> decoded value, or unchanged text. Uses ast.literal_eval, never eval; undecodable text remains text. | [Indirect test](tests/test_evaluation.py#L170) |
| [`ReplayOutput.as_observation`](evaluation.py#L92) | Mark stored output explicitly as replay evidence. | Call ID and optional waveform hash -> ToolObservation. Source file/line identify replay; absent waveform hash is not invented. | [Direct test](tests/test_evaluation.py#L63) |
| [`_replay_output`](evaluation.py#L116) | Adapt recorded numeric fields or formatted class selections. | Record/tool/value/source location -> ReplayOutput. Rejects unknown fields, aliases, labels/probabilities; preserves empty-selection ambiguity. | [Indirect test](tests/test_evaluation.py#L103) |
| [`load_tool_csv`](evaluation.py#L188) | Load strict recorded measurement/classification CSV schemas. | CSV path and kind -> normalized record-to-replay mapping. Rejects schema errors, malformed rows, and colliding record aliases. | [Direct test](tests/test_evaluation.py#L80) |
| [`_dialogue_groups`](evaluation.py#L235) | Group recorded assistant steps under their preceding user question. | Dialogue list -> ordered question/assistant groups. Rejects unknown roles, empty questions, missing actions, and orphan assistants. | [Indirect test](tests/test_evaluation.py#L156) |
| [`_record_alias`](evaluation.py#L256) | Unwrap a recorded single-ECG selection. | Scalar/encoded list -> one identifier. Rejects selection lists containing zero or multiple records. | [Indirect test](tests/test_evaluation.py#L178) |
| [`iter_transcripts`](evaluation.py#L265) | Stream both supported without_gt transcript schemas. | JSONL path -> RecordedCase iterator with stable per-line/turn identity. Rejects teacher forcing, ambiguous schemas, mismatched questions, and malformed rows. | [Direct test](tests/test_evaluation.py#L113) |
| [`legacy_nap`](evaluation.py#L332) | Compute positional action agreement only from saved labels. | RecordedCase -> counts/accuracy or explicit unavailable result. Does not infer missing references or judge answer quality. | [Direct test](tests/test_evaluation.py#L156) |
| [`evaluate_claims`](evaluation.py#L354) | Compare explicit released numbers against named observations. | Claims, evidence, optional record/requested fields -> verdicts and coverage. Rejects duplicate IDs/bad precision; wrong values/units contradict, unavailable evidence is unverifiable. | [Direct test](tests/test_evaluation.py#L195) |
| [`_json_sha256`](evaluation.py#L419) | Hash canonical finite JSON for reproducibility. | JSON-compatible value -> SHA256. Nonfinite or unserializable data raises. | [Indirect test](tests/test_evaluation.py#L224) |
| [`_session_configuration`](evaluation.py#L424) | Capture provider settings, limits, prompt/schema/source hashes. | Session -> JSON-compatible configuration. Collects explicit fields, not credentials; source-file read errors propagate. | [Indirect test](tests/test_evaluation.py#L224) |
| [`_error_kind`](evaluation.py#L448) | Classify a turn error as provider/tool/runner/judge. | Turn dictionary -> category or None. Rejects unknown explicit kind/nontext error; unknown prefix becomes runner. | [Indirect test](tests/test_evaluation.py#L312) |
| [`_package_versions`](evaluation.py#L464) | Read installed measurement dependency versions. | Distribution metadata -> version mapping. Missing numpy/scipy/neurokit2 is represented as None. | [Indirect test](tests/test_evaluation.py#L224) |
| [`run_manifest`](evaluation.py#L474) | Execute sorted fixed cases through the interactive run_turn loop. | Cases, session factory, optional claim extractor -> manifest/turns/summary. Rejects duplicate IDs, invalid history/questions, or wrong records; always closes created sessions. | [Direct test](tests/test_evaluation.py#L224) |
| [`run_manifest.collect_models`](evaluation.py#L512) | Capture actual model/usage metadata while forwarding session events. | Event type and fields -> response trace/log updates. Only model_response events enter the trace. | [Indirect test](tests/test_evaluation.py#L224) |
| [`summarize`](evaluation.py#L560) | Aggregate completion, failure kinds, elapsed time, reuse, and numeric support. | Evaluated-turn iterable -> summary. Rejects invalid IDs/counts/timing; excludes failed/provider/judge turns from claim scores; clinical accuracy stays null. | [Direct test](tests/test_evaluation.py#L321) |
| [`main`](evaluation.py#L645) | Inspect replay CSV/transcripts or summarize evaluated-turn JSONL. | Subcommand/path -> printed JSON and exit code. Bad files/schema exit 2; makes no model requests. | [Direct test](tests/test_evaluation.py#L345) |

### `demo.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`_digest`](demo.py#L43) | Hash exact demo source, question, or MAT bytes. | Bytes -> SHA256. No decoding or clinical interpretation. | [Indirect test](tests/test_demo.py#L60) |
| [`_download`](demo.py#L47) | Fetch one bounded public demo artifact. | Fixed URL and byte limit -> bytes. Thirty-second socket timeout; rejects declared/actual size overflow. | [Direct test](tests/test_demo.py#L136) |
| [`_published_case`](demo.py#L59) | Extract only the pinned first transcript line's measurement turn. | Pinned source -> question, stored measurements, source identity. Rejects wrong hash/record/batch/action/question/values or oversized first line. | [Direct test](tests/test_demo.py#L114) |
| [`convert_record`](demo.py#L88) | Decode the exact 00705_hr WFDB16 source into mV. | Pinned header/data -> float64 12x5000 waveform. Checks header metadata, byte count, initial samples, and per-lead checksums. | [Direct test](tests/test_demo.py#L47) |
| [`_read_bounded`](demo.py#L114) | Read a cached demo file within its limit. | Path and byte cap -> bytes. Oversized input raises; filesystem errors propagate. | [Indirect test](tests/test_demo.py#L65) |
| [`_verify_source`](demo.py#L122) | Require exact pinned header and digital-data hashes. | Header/data bytes -> None. Any changed source byte fails verification. | [Direct test](tests/test_demo.py#L60) |
| [`_verify_existing`](demo.py#L127) | Revalidate a complete cached demo bundle and provenance. | Directory -> existing MAT path. Rejects changed manifest, source, MAT hash/layout/units, or sample values. | [Indirect test](tests/test_demo.py#L65) |
| [`prepare_demo`](demo.py#L160) | Prepare or reuse the exact published sample bundle without overwrite. | Output directory -> verified MAT path. Incomplete bundles fail; partial writes created by this call are cleaned up. | [Direct test](tests/test_demo.py#L65) |
| [`main`](demo.py#L221) | Expose the public-demo preparation command. | CLI output directory -> printed verified MAT path. Missing dependencies or preparation errors exit 1. | Source only |

### `reporting.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`report_instructions`](reporting.py#L51) | Give the model an exact reference-only final-report contract. | Registered tool names -> prompt including enabled label catalog. Classifier instructions are omitted when its tool is absent. | [Direct test](tests/test_reporting.py#L331) |
| [`_unique_object`](reporting.py#L92) | Reject duplicate keys while parsing report JSON. | JSON key/value pairs -> dictionary. Repeated keys raise ValueError rather than overwriting. | [Indirect test](tests/test_reporting.py#L115) |
| [`_usable`](reporting.py#L101) | Check record/hash/status/source/tool eligibility for report evidence. | Observation, current record, allowed tools -> boolean. Replay, runner, failed, foreign, or unregistered evidence is ineligible. | [Indirect test](tests/test_reporting.py#L105) |
| [`_codes`](reporting.py#L109) | Validate fixed explanation/unavailability code lists. | Value and allowed set -> same list. Rejects unknown codes, wrong containers, and duplicates. | [Indirect test](tests/test_reporting.py#L137) |
| [`parse_report`](reporting.py#L117) | Parse only the three permitted final-report fields. | JSON text -> normalized reference/code dictionary. Rejects prose, extra keys, duplicate references, invalid types, or empty reports. | [Direct test](tests/test_reporting.py#L202) |
| [`_latest_classifier`](reporting.py#L147) | Find the latest eligible complete pinned classifier result. | Record, evidence, tools -> observation or None. Malformed/incomplete classifier results are skipped. | [Indirect test](tests/test_reporting.py#L367) |
| [`_explanation`](reporting.py#L159) | Select fixed capability/measurement-limit text for enabled tools. | Explanation code and classifier flag -> text. Only validated codes reach it; no generated medical prose. | [Indirect test](tests/test_reporting.py#L315) |
| [`render_report`](reporting.py#L170) | Resolve references into trusted numbers, units, missingness, and labels. | Report JSON, record, evidence, allowed tools -> answer text. Rejects unusable/ambiguous evidence or score/provenance mismatch; appends fixed limits. | [Direct test](tests/test_reporting.py#L248) |
| [`fallback_report`](reporting.py#L242) | Choose latest measurements and top five valid classifier scores. | Record/evidence/tools -> exact replacement report references. Ignores unusable evidence; stable ties; reports classifier unavailable when absent. | [Direct test](tests/test_reporting.py#L345) |
| [`safe_report`](reporting.py#L261) | Render the deterministic replacement for an invalid final answer. | Record/evidence/tools -> safe answer text. Shares render_report validation; does not rehabilitate generated prose. | [Direct test](tests/test_reporting.py#L163) |

### `classifier.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`load_labels`](classifier.py#L38) | Authenticate the exact 150-label local manifest. | Pinned label bytes -> unique ordered label tuple. Rejects checksum/count/uniqueness changes. | [Direct test](tests/test_classifier.py#L59) |
| [`score_key`](classifier.py#L48) | Format a valid zero-based model head index. | Integer 0..149 -> ecgfounder_NNN. Rejects boolean/noninteger/out-of-range indices. | [Direct test](tests/test_classifier.py#L64) |
| [`label_for_key`](classifier.py#L54) | Resolve a score key through the authenticated manifest. | Exact score key -> official label. Rejects malformed/noncanonical/out-of-range keys. | [Direct test](tests/test_classifier.py#L49) |
| [`provenance`](classifier.py#L63) | Describe the pinned model/source/labels/preprocessing contract. | No inputs -> fresh fixed metadata dictionary. Metadata is a contract, not proof that inference ran. | [Indirect test](tests/test_classifier.py#L92) |
| [`validate_output`](classifier.py#L77) | Authenticate all 150 numeric outputs before evidence use. | Raw observation -> same dictionary after validation. Requires success/live/version, full keys, finite [0,1] scores, units/methods, and exact provenance. | [Direct test](tests/test_classifier.py#L77) |
| [`preprocess`](classifier.py#L103) | Revalidate MAT identity and apply the pinned PTB-XL recipe. | Record -> canonical contiguous float32 12x5000 array. Requires verified 500Hz/mV/ten seconds; rejects changes/flat signal; global z-score, no filtering. | [Direct test](tests/test_classifier.py#L131) |
| [`_checkpoint_bytes`](classifier.py#L130) | Read the same bounded immutable bytes to verify and load. | Local checkpoint path -> verified payload. Requires exact size and SHA256; no runtime download. | [Direct test](tests/test_classifier.py#L184) |
| [`_load_model`](classifier.py#L138) | Load the complete pinned CPU architecture and state dictionary. | Verified payload -> eval-mode frozen Net1D. weights_only=True with scoped float64 metadata allowlist; strict=True rejects missing/extra weights. | [Direct test](tests/test_classifier.py#L200) |
| [`_scores`](classifier.py#L162) | Run one inference-mode forward and sigmoid transform. | Model and batch -> 150 Python floats. Rejects wrong 1x150 head or nonfinite logits/scores. | [Direct test](tests/test_classifier.py#L251) |
| [`Classifier.__init__`](classifier.py#L176) | Authenticate weights and require a finite 150-head CPU smoke forward. | Checkpoint path -> ready classifier. Any load/head/forward failure prevents readiness; no random-head fallback. | [Direct test](tests/test_classifier.py#L231) |
| [`Classifier.provenance`](classifier.py#L187) | Expose the classifier's pinned backend metadata. | Ready object -> provenance dictionary. Delegates to the same fixed module contract. | [Actual](evidence/classifier-smoke.json) |
| [`Classifier.classify`](classifier.py#L190) | Run live scores bound to the revalidated selected recording. | Record -> successful full numeric observation with methods/provenance. Requires readiness; preprocessing/model/output failures raise. | [Direct test](tests/test_classifier.py#L271) |
| [`_file_matches`](classifier.py#L218) | Stream-check an existing checkpoint's size and digest. | Path -> boolean. Overlimit/mismatch returns false; filesystem errors propagate. | [Indirect test](tests/test_classifier.py#L320) |
| [`download`](classifier.py#L230) | Explicitly fetch and atomically install only pinned public weights. | Directory -> verified local checkpoint path. Thirty-second socket timeout; size/hash required; preserves existing bad files; cleans temporary files. | [Direct test](tests/test_classifier.py#L309) |
| [`check`](classifier.py#L271) | Measure CPU readiness, repeat inference, shared worker, and child RSS. | Checkpoint, ECG path, timeout -> reproducibility/timing report. Requires full valid outputs, measurement completion, identical repeated scores, and same process. | [Actual](evidence/classifier-smoke.json) |
| [`main`](classifier.py#L312) | Expose explicit download and local CPU-check subcommands. | CLI arguments -> printed path/hash or JSON and exit code. Catches file/verification errors; no key needed for either command. | [Actual](evidence/classifier-smoke.json) |

### `ecgfounder.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`MyConv1dPadSame.__init__`](ecgfounder.py#L41) | Build the upstream grouped Conv1d with dynamic same padding. | Channel/kernel/stride/group settings -> convolution module. PyTorch rejects incompatible dimensions/groups. | [Indirect test](tests/test_classifier.py#L262) |
| [`MyConv1dPadSame.forward`](ecgfounder.py#L55) | Pad time samples then apply the stored convolution. | Batch/channels/time tensor -> convolved tensor. Requires compatible channels and dtype/device. | [Indirect test](tests/test_classifier.py#L262) |
| [`MyMaxPool1dPadSame.__init__`](ecgfounder.py#L81) | Build the upstream max pool with stride equal to kernel width. | Kernel width -> pooling module. Invalid PyTorch pooling settings fail. | [Indirect test](tests/test_classifier.py#L262) |
| [`MyMaxPool1dPadSame.forward`](ecgfounder.py#L86) | Pad and max-pool the temporal axis. | Tensor -> pooled tensor. Used by residual downsampling; shape/type errors propagate. | [Indirect test](tests/test_classifier.py#L262) |
| [`Swish.forward`](ecgfounder.py#L101) | Apply the upstream x times sigmoid(x) activation. | Tensor -> same-shape activated tensor. Requires a PyTorch-compatible numeric tensor. | [Indirect test](tests/test_classifier.py#L262) |
| [`BasicBlock.__init__`](ecgfounder.py#L123) | Construct bottleneck convolutions, squeeze/excitation, and residual path. | Block dimensions/settings -> stored modules. Pinned classifier disables BN/dropout; other generic settings are not separately validated here. | [Indirect test](tests/test_classifier.py#L262) |
| [`BasicBlock.forward`](ecgfounder.py#L181) | Apply the bottleneck, attention scaling, and residual addition. | Tensor -> residual block output. Incompatible tensor shapes fail; no clinical logic. | [Indirect test](tests/test_classifier.py#L262) |
| [`BasicStage.__init__`](ecgfounder.py#L241) | Stack configured residual blocks for one resolution stage. | Stage channels/block count/stride -> block list. First-block/downsampling flags follow pinned upstream construction. | [Indirect test](tests/test_classifier.py#L262) |
| [`BasicStage.forward`](ecgfounder.py#L287) | Pass the tensor through the stage's ordered blocks. | Tensor -> stage output. Propagates convolution/residual errors. | [Indirect test](tests/test_classifier.py#L262) |
| [`Net1D.__init__`](ecgfounder.py#L327) | Construct the convolutional stem, seven-stage body, and classifier head. | Architecture settings -> Net1D modules. Adapter fixes 12 input leads and 150 outputs; other configurations are upstream generic behavior. | [Indirect test](tests/test_classifier.py#L262) |
| [`Net1D.forward`](ecgfounder.py#L379) | Compute temporal features, global mean pooling, and dense logits. | Batch x channels x samples -> logits, optionally features. Returns logits without sigmoid; output head checked by adapter. | [Indirect test](tests/test_classifier.py#L262) |

### `benchmark.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`_sha`](benchmark.py#L44) | Hash exact benchmark artifacts or serialized manifests. | Bytes -> SHA256. Does not authenticate a source without a pinned comparison. | [Indirect test](tests/test_benchmark.py#L84) |
| [`_download`](benchmark.py#L49) | Fetch a bounded public benchmark artifact with one transport retry. | URL and byte cap -> bytes. Thirty-second socket timeout; oversize fails; a second transport error propagates. | Source only |
| [`select_cases`](benchmark.py#L68) | Select five distinct fold-9 patients per frozen annotation group. | Pinned metadata/SCP registry -> twenty-case manifest. Rejects hash changes or insufficient groups; excludes demo705 and its patient. | [Direct test](tests/test_benchmark.py#L75) |
| [`decode_record`](benchmark.py#L121) | Convert verified-format public WFDB16 records into canonical mV. | Header, data, ECG ID -> float64 12x5000 array. Checks format, length, 500Hz, gain/baseline, lead set, initial samples, and checksums. | [Direct test](tests/test_benchmark.py#L224) |
| [`_read`](benchmark.py#L153) | Bound a benchmark cache read. | Path and byte limit -> bytes. Oversized cached input raises. | [Indirect test](tests/test_benchmark.py#L244) |
| [`_case_path`](benchmark.py#L162) | Constrain manifest waveforms to expected public record paths. | Case -> relative records500 path. Rejects arbitrary paths/URLs and filename/ECG-ID mismatch. | [Indirect test](tests/test_benchmark.py#L89) |
| [`_prepare_case`](benchmark.py#L172) | Verify/fetch raw sources and exclusively create or verify their MAT. | Case, cache directory, freezing flag -> MAT path plus source hashes. Rejects source/MAT changes; preserves existing bytes rather than overwriting. | [Direct test](tests/test_benchmark.py#L244) |
| [`freeze_manifest`](benchmark.py#L211) | Freeze source hashes after deterministic patient selection. | Pinned metadata/registry and cache directory -> complete manifest. Uses up to four download/conversion workers; any invalid source fails. | Source only |
| [`load_manifest`](benchmark.py#L222) | Validate the fixed twenty-patient benchmark sidecar. | Manifest path -> dictionary. Requires pinned provenance, groups, unique patients, fold9, excluded IDs, references, and frozen URLs/hashes. | [Direct test](tests/test_benchmark.py#L89) |
| [`prepare`](benchmark.py#L247) | Prepare or revalidate all cases against the frozen sidecar. | Manifest and cache directory -> ECG-ID/MAT-path mapping. Parallel source/converter errors propagate; reference annotations stay outside MAT inputs. | [Actual](evidence/benchmark-direct.json) |
| [`rank_auroc`](benchmark.py#L255) | Calculate exploratory AUROC using average ranks for ties. | Boolean labels and finite scores -> AUROC or None. Rejects length/type/nonfinite errors; returns None without both classes. | [Direct test](tests/test_benchmark.py#L95) |
| [`_validate_result`](benchmark.py#L276) | Authenticate direct-tool record identity and full scientific output contracts. | Raw result, selected record, tool -> validated raw dictionary. Rejects wrong record/hash, malformed150score output, invalid measurement fields/units/methods/missingness, and nonfinite JSON. | [Indirect test](tests/test_benchmark.py#L150) |
| [`run_direct`](benchmark.py#L311) | Run both real tools for every public challenge record. | Manifest, paths, checkpoint, timeout -> completions, scores, metrics, failures. Keeps annotation references outside model inputs; partial measurements count separately from clinical accuracy. | [Actual](evidence/benchmark-direct.json) |
| [`MockBoundaryModel.complete`](benchmark.py#L375) | Drive five fixed dialogue turns using native calls and evidence references. | History and schemas -> Decision. Mocked language-model boundary; numerical tools remain real; no reference labels enter it. | [Actual](evidence/benchmark-dialogue.json) |
| [`MockBoundaryModel.close`](benchmark.py#L400) | Satisfy provider cleanup for the deterministic fixture. | No inputs -> None. Owns no network or worker resources. | [Actual](evidence/benchmark-dialogue.json) |
| [`MockBoundaryModel.interrupt`](benchmark.py#L403) | Satisfy provider cancellation for the deterministic fixture. | No inputs -> None. Owns no network request to cancel. | Source only |
| [`released_claims`](benchmark.py#L407) | Extract numbers only from the report references actually released. | Case/index/result -> explicit MeasurementClaim tuple. Rejects references to absent evidence; omitted or null findings do not become claims. | [Direct test](tests/test_benchmark.py#L103) |
| [`run_dialogue`](benchmark.py#L423) | Evaluate five-turn sessions for one case per annotation group. | Manifest, paths, checkpoint, optional model -> canonical dialogue report. Without key, live mode is pending before worker creation; actual artifact uses a mocked model with real tools. | [Actual](evidence/benchmark-dialogue.json) |
| [`run_dialogue.session_factory`](benchmark.py#L441) | Create one prepared worker, provider, log, and Session per case. | EvaluationCase -> shared-runtime Session. Preparation/provider setup failure closes the worker; annotations are excluded from history. | [Actual](evidence/benchmark-dialogue.json) |
| [`main`](benchmark.py#L480) | Expose prepare, direct, and mock/live dialogue benchmark commands. | CLI arguments -> output JSON or exclusive report file and exit code. Bad files/options/dependencies exit1; pending/incomplete runs return1; refuses report overwrite. | [Direct test](tests/test_benchmark.py#L183) |

### `providers/openrouter.py`

| Function | Purpose | Inputs, outputs, and failures | Evidence |
|---|---|---|---|
| [`_reject_constant`](providers/openrouter.py#L21) | Reject nonfinite constants while decoding tool-argument JSON. | NaN/Infinity token -> exception. Always raises ValueError. | [Indirect test](tests/test_openrouter.py#L184) |
| [`OpenRouter.__init__`](providers/openrouter.py#L28) | Configure the stdlib native-tool adapter and cancellation state. | Explicit model/key, token cap, timeout -> adapter. Requires nonempty model/key and positive token/timeout settings. | [Direct test](tests/test_openrouter.py#L223) |
| [`OpenRouter.set_deadline`](providers/openrouter.py#L47) | Assign or clear the next request's monotonic deadline. | Deadline or None -> adapter state. The runner resets this per model request. | [Direct test](tests/test_openrouter.py#L201) |
| [`OpenRouter._safe`](providers/openrouter.py#L50) | Redact the configured API key from error text. | Error/object -> sanitized string. Only exact configured-key occurrences are replaced. | [Indirect test](tests/test_openrouter.py#L166) |
| [`OpenRouter.complete`](providers/openrouter.py#L53) | Send native schemas/messages and parse a bounded-retry decision. | Messages and tools -> Decision with actual model/usage. One retry for transient/empty responses within one watchdog deadline; records attempts and redacts errors. | [Direct test](tests/test_openrouter.py#L120) |
| [`OpenRouter._request`](providers/openrouter.py#L109) | Make one HTTPS chat-completions request and read its JSON. | Encoded body and deadline -> response dictionary. 408/429/5xx retryable; authentication/API/JSON errors fail; always closes connection. | [Indirect test](tests/test_openrouter.py#L160) |
| [`OpenRouter._set_socket_timeout`](providers/openrouter.py#L155) | Apply remaining request time to the active socket. | Connection and monotonic deadline -> updated timeout. Raises when the request deadline has elapsed. | [Indirect test](tests/test_openrouter.py#L61) |
| [`OpenRouter._decision`](providers/openrouter.py#L162) | Validate a provider response and preserve native tool-call identities. | Response dictionary -> Decision. Rejects truncation, invalid roles/content/calls/IDs/argument JSON/usage; empty choices can retry. | [Indirect test](tests/test_openrouter.py#L184) |
| [`OpenRouter._expire`](providers/openrouter.py#L217) | Mark watchdog expiry and abort the request socket. | Timer callback -> expired flag and socket shutdown. Late success cannot bypass expired state. | [Direct test](tests/test_openrouter.py#L126) |
| [`OpenRouter._abort_socket`](providers/openrouter.py#L221) | Shut down an active HTTPS socket to unblock I/O. | Adapter connection -> socket shutdown. Absent sockets are harmless; shutdown OSError is ignored. | [Indirect test](tests/test_openrouter.py#L209) |
| [`OpenRouter.interrupt`](providers/openrouter.py#L229) | Mark cancellation and abort the active request. | Adapter -> interrupted flag and socket shutdown. Interrupted requests do not retry. | [Direct test](tests/test_openrouter.py#L209) |
| [`OpenRouter.close`](providers/openrouter.py#L233) | Cancel outstanding provider work. | Adapter -> interrupt side effects. Delegates to interrupt; has no message-history store. | [Indirect test](tests/test_integration.py#L20) |
