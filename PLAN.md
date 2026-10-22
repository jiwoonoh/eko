# ECG runtime implementation record

The original direction was named tools, OpenRouter first, live measurements first,
then classification after checkpoint verification. The measurement reference is
ECG-Agent revision `0ab07899ec0a39b087bebd560b280cad484eb0d0`.

## Completed foundation

- Replaced Eko's generated-code agent, child agents, sockets, Linux sandbox, and
  terminal UI with `Decision`, `ToolObservation`, `Session`, and shared `run_turn`.
- Kept repeated native tool calls, conversation, cancellation, hard worker timeouts,
  request limits, and bounded transient provider retry.
- Added metadata-validated twelve-lead MAT inputs and real NeuroKit2 measurements.
  Corrected status preservation, measurement labels, and beat-aligned QT/RR pairing.
- Added a verified public HR00705 download, explicit evidence identities, input and
  source hashes, dependency versions, and append-only JSONL logs.
- Added CSV/transcript adapters that preserve aliases, missingness, repeated
  generations, and user-turn/tool-step alignment.
- Replaced unsupported model prose with rendered evidence references and fixed
  explanations. Rejected answers are failed and logged; they do not enter
  follow-up history. Median intervals do not establish a rhythm diagnosis.

History: `ac4aad8`, `37a7557`, `344fc8b`, `8d4e838`, `973537b`.

## Completed classification delivery

1. **Backend (`1f96d8f`):** extracted the attributed ECGFounder inference
   architecture, preserved the trained 150-label head, pinned source/model/label
   revisions, and required full parameter matching. Explicit checkpoint download
   verifies SHA-256 and preserves existing files. Inputs require verified twelve
   leads, 500 Hz, 5,000 samples, and mV units. Global z-score normalization follows
   the pinned PTB-XL evaluator without added filtering. CPU readiness, finite
   dimensions, repeatability, latency, and memory were checked on HR00705.
2. **Runtime/reporting (`8bc21d9`):** added `call_classification_tool(record_id)` to
   the persistent worker. Readiness failure keeps measurements available. All 150
   independent scores and provenance remain in JSON/logs. Rendered labels come
   from the trusted manifest; no diagnostic thresholds are imposed. Optional
   `TurnResult.report` records released references for coverage scoring.
3. **Benchmark/documentation (this revision):** froze twenty distinct fold-9
   patients, excluding HR00705 and its patient. Ran both scientific tools on all
   twenty and five dialogue turns on four representatives. Dialogue used a
   scripted provider and real tools. Dataset references stayed outside model
   history and classifier inputs. README records the teardown, capabilities,
   methods, execution evidence, limitations, and every production function.

The public 150-label checkpoint is an explicit alternative to ECG-Agent's
unavailable 71-label weights. These are different models and label spaces.

## Verification and delivery

Keep weights, waveforms, secrets, and raw interaction logs outside Git. Commit
sanitized evidence with input/configuration/source hashes and actual dependency
versions. Test checkpoint integrity, preprocessing, dimensions, record binding,
malformed outputs, score/report validation, evidence reuse, timeout restart,
partial results, aliases, repeated samples, and separate failure counts.

Deliver the backend, runtime/reporting, and benchmark/documentation as three
reviewable commits on `ecg-core` in `jiwoonoh/eko`. README links each behavior to
implementation and evidence. Full function-reference links are checked against
final source definitions. Automated provider tests are explicitly mocked.

## Pending, blocked, and deferred

- **Historical live baseline:** the original free Cohere run accepted nine of
  twenty reports and completed all twelve scientific calls. The later controlled
  baseline completed four full tasks. These are different metrics; all baseline
  artifacts remain available.
- **Current candidate:** native report submission, validated JSON in assistant
  history, exact classifier abbreviations, and explanation meanings in the schema.
  The screen completed 18/20 tasks without accepted SR/NSR substitutions. Confirmation
  completed 18/20, 15/20 and 15/20; the gate failed. All 45 requested nulls had
  displayed explanations and no accepted SR/NSR substitutions occurred. A
  role-description screen changes only the model responsibility wording.
- **Blocked:** exact ECG-Agent 71-label reproduction without its checkpoint,
  verified ordering/input dimensions, preprocessing, and fairseq revision.
- **Deferred:** diagnostic thresholds/calibration, clinical validation, local
  dialogue adapters, broader structured claim verification, structured ECG
  interpretation, training, and SFT/RL. Reserve fold 10 for later evaluation.
- **Required before clinical claims:** independent measurement annotations and
  diagnosis references, an appropriate evaluation population, and a documented
  validation protocol. Development AUROCs are exploratory only.


## Reporting reliability study (22 October 2026)

Full-task checks include nulls, requested labels, evidence freshness, the frozen
tool schedule and both limits explanations. The configuration fixes the model,
cases, questions and execution limits. Every screen records source/configuration
hashes and all reported provider-attempt usage.

Each screen changes one factor relative to its stated parent. Native submission
completed 8/20 tasks; structured history completed 7/20 but reduced rejected
formats. Exact aliases with that history reached 16/20. Exposing the existing
explanation meanings reached 18/20. The README and sanitized comparison preserve
all screens, including failures and removed interventions. No report correction,
required-tool option or compact observation view remains in the runtime.

The development gate requires at least 18/20 full tasks in each of three repeats,
no accepted SR/NSR substitutions, explained requested nulls, and passing negative
evidence tests. The first confirmation failed at 18/20, 15/20 and 15/20. A role-description screen is next. The user selected the existing free
model only; paid comparisons will not run. These checks do not establish clinical
accuracy or performance on unseen dialogue.
