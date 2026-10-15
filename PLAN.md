# Reduce Eko into an ECG research runtime

Accepted scope: named tools, OpenRouter initially, real measurements first,
classification after checkpoint verification. Source audit is pinned to official
ECG-Agent commit `0ab07899ec0a39b087bebd560b280cad484eb0d0`.

## Foundation

Keep repeated model decisions, actual observations, conversation, timeouts,
interruption, and attribution. Remove generated Python, child agents, Unix sockets,
world relay, Linux sandboxing, rich terminal rendering, Claude/HF dialogue backends,
feral mode, and automatic context resets. Develop and publish on `ecg-core` in
`jiwoonoh/eko`, retaining the original in Git history.

Use `Decision`, `ToolObservation`, and `run_turn(session, question) -> TurnResult`.
One session owns history and evidence. OpenRouter native calls translate into the
same model-neutral decisions a later local model adapter will use. Predicted
Tool_Output is never accepted as an observation.

Render final answers from evidence references and fixed explanations. Do not
display unrestricted generated medical prose. Reject malformed reports and
unsupported conclusions with a safe evidence summary, failed completion, and an
auditable rejection event; keep rejected text out of follow-up history. Median RR
does not establish regularity and estimator filter bounds do not establish normality.

One persistent registered-tool worker reuses scientific state and terminates on
hard timeout. The controller owns credentials and logs. Defaults: eight responses,
four tool calls per turn, thirty seconds per tool, sixty seconds per model request,
one bounded transient retry. Expose a basic interactive/single-question CLI with
explicit model and record selection.

## Measurements

Extract a NumPy/SciPy/NeuroKit2 backend without importing MedRAX or its image/model
packages. Require full 12-lead MAT data, valid metadata sampling frequency, and
verified lead order. Unknown waveform units stay unknown.

Preserve beat identity for QT/RR pairing. Return heart rate, RR, PR, Q–S peak span,
and documented Bazett QTc, with numeric/null values, units and methods. Preserve
raw estimates, filtering reasons, partial status and missingness. Exclude hardcoded
diagnostic interpretation. Version corrections separately from the upstream
reference, including preceding-RR QTc pairing and metadata-aware sampling rate.

Persist actual evidence, record hashes, tool/runtime versions, call IDs, errors and
timings in unique append-only JSONL run logs. Keep data, logs, weights and secrets
out of Git. Preserve upstream MIT attribution.

## Evaluation

Use separate tracks for recorded-output regression, live measurement validity,
and agent reporting. Interactive and manifest evaluation use one runner.

Support both published `{ecg_id, dialogue, batch_idx}` and current inference
transcript formats through explicit adapters. Keep source SHA/line identities and
replicates; do not deduplicate by ECG. Resolve padded aliases uniquely, reject
collisions, and preserve blank/null measurement reasons as unknown when absent.

Align by sample/user turn/tool step; legitimate additional calls do not shift
later comparisons. Include failed/unanswered cases in completion counts and keep
provider/judge failures separate from answer-quality scores. Freeze sorted manifests
and configurations; preserve assisted vs autonomous history labels. Legacy action
agreement is secondary and requires actual reference actions. Broad prose and
clinical correctness verification remain later capabilities.

## Delivery and acceptance

Three reviewable changes: core teardown; live measurements and evidence; evaluation
adapters and acceptance tests. Push the tested branch with this plan and README.

Test multiple calls, worker reuse/restart, invalid/empty/truncated model output,
limits/cancellation, non-500-Hz input, invalid/missing lead order, flat/nonfinite
signals, partial delineation, missing intermediate QT, units/nulls/filtering,
record aliases/collisions, repeated queries/generations, record-bound evidence,
unique logs, and replay/live distinction. Use controlled signals and mock model
responses. Since no local records were supplied, use the exact HR00705 record from
the first published ECG-Agent transcript, fetching its public PTB-XL waveform with
pinned checksums and preserving the published measurement question/reference.

First milestone: actual measurements from a selected ECG reach the model, produce
an explanation, and support follow-up questions. Classification stays unavailable.
Then verify checkpoint readiness, expected 71-label dimensions/order, preprocessing,
window/frequency, thresholds, and fairseq compatibility before exposing classification.
Later: local dialogue models, structured claim checks, clinical interpretation,
independent measurement/diagnosis validation, and SFT/RL.
