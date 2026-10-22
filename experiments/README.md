# Reporting ablation protocol

Measure full request completion while preserving evidence validation. Clinical
accuracy is outside this experiment.

Freeze the free model, waveform/checkpoint bytes, twenty dialogue turns, tool
versions and execution limits. Reference ECG annotations stay outside model
history. The five-turn schedule tests acquisition, reuse, classifier refresh,
measurement follow-up and explanation of limits.

The evaluator requires every requested finding, including explained nulls, the
latest relevant evidence, the specified tool schedule and both limits explanations
on the final turn. Rejected replacement reports never count as completed tasks.
Renderer acceptance is reported separately. Do not alter prompts, cases, expected
fields or the gate to raise a variant's score.

## Screens and parents

Each row changes one factor from its parent. These are single-run screens with
provider variation, not estimates of statistical significance. Preserve every
result, including nonimproving runs. The source and configuration hashes in the
[saved comparison](../evidence/ablations/summary.json) identify each actor.

| Screen | Parent | Change | Full tasks | Decision |
|---|---|---|---:|---|
| Control | Original runtime | Task scoring and recorder only | 4/20 | Reference |
| Aliases | Control | Exact classifier abbreviations | 3/20 | Removed |
| Native report | Control | Controller-handled report submission | 8/20 | Provisional |
| Required report | Native report | Require native decisions | 2/20 | Removed |
| Compact view | Native report | Compact observation messages | 5/20 | Removed |
| Report correction | Native report | One bounded report correction | 8/20 | Removed |
| Structured history | Native report | Keep validated report JSON in history | 7/20 | Provisional format benefit |
| History and aliases | Structured history | Exact classifier abbreviations | 16/20 | Retained for next screen |
| Explanation meanings | History and aliases | Expose existing fixed explanation text in schema | 18/20 | Confirmation failed: 18, 15, 15 |
| Selector role | Explanation meanings | Describe model as tool/report selector | 18/20 | Removed |
| Temperature zero | Explanation meanings | Set sampling temperature to zero | 18/20 | Freeze for confirmation |

Structured history accepted 17/20 reports versus 11/20 for native submission, but
completed fewer full tasks. It remained provisional to isolate the residual label
selection error. Reintroducing exact aliases with structured history completed
16/20 tasks with no accepted SR/NSR substitutions. All four remaining failures
were on the limits question. Explanation meanings then completed 18/20 tasks with
two rejected final reports and no accepted SR/NSR substitutions. No finding
values, diagnostic thresholds, request matching or reference answers were added.

Required-tool, compact-view and correction behavior was removed before subsequent
screens. Rejected combinations have no runtime switches. The correction trial
recovered two tasks after rejection, matched native submission overall and used
more tokens. Compact observations saved tokens but completed fewer tasks. The
endpoint did not reliably honor required native decisions.

## Confirmation gate

Freeze the selected implementation for three sequential repeats. Preserve all
three even if an early repeat fails. The gate requires:

- At least eighteen fully completed tasks in each twenty-turn repeat.
- No accepted SR/NSR substitutions.
- Complete requested missingness handling.
- All negative evidence-binding tests passing.

This gate covers four development records and fixed questions. Held-out dialogue,
clinical labels and broader generalization remain separate work. Model identity,
source/configuration hashes, usage and latency must be retained for every repeat.
Provider sampling defaults and endpoint behavior can vary.

The user selected `cohere/north-mini-code:free` only. No paid model comparison will
run. Run from the repository root:

```bash
./.venv/bin/python -m experiment experiments/reporting.json --label candidate --repeats 1
./.venv/bin/python -m experiment experiments/reporting.json --label confirmation
```

The first command screens one candidate; the second uses the configuration's
three repeats. Each creates a new directory under ignored `runs/ablations/`. The
key is read from the environment or ignored `.env` as data. Commit sanitized
reports only after checking that they contain no credentials, waveforms or raw
interaction logs. Reported input/cost totals include every provider attempt that
supplied usage; missing usage is not reconstructed.

The explanation-meanings candidate completed 18/20, 15/20 and 15/20 tasks in
confirmation. The gate failed. No accepted SR/NSR substitutions occurred and all
45 requested null values had displayed explanations. All 60 released reports
were independently re-rendered from local evidence in the saved audit.

The selector-role screen changed only the role description. Its hypothesis was
that describing a tool/report selector would reduce prose completions.

The selector-role screen matched the previous screen at 18/20 and was removed.
The sampling screen then changed only temperature from the endpoint default to
zero. All role wording, cases, limits and validation remain those of the failed
confirmation candidate. The current free endpoint advertises temperature support.
Zero does not guarantee deterministic inference.

The temperature-zero screen accepted all twenty reports and completed eighteen
full tasks. Two accepted reports omitted the classifier-limits code. No tool or
provider errors occurred. `zero-confirmation` then ran all three prespecified
repeats with the original role wording and the same gate.

Temperature-zero confirmation completed 17/20, 20/20 and 16/20 tasks. It failed
the gate. No accepted SR/NSR substitutions occurred; all 45 requested nulls had
displayed reasons. This improves observed aggregate completion over the first
confirmation (53/60 versus 48/60), without a significance claim.

A one-correction prototype is now implemented and unit-tested. Its live screen
on the zero-temperature reporter is pending; research execution is paused. The original correction screen preceded structured
history, exact aliases and explanation meanings; it completed 8/20 and was
removed. The planned retest addresses the current residual validation failures only.
It cannot execute scientific tools or exceed eight model responses. Raw rejected
text remains excluded from model history. First-pass tasks, recovered tasks,
rejections, extra responses and usage are reported separately. Valid but incomplete
reports remain incomplete; no task predicate or expected explanation changes.
