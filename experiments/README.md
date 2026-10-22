# Reporting ablation protocol

Objective: improve full request completion while preserving evidence validation.
Clinical accuracy is outside this experiment.

Freeze the model, waveform/checkpoint bytes, twenty dialogue turns, tool versions,
and execution limits before changing the agent. Reference ECG annotations remain
outside model history. The five-turn schedule tests acquisition, reuse, explicit
classifier refresh, measurement follow-up, and explanation of limits.

The evaluator requires every requested finding, including explained null values,
the latest relevant evidence, the specified tool schedule, and both measurement
and classifier limits on the final turn. Rejected replacement reports never count
as completed tasks. Renderer acceptance is reported separately.

| Stage | Single intervention | Hypothesis |
|---|---|---|
| Control | Existing agent; task metrics and experiment recorder only | Reproduce current format and label-selection failures. |
| A | Explicit canonical label aliases | Remove SR/NSR selection substitutions. |
| B | Controller-handled report submission schema | Reduce malformed reports without weakening evidence checks. |
| B2 | Require native tool decisions with a terminal report action | Prevent plain-text completions that ignore the report schema. |
| C | Compact model observation messages | Reduce prompt tokens while preserving all scores, findings and missingness. |
| Conditional | One bounded report correction | Consider only if residual schema failures prevent the development gate. |

Screen each intervention on one fixed twenty-turn run. The alias-only screen
completed 3/20 tasks versus 4/20 in the fresh control and was removed before
stage B. Stage B therefore compares native report submission with the control
agent. A label-guidance follow-up is warranted only if its accepted reports still
select incorrect labels. Preserve failed and
nonimproving runs. Keep a change only for a stated measured benefit; do not add
runtime switches for rejected combinations. Confirm the selected implementation
with three repeats and report each repeat, model identity, source/configuration
hashes, cost, token use and latency. Sampling defaults can vary at the provider;
repeat results are evidence of variation, not a claim of deterministic inference.

Development gate: at least eighteen fully completed tasks in each twenty-turn
repeat, no accepted SR/NSR substitutions, complete requested missingness handling,
and all negative evidence-binding tests still passing. This gate measures the
specified dialogue suite; independent clinical labels and broader generalization
remain separate work. Do not alter prompts, cases, expected fields or the gate to
raise a variant's score.

Run from the repository root. The key file is read as data and never committed:

```bash
./.venv/bin/python -m experiment experiments/reporting.json --label control --repeats 1
```

Omit `--repeats 1` for the configuration's three-repeat confirmation. Each run
creates a new directory under ignored `runs/ablations/`. Commit only sanitized
reports and a concise comparison after verifying that no credentials or raw
interaction logs are included.
