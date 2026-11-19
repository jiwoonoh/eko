# Add an ECG tool

Eko and the benchmark read the same `TOOLS` registry. For a Python tool, write
the function, register it once, then select it in each run. The example below
adds `recording_duration`, which returns the length of an ECG in seconds.
Run the commands from the repository root.

## 1. Write the function

Create `toolbox/recording_duration.py`. Give the function the same name as the
module, accept the ECG file path, and return the value the agent should see.

```python
from .record import load_record


def recording_duration(ecg_path: str) -> float:
    """Return the recording duration in seconds."""
    fields, sample_rate = load_record(ecg_path)
    return fields["feats"].shape[1] / sample_rate
```

General Eko uses the ECG path in the user's request. The benchmark supplies
`record.mat` with a 500 Hz waveform in `feats`, shaped `(12, samples)`.
Single-lead cases keep the named channel and zero the other eleven.

## 2. Register it

In [registry.py](registry.py), add `from .base import Tool` alongside the imports.
Add this entry **inside the existing `TOOLS` dictionary**.

```python
"recording_duration": Tool.from_module(
    "recording_duration", "To measure recording duration (seconds)", files=("record.py",),
),
```

The dictionary key is the name used by `--tools`. The first argument is the
module path relative to `toolbox`. Eko copies that module and generates the
agent's import, call, and print example. List any other toolbox files the
function needs in `files`. This example imports `.record`, so it includes
`record.py`. For a new third-party dependency, add it with `uv add PACKAGE`.
For a nested package, include an `__init__.py` and use its dotted module path.
For a different call signature, use `Tool(files=..., prompt=...)` and supply a
`python-run` example that prints the result. See [base.py](base.py).

To check the calculation directly, replace the path with a known ECG. A
5,000-sample recording at 500 Hz should return `10.0`.

```bash
uv run python -c "from toolbox.recording_duration import recording_duration; print(recording_duration('/path/to/ecgs/example.mat'))"
```

## 3. Use it in general Eko

With SGLang serving the model below on port 30001, point `--cwd` at the
directory containing your ECG. Replace the directory and filename with yours.

```bash
uv run python host.py "Use recording_duration on example.mat." \
  --cwd /path/to/ecgs --model Qwen/Qwen3.8-27B \
  --sglang-url http://127.0.0.1:30001 --tools recording_duration
```

`--tools` specifies the entire selection. To offer the new tool alongside the
defaults, use `--tools measurement classification recording_duration` and supply
`--classifier-checkpoint /path/to/checkpoint.pt`.

## 4. Use it in a benchmark experiment

Copy the config for the lead condition you want. Start this example with the
[12-lead baseline](../benchmark/experiments/12-lead-gpt.json).

```bash
cp benchmark/experiments/12-lead-gpt.json benchmark/experiments/12-lead-duration.json
```

Change these four fields in the copy.

| JSON field | Value |
| --- | --- |
| `run.tools` | `["measurement", "classification", "recording_duration"]` |
| `run.output` | `"benchmark-results/12-lead-duration/responses.jsonl"` |
| `score.responses` | `"benchmark-results/12-lead-duration/responses.jsonl"` |
| `score.output` | `"benchmark-results/12-lead-duration/scores.jsonl"` |

This adds the tool to the baseline, gives the experiment its own outputs, and
uses the copied config's dataset, references, and judge. Set `ECG_DIR` to the
preprocessed MAT directory and `CLASSIFIER_CHECKPOINT` to the baseline checkpoint.
Scoring also requires `OPENROUTER_API_KEY`.

```bash
config=benchmark/experiments/12-lead-duration.json
uv run python -m benchmark.run --config "$config" \
  --model Qwen/Qwen3.8-27B --sglang-url http://127.0.0.1:30001 \
  --ecg-dir "$ECG_DIR" --classifier-checkpoint "$CLASSIFIER_CHECKPOINT"
uv run python -m benchmark.score --config "$config"
```

For a one-case check, append `--max-samples 1` to both commands.

## When the tool needs beat measurements

In `toolbox/measurements/<name>.py`, import `prepare_beats` from `.preprocessing`
and call `beats = prepare_beats(ecg_path)`. It cleans the measurement lead
(default II) and detects R-peaks. Use `beats.rr_seconds` for RR intervals or
`beats.waves` for wave landmarks. See [heart_rate.py](measurements/heart_rate.py).

For a function named `rr_interval` in `toolbox/measurements/rr_interval.py`, the
registry entry is one line. The helper includes the preprocessing files and
identifies the lead in the agent's instructions.

```python
"rr_interval": measurement_tool("rr_interval", "RR interval (ms)"),
```

Select it by name as in steps 3 and 4. The baseline `measurement` tool returns
heart rate, PR, QRS, and QTc together. For separate calls, select any of
`heart_rate`, `pr_interval`, `qrs_duration`, and `qtc_interval`.

## When the tool needs a host service

Subclass `Tool`, register the instance in `TOOLS`, and list the agent-side client
files in `files`. Keep the checkpoint and service code on the host. See the
[classifier implementation](neural_networks/classification/tool.py).

- `add_arguments(parser)` declares options shared by both entry points.
  Benchmark JSON uses the underscore spelling, such as `run.classifier_checkpoint`.
- `load(args)` validates settings and loads the model once per run when selected.
  Defer model imports to this hook.
- `start(state, resources, workspace, runtime, *, sandbox)` starts the service
  for each workspace and returns the client's environment variables. Register
  cleanup with `resources.callback(...)`. A socket under `runtime` is visible
  to the sandbox under `/run/eko`.
