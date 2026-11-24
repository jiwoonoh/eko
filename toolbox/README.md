# Add an ECG tool

NOTE: Probably your agent can just add the tool for you, but here is a step by step of whats actually needed.

Eko and the benchmark read the same `TOOLS` registry. For a Python tool, write
the function, register it once, then select it in each run. The example below
adds `recording_duration`, which returns the length of an ECG in seconds.
Run the commands from the repository root.

Put each tool in a category package. Use `measurements/` for direct ECG measurements, `neural_networks/` for neural network tools. For tools that don't fit in either of these categories, create a new folder.

## 1. Write the function

Create `toolbox/measurements/recording_duration.py`. Give the function the same
name as the file, accept the ECG file path, and return the value the agent
should see.

```python
from ..record import load_record


def recording_duration(ecg_path: str) -> float:
    """Return the recording duration in seconds."""
    fields, sample_rate = load_record(ecg_path)
    return fields["feats"].shape[1] / sample_rate
```

## 2. Register it

Define the tool in [measurements/tool.py](measurements/tool.py).

```python
RECORDING_DURATION = Tool.from_module(
    "measurements.recording_duration",
    "To measure recording duration (seconds)",
    files=("record.py",),
)
```

Use the dotted module path relative to `toolbox`. List any other toolbox files
the function imports in `files`.

In [registry.py](registry.py), import `RECORDING_DURATION` from
`.measurements.tool` and add this entry to `TOOLS`.

```python
"recording_duration": RECORDING_DURATION,
```

The dictionary key is the name used by `--tools`.

## 4. Integrating tool with benchmark

Create the benchmark config with your tools. Change these four fields in the json config.

| JSON field | Value |
| --- | --- |
| `run.tools` | `["measurement", "classification", "recording_duration"]` |

This adds the tool to the baseline.

## When the tool needs a host service

The classifier and delineator use host services. Each loads its model once per
run and serves predictions to the sandbox. If your tool also needs to
run outside the sandbox, follow the [classifier example](neural_networks/classification/tool.py).
