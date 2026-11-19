# Tools

Each tool owns its code, usage example, and optional host setup.
Define it beside its implementation in `tool.py`, then register it in
`toolbox/registry.py`. `toolbox/base.py` defines the shared interface.
Regular Eko and benchmark generation share the registry, CLI options, examples,
and host setup.

## Select tools

Both entry points default to measurement and classification. Override either with:

```bash
--tools measurement classification --classifier-checkpoint /path/to/checkpoint.pt
--tools measurement
--tools classification --classifier-checkpoint /path/to/checkpoint.pt
--tools
```

In an experiment JSON, set `"tools": ["measurement"]` under `"run"`.
Use `[]` for no tools. Give each ablation separate response and score files.
The classifier checkpoint is required only when classification is enabled.
Newly registered tools require explicit selection; edit `DEFAULT_TOOLS` to change
the defaults.

Scoring uses the original fixed judge prompts, independent of tool selection.
Edit `benchmark/prompts.py` manually when needed.

## Add a tool

1. Put agent code under `toolbox/`, for example `toolbox/rhythm/analyze.py`.
   Keep its `__init__.py` lightweight.
   Install dependencies in Eko's environment. Add new packages to `pyproject.toml`
   if distributing them. The benchmark input is `record.mat`: `feats` has shape
   `(12, samples)` in ECG-Agent lead order and `curr_sample_rate` is 500 Hz.
   Single-lead records keep only the named waveform channel and include
   `measurement_lead` for the measurement tool. Standard records default to Lead II.
2. Create `toolbox/rhythm/tool.py`:

````python
from ..base import Tool

TOOL = Tool(
    files=("rhythm/__init__.py", "rhythm/analyze.py"),
    prompt="""To inspect rhythm:
```python-run
from toolbox.rhythm.analyze import analyze
print(analyze('record.mat'))
```""",
)
````

3. Import and register it in `toolbox/registry.py`:

```python
from .rhythm.tool import TOOL as rhythm

TOOLS = {"measurement": measurement, "classification": classification, "rhythm": rhythm}
```

Select it with `--tools measurement classification rhythm`.

`files` lists paths relative to `toolbox/`. Directories are copied recursively;
list individual files to keep host code out of the agent workspace.
Only selected tools are copied into the temporary runtime and started. Their
examples reach the model automatically. Existing workspace code and installed
dependencies remain available; tool selection is not a security boundary.

## Host services

For checkpoints or private services, subclass `Tool` as in
`neural_networks/classification/tool.py`:

- `add_arguments(parser)`: register CLI options; JSON uses their underscore names.
- `load(args)`: load host state once, only when enabled. Validate required options here.
- `start(state, resources, workspace, runtime, *, sandbox)`: start each workspace's
  services, register cleanup with `resources.callback(...)`, and return agent
  environment variables. Put sockets under `runtime`; use `/run/eko/<socket>`
  in sandbox environment values and the actual runtime path locally.

Keep checkpoints, credentials, definitions, and service code outside `files`.
Package initializers and definitions must defer heavy dependency imports so
disabled tools do not load them. Examples use `record.mat`; regular Eko substitutes
the user's ECG path. Benchmark input remains `record.mat`.
