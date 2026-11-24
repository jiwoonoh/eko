"""Shared tool registration and setup. Keep tool dependencies lazy."""

from pathlib import Path
import shutil

from .leads import select_leads
from .measurements.tool import BENCHMARK_MEASUREMENT, measurement_tool
from .neural_networks.classification.tool import TOOL as classification
from .neural_networks.delineation.tool import TOOL as delineation

TOOLS = {
    "measurement": BENCHMARK_MEASUREMENT,
    "classification": classification,
    "delineation": delineation,
    "heart_rate": measurement_tool("heart_rate", "heart rate (bpm)"),
    "pr_interval": measurement_tool("pr_interval", "PR interval (ms)"),
    "qrs_duration": measurement_tool("qrs_duration", "QRS duration (ms)"),
    "qtc_interval": measurement_tool("qtc_interval", "QTc interval (ms)"),
}
DEFAULT_TOOLS = ("measurement", "classification")
ROOT = Path(__file__).resolve().parent


def add_arguments(parser):
    parser.add_argument("--tools", nargs="*", choices=TOOLS, default=list(DEFAULT_TOOLS),
                        help=f"Default: {' '.join(DEFAULT_TOOLS)}. Use --tools alone to disable all.")
    for tool in TOOLS.values():
        tool.add_arguments(parser)


class Toolbox:
    def __init__(self, args):
        names = args.tools
        if not isinstance(names, list) or any(
                not isinstance(name, str) or name not in TOOLS for name in names):
            raise ValueError(f"--tools must be a list drawn from: {', '.join(TOOLS)}")
        if len(names) != len(set(names)):
            raise ValueError("--tools must not contain duplicates")
        leads = select_leads(getattr(args, "leads", None))
        selected = [TOOLS[name] for name in names]
        self._tools = [(tool, tool.load(args)) for tool in selected]
        self.prompt = "\n\n".join(tool.prompt_for(leads=leads) for tool in selected)

    def start(self, resources, workspace, runtime, *, sandbox):
        _stage((tool for tool, _ in self._tools), runtime)
        environment = {}
        for tool, state in self._tools:
            values = tool.start(state, resources, workspace, runtime, sandbox=sandbox)
            overlap = environment.keys() & values.keys()
            if overlap:
                raise ValueError(f"Tools share environment variables: {', '.join(sorted(overlap))}")
            environment.update(values)
        return environment


def _stage(tools, directory):
    """Copy only selected agent code; keep host services outside."""
    target = directory / "toolbox"
    target.mkdir()
    (target / "__init__.py").touch()
    files = dict.fromkeys(name for tool in tools for name in tool.files)
    for name in files:
        source = (ROOT / name).resolve()
        destination = target / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, destination)
        for parent in destination.relative_to(target).parents:
            (target / parent / "__init__.py").touch()
