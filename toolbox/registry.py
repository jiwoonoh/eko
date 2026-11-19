"""Shared tool registration and setup. Keep tool dependencies lazy."""

from pathlib import Path
import shutil

from .measurements.tool import TOOL as measurement
from .neural_networks.classification.tool import TOOL as classification

TOOLS = {"measurement": measurement, "classification": classification}
DEFAULT_TOOLS = ("measurement", "classification")
ROOT = Path(__file__).resolve().parent


def add_arguments(parser):
    parser.add_argument("--tools", nargs="*", choices=TOOLS, default=list(DEFAULT_TOOLS),
                        help=f"Enabled tools (default: {' '.join(DEFAULT_TOOLS)}); --tools disables all")
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
        self._tools = {name: TOOLS[name] for name in names}
        self._states = {name: tool.load(args) for name, tool in self._tools.items()}

    @property
    def prompt(self):
        return "\n\n".join(tool.prompt for tool in self._tools.values())

    def start(self, resources, workspace, runtime, *, sandbox):
        _stage(self._tools, runtime)
        environment = {}
        for name, tool in self._tools.items():
            values = tool.start(self._states[name], resources, workspace, runtime, sandbox=sandbox)
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
    for tool in tools.values():
        for name in tool.files:
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
