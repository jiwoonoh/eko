"""One tool's agent files, instructions, and host lifecycle."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    files: tuple[str, ...]
    prompt: str

    @classmethod
    def from_module(cls, module, description, *, files=()):
        """Expose a function named after its module that takes an ECG path."""
        function = module.rsplit(".", 1)[-1]
        return cls(
            files=(module.replace(".", "/") + ".py", *files),
            prompt=f"""{description}:
```python-run
from toolbox.{module} import {function}
print({function}('record.mat'))
```""",
        )

    def prompt_for(self, *, leads):
        """Render usage instructions for the available waveform leads."""
        return self.prompt

    def add_arguments(self, parser):
        """Register optional CLI/config settings."""

    def load(self, args):
        """Load host state once per run."""

    def start(self, state, resources, workspace, runtime, *, sandbox):
        """Start workspace services; return agent environment variables."""
        return {}
