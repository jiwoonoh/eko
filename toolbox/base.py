"""One tool's agent files, instructions, and host lifecycle."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    files: tuple[str, ...]
    prompt: str

    def prompt_for(self, *, single_lead=None):
        """Render usage instructions for the available waveform leads."""
        return self.prompt

    def add_arguments(self, parser):
        """Register optional CLI/config settings."""

    def load(self, args):
        """Load host state once per run."""

    def start(self, state, resources, workspace, runtime, *, sandbox):
        """Start workspace services; return agent environment variables."""
        return {}
