"""Register individual measurements separately from the fixed ECG-Agent panel."""

from ..base import Tool
from .ecg_analysis import PANEL


class Measurement(Tool):
    def prompt_for(self, *, single_lead=None):
        return self.prompt.format(lead=single_lead or "II")


def measurement_tool(name, quantity, *, files=()):
    return Measurement.from_module(
        f"measurements.{name}", f"To measure {quantity}, using lead {{lead}}",
        files=("record.py", "leads.py", "measurements/preprocessing.py", *files),
    )


BENCHMARK_MEASUREMENT = measurement_tool(
    "ecg_analysis", "heart rate (bpm), PR interval, QRS duration, and QTc interval (ms)",
    files=("measurements/__init__.py", *(f"measurements/{name}.py" for name in PANEL)),
)
