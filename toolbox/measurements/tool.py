"""ECG-Agent measurements."""

from ..base import Tool


class Measurement(Tool):
    def prompt_for(self, *, single_lead=None):
        return self.prompt.format(lead=single_lead or "II")


TOOL = Measurement(
    files=("leads.py", "measurements/__init__.py", "measurements/ecg.py", "measurements/ecg_analysis.py",
           "measurements/heart_rate.py", "measurements/pr_interval.py",
           "measurements/qrs_duration.py", "measurements/qtc_interval.py"),
    prompt="""To measure heart rate (bpm), PR interval, QRS duration, and QTc interval (ms), using lead {lead}:
```python-run
from toolbox.measurements.ecg_analysis import ecg_analysis
print(ecg_analysis('record.mat'))
```""",
)
