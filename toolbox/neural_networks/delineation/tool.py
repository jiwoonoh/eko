"""Host-resident R-U-Net delineator."""

from pathlib import Path

from ...base import Tool
from ...leads import measurement_lead


class Delineation(Tool):
    def prompt_for(self, *, leads):
        return self.prompt.format(lead=measurement_lead(leads))

    def add_arguments(self, parser):
        parser.add_argument("--delineator-checkpoint", type=Path,
                            help="R-U-Net checkpoint; required for delineation")
        parser.add_argument("--delineator-device")

    def load(self, args):
        if args.delineator_checkpoint is None:
            raise ValueError("--delineator-checkpoint is required when delineation is enabled")
        from .model import Delineator

        return Delineator(args.delineator_checkpoint, args.delineator_device)

    def start(self, state, resources, workspace, runtime, *, sandbox):
        from .service import DelineatorServer

        service = DelineatorServer(runtime / "delineator.sock", state, workspace, sandbox=sandbox)
        service.start()
        resources.callback(service.close)
        endpoint = Path("/run/eko") if sandbox else runtime
        return {"EKO_DELINEATOR": str(endpoint / "delineator.sock")}


TOOL = Delineation(
    files=("leads.py", "neural_networks/delineation/__init__.py",
           "neural_networks/delineation/ecg_delineator.py"),
    prompt="""To locate P, QRS, and T wave onsets and offsets in lead {lead}.
Each wave is an inclusive [onset, offset] pair of sample indices at sampling_rate, so its
duration is offset - onset + 1 samples. Waves cut off by the recording's start or end
are omitted, so P, QRS, and T lists may not align by position:
```python-run
from toolbox.neural_networks.delineation import ecg_delineator
print(ecg_delineator('record.mat', lead='{lead}'))
```""",
)
