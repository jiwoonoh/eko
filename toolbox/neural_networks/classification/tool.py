"""Host-resident ECG-Agent classifier."""

from pathlib import Path

from ...base import Tool
from ...leads import SINGLE_LEADS


class Classification(Tool):
    def prompt_for(self, *, single_lead=None):
        return self.prompt.format(leads=f"lead {single_lead}" if single_lead else "all 12 leads")

    def add_arguments(self, parser):
        parser.add_argument("--classifier-checkpoint", type=Path,
                            help="Fine-tuned 71-label checkpoint; required for classification")
        parser.add_argument("--classifier-device")

    def load(self, args):
        if args.classifier_checkpoint is None:
            raise ValueError("--classifier-checkpoint is required when classification is enabled")
        from .model import Classifier

        single_lead = SINGLE_LEADS.get(getattr(args, "leads", None))
        return Classifier(args.classifier_checkpoint, args.classifier_device, single_lead=single_lead)

    def start(self, state, resources, workspace, runtime, *, sandbox):
        from .service import ClassifierServer

        service = ClassifierServer(runtime / "classifier.sock", state, workspace, sandbox=sandbox)
        service.start()
        resources.callback(service.close)
        endpoint = Path("/run/eko") if sandbox else runtime
        return {"EKO_CLASSIFIER": str(endpoint / "classifier.sock")}


TOOL = Classification(
    files=("neural_networks/classification/__init__.py",
           "neural_networks/classification/ecg_classifier.py"),
    prompt="""To identify arrhythmias, abnormalities, and other ECG findings using {leads}:
```python-run
from toolbox.neural_networks.classification import ecg_classifier
print(ecg_classifier('record.mat'))
```""",
)
