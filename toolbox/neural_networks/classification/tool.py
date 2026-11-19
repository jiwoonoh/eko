"""Host-resident ECG-Agent classifier."""

from pathlib import Path

from ...base import Tool


class Classification(Tool):
    def add_arguments(self, parser):
        parser.add_argument("--classifier-checkpoint", type=Path,
                            help="Fine-tuned 71-label checkpoint; required for classification")
        parser.add_argument("--classifier-device")

    def load(self, args):
        if args.classifier_checkpoint is None:
            raise ValueError("--classifier-checkpoint is required when classification is enabled")
        from .model import Classifier

        return Classifier(args.classifier_checkpoint, args.classifier_device)

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
    prompt="""To identify arrhythmias, abnormalities, and other ECG findings using all 12 leads:
```python-run
from toolbox.neural_networks.classification import ecg_classifier
print(ecg_classifier('record.mat'))
```""",
)
