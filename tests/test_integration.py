import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from contracts import STANDARD_LEADS
from eko import Session, run_turn
from providers import OpenRouter
from records import load_record
from traces import EventLog
from worker import ToolWorker

SCIENCE_AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "scipy", "neurokit2"))


@unittest.skipUnless(SCIENCE_AVAILABLE, "install the ECG extra")
class IntegrationTests(unittest.TestCase):
    def test_native_model_loop_measures_waveform_then_reuses_evidence(self):
        import neurokit2 as nk
        import numpy as np
        from scipy.io import savemat

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "HR25.mat"
            lead = nk.ecg_simulate(duration=10, sampling_rate=250, heart_rate=80, random_state=42)
            savemat(path, {"feats": np.tile(lead, (12, 1)), "curr_sample_rate": 250})
            record, _ = load_record(path, STANDARD_LEADS)
            model = OpenRouter("fixture-model", api_key="fixture-key")
            requests = []

            def response(body, deadline):
                payload = json.loads(body)
                requests.append(payload)
                observations = [json.loads(message["content"]) for message in payload["messages"] if message["role"] == "tool"]
                if not observations:
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "measurement-1", "type": "function", "function": {
                            "name": "call_measurement_tool",
                            "arguments": json.dumps({"record_id": record.record_id}),
                        },
                    }]}
                    finish = "tool_calls"
                else:
                    observation = observations[-1]
                    message = {"role": "assistant", "content": json.dumps({
                        "findings": [{"evidence_id": observation["evidence_id"], "finding": "heart_rate_bpm"}],
                        "explanations": [], "unavailable": [],
                    })}
                    finish = "stop"
                return {"model": "fixture-model", "choices": [{"finish_reason": finish, "message": message}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

            with EventLog(Path(directory) / "runs", record, {"model": "fixture-model"}) as log:
                session = Session(model, ToolWorker(), record, log=log)
                self.addCleanup(session.close)
                with mock.patch.object(model, "_request", side_effect=response):
                    first = run_turn(session, "Measure the heart rate.")
                    self.assertEqual(first.status, "success", first.error)
                    self.assertEqual(first.tool_calls, 1)
                    self.assertAlmostEqual(first.evidence[0].findings["heart_rate_bpm"], 80, delta=5)
                    pid = session.worker.process.pid
                    second = run_turn(session, "What was the heart rate again?")
                    self.assertEqual(second.status, "success", second.error)
                    self.assertEqual(second.tool_calls, 0)
                    self.assertEqual(second.evidence[0].evidence_id, first.evidence[0].evidence_id)
                    self.assertEqual(session.worker.process.pid, pid)
                session.close()
            self.assertEqual(len(requests), 3)
            serialized = json.dumps(requests)
            self.assertNotIn("fixture-key", serialized)
            self.assertNotIn(str(path), serialized)
            events = [json.loads(line) for line in log.path.read_text().splitlines()]
            self.assertEqual(sum(event["type"] == "tool_result" for event in events), 1)
            self.assertEqual(sum(event["type"] == "turn_end" for event in events), 2)
            self.assertNotIn("fixture-key", log.path.read_text())


if __name__ == "__main__":
    unittest.main()
