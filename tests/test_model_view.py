"""Compact model messages preserve evidence while avoiding repeated metadata."""

from copy import deepcopy
from dataclasses import replace
import json
import unittest

from reporting import model_observation
from tests.test_classification_runtime import classification, ECG
from contracts import ToolObservation


class ModelViewTests(unittest.TestCase):
    def test_all_scores_labels_and_record_binding_survive_compaction(self):
        raw = classification()
        observation = ToolObservation(ECG.record_id, ECG.sha256, "call", "ev",
                                      "call_classification_tool", raw["version"], raw["status"],
                                      raw["findings"], raw["units"], raw["methods"], raw["missingness"], raw["details"])
        before = deepcopy(observation.to_dict())
        view = model_observation(observation)
        self.assertEqual(view["findings"], before["findings"])
        self.assertEqual(len(view["labels"]), 150)
        self.assertEqual(view["labels"]["ecgfounder_003"], "SINUS RHYTHM")
        self.assertEqual(view["labels"]["ecgfounder_001"], "NORMAL SINUS RHYTHM")
        for name in ("evidence_id", "record_id", "record_hash", "status", "source", "tool", "version"):
            self.assertEqual(view[name], before[name])
        self.assertEqual(observation.to_dict(), before)
        self.assertLess(len(json.dumps(view)), len(json.dumps(before)) * .75)

    def test_partial_measurements_keep_nulls_methods_and_filter_exclusions(self):
        observation = ToolObservation(ECG.record_id, ECG.sha256, "call", "ev",
                                      "call_measurement_tool", "v1", "partial",
                                      {"heart_rate_bpm": 70, "pr_interval_ms": None},
                                      {"heart_rate_bpm": "bpm", "pr_interval_ms": "ms"},
                                      {"heart_rate_bpm": "median RR", "pr_interval_ms": "onsets"},
                                      {"pr_interval_ms": "no retained estimates"},
                                      {"per_beat": {"private_detail": [1, 2]},
                                       "filtering": {"pr_interval_ms": {"retained": 0, "excluded": 8}},
                                       "lead_used": "II", "sampling_rate_hz": 500})
        view = model_observation(observation)
        for name in ("findings", "units", "methods", "missingness"):
            self.assertEqual(view[name], observation.to_dict()[name])
        self.assertEqual(view["details"]["filtering"], observation.details["filtering"])
        self.assertNotIn("per_beat", view["details"])
        self.assertEqual(observation.details["per_beat"], {"private_detail": [1, 2]})
