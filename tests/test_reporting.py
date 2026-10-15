import json
from dataclasses import replace
import unittest

from contracts import Record, STANDARD_LEADS, ToolObservation
from reporting import REPORT_INSTRUCTIONS, render_report, safe_report


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.record = Record("/private/local/HR00705.mat", "ptbxl:705", "a" * 64, 500,
                             STANDARD_LEADS, ("mV",) * 12)
        self.observation = ToolObservation(
            record_id=self.record.record_id,
            record_hash=self.record.sha256,
            call_id="call-1",
            evidence_id="e-1",
            tool="call_measurement_tool",
            version="ecg-measurements-v1",
            status="success",
            findings={"heart_rate_bpm": 75.95, "rr_interval_ms": 790.0, "pr_interval_ms": 116.0,
                      "q_s_peak_span_ms": 93.0, "qtc_bazett_ms": 432.53},
            units={"heart_rate_bpm": "bpm", "rr_interval_ms": "ms", "pr_interval_ms": "ms",
                   "q_s_peak_span_ms": "ms", "qtc_bazett_ms": "ms"},
            methods={"q_s_peak_span_ms": "Q-peak to S-peak; not onset-to-offset QRS duration",
                     "qtc_bazett_ms": "Bazett correction using the preceding RR interval"},
            details={"sampling_rate_hz": 500, "lead_used": "II", "waveform_units": "mV"},
        )

    def report(self, findings=(), explanations=(), unavailable=()):
        return json.dumps({"findings": [{"evidence_id": "e-1", "finding": field} for field in findings],
                           "explanations": list(explanations), "unavailable": list(unavailable)})

    def render(self, content, evidence=None):
        return render_report(content, self.record, (self.observation,) if evidence is None else evidence)

    def test_numbers_are_sourced_from_requested_observations(self):
        output = self.render(self.report(["heart_rate_bpm", "pr_interval_ms"]))
        self.assertIn("75.95", output)
        self.assertIn("116", output)
        self.assertIn("bpm", output.lower())
        self.assertIn("ms", output)
        self.assertNotIn("432.53", output)
        self.assertNotRegex(output.lower(), r"within (?:the |a )?normal range")
        self.assertNotRegex(output.lower(), r"(?:^|\n)(?:your|the) ecg is normal\b")

    def test_peak_span_and_qtc_limits_are_explicit(self):
        output = self.render(self.report(["q_s_peak_span_ms", "qtc_bazett_ms"], ["measurement_limits"]))
        self.assertIn("93", output)
        self.assertIn("432.53", output)
        self.assertRegex(output.lower(), r"q[ /-]+(?:to[- ]+)?s|q.peak.*s.peak")
        self.assertRegex(output.lower(), r"not (?:full qrs|an? onset.to.offset qrs)")
        self.assertIn("bazett", output.lower())
        self.assertNotRegex(output.lower(), r"within (?:the |a )?normal range")

    def test_record_explanation_uses_current_public_metadata(self):
        output = self.render(self.report(explanations=["record"]))
        self.assertIn("ptbxl:705", output)
        self.assertIn("500", output)
        self.assertIn("hz", output.lower())
        self.assertNotIn("/private/local", output)

    def test_null_has_missingness_reason_and_never_becomes_zero(self):
        partial = replace(self.observation, status="partial",
                          findings={**self.observation.findings, "pr_interval_ms": None},
                          missingness={"pr_interval_ms": "missing P onset"})
        output = self.render(self.report(["pr_interval_ms"]), (partial,))
        self.assertIn("missing P onset", output)
        self.assertNotRegex(output, r"\b0(?:\.0+)?\s*ms\b")
        self.assertNotIn("116", output)
        self.assertRegex(output.lower(), r"unavailable|not available|not measured|missing")

    def test_plain_text_diagnosis_and_extra_fields_are_rejected(self):
        bad = ["Your ECG is normal with a heart rate of 99 bpm.", json.dumps("You have atrial fibrillation."),
               json.dumps({"findings": [], "explanations": [], "unavailable": [], "diagnosis": "Normal"}),
               json.dumps({"findings": [{"evidence_id": "e-1", "finding": "heart_rate_bpm", "value": 99}],
                           "explanations": [], "unavailable": []})]
        for content in bad:
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.render(content)

    def test_all_three_keys_and_correct_container_types_are_required(self):
        valid = json.loads(self.report(["heart_rate_bpm"]))
        for name in valid:
            missing = {key: value for key, value in valid.items() if key != name}
            with self.subTest(missing=name), self.assertRaises(ValueError):
                self.render(json.dumps(missing))
            invalid = {**valid, name: "free prose"}
            with self.subTest(invalid=name), self.assertRaises(ValueError):
                self.render(json.dumps(invalid))

    def test_reference_must_exist_and_name_a_supported_finding(self):
        for reference in [{"evidence_id": "unknown", "finding": "heart_rate_bpm"},
                          {"evidence_id": "e-1", "finding": "diagnosis"},
                          {"evidence_id": "e-1", "finding": "blood_pressure"},
                          {"evidence_id": "e-1"}, {"finding": "heart_rate_bpm"}]:
            content = json.dumps({"findings": [reference], "explanations": [], "unavailable": []})
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                self.render(content)
        empty = replace(self.observation, findings={})
        with self.assertRaises(ValueError):
            self.render(self.report(["heart_rate_bpm"]), (empty,))

    def test_reference_rejects_other_record_hash_failed_replay_and_tool(self):
        bad = [replace(self.observation, record_id="ptbxl:706"),
               replace(self.observation, record_hash="b" * 64),
               replace(self.observation, status="failed"),
               replace(self.observation, source="replay"),
               replace(self.observation, tool="unregistered_tool")]
        for observation in bad:
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                self.render(self.report(["heart_rate_bpm"]), (observation,))

    def test_evidence_id_collisions_and_duplicate_references_are_rejected(self):
        collision = replace(self.observation, call_id="call-2",
                            findings={**self.observation.findings, "heart_rate_bpm": 99.0})
        with self.assertRaises(ValueError):
            self.render(self.report(["heart_rate_bpm"]), (self.observation, collision))
        with self.assertRaises(ValueError):
            self.render(self.report(["heart_rate_bpm", "heart_rate_bpm"]))
        for name, values in [("explanations", ["record", "record"]),
                             ("unavailable", ["classification", "classification"])]:
            content = {"findings": [], "explanations": [], "unavailable": [], name: values}
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.render(json.dumps(content))
        duplicate_keys = '{"findings": [], "findings": [], "explanations": ["record"], "unavailable": []}'
        with self.assertRaises(ValueError):
            self.render(duplicate_keys)

    def test_nonfinite_or_nonnumeric_observation_values_are_rejected(self):
        for value in [float("nan"), float("inf"), float("-inf"), True, "75.95"]:
            observation = replace(self.observation, findings={"heart_rate_bpm": value})
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.render(self.report(["heart_rate_bpm"]), (observation,))

    def test_unsupported_explanation_or_unavailable_codes_are_rejected(self):
        for name, value in [("explanations", "Your ECG is normal."), ("explanations", "diagnosis"),
                            ("unavailable", "take medication"), ("unavailable", "heart_rate_bpm")]:
            content = {"findings": [], "explanations": [], "unavailable": [], name: [value]}
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.render(json.dumps(content))

    def test_supported_explanation_and_unavailability_codes_render(self):
        explanations = ["capabilities", "record", "measurement_limits", "heart_rate_bpm", "rr_interval_ms",
                        "pr_interval_ms", "q_s_peak_span_ms", "qtc_bazett_ms"]
        unavailable = ["classification", "diagnosis", "normality", "treatment", "unregistered_tool"]
        output = self.render(self.report(explanations=explanations, unavailable=unavailable))
        self.assertTrue(output.strip())
        self.assertIn("classification", output.lower())
        self.assertIn("bazett", output.lower())

    def test_numeric_custom_tool_requires_explicit_registered_tool_name(self):
        observation = replace(self.observation, tool="quality_tool", source="fixture",
                              findings={"quality_index": .875}, units={"quality_index": "ratio"})
        content = self.report(["quality_index"])
        with self.assertRaises(ValueError):
            self.render(content, (observation,))
        output = render_report(content, self.record, (observation,), tool_names={"quality_tool"})
        self.assertIn("0.875", output)
        self.assertIn("ratio", output)

    def test_safe_report_uses_latest_live_results_and_unavailable_classifier(self):
        latest = replace(self.observation, evidence_id="e-2", call_id="call-2",
                         findings={**self.observation.findings, "heart_rate_bpm": 81.25})
        output = safe_report(self.record, (self.observation, latest))
        self.assertIn("81.25", output)
        self.assertNotIn("75.95", output)
        self.assertIn("790", output)
        self.assertIn("116", output)
        self.assertIn("93", output)
        self.assertIn("432.53", output)
        self.assertIn("classification", output.lower())
        self.assertEqual(output, safe_report(self.record, (self.observation, latest)))

    def test_safe_report_does_not_replace_latest_missing_value_with_old_number(self):
        latest = replace(self.observation, evidence_id="e-2", call_id="call-2", status="partial",
                         findings={**self.observation.findings, "pr_interval_ms": None},
                         missingness={"pr_interval_ms": "missing P onset"})
        output = safe_report(self.record, (self.observation, latest))
        self.assertIn("missing P onset", output)
        self.assertNotIn("116", output)

    def test_safe_report_ignores_wrong_record_hash_replay_and_failed_evidence(self):
        invalid = [replace(self.observation, evidence_id="wrong-record", record_id="ptbxl:706"),
                   replace(self.observation, evidence_id="wrong-hash", record_hash="b" * 64),
                   replace(self.observation, evidence_id="replay", source="replay"),
                   replace(self.observation, evidence_id="failed", status="failed")]
        for observation in invalid:
            with self.subTest(evidence_id=observation.evidence_id):
                output = safe_report(self.record, (observation,))
                self.assertNotIn("75.95", output)
                self.assertNotIn("432.53", output)
                self.assertIn("classification", output.lower())

    def test_report_instructions_describe_json_references_and_no_free_prose(self):
        self.assertIsInstance(REPORT_INSTRUCTIONS, str)
        for key in ["findings", "evidence_id", "explanations", "unavailable"]:
            self.assertIn(key, REPORT_INSTRUCTIONS)
        self.assertTrue(REPORT_INSTRUCTIONS.strip())


if __name__ == "__main__":
    unittest.main()
