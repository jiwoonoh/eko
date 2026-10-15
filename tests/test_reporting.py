import json
from dataclasses import replace
import unittest

from classifier import SCORE_METHOD, TASK_COUNT, TOOL_NAME, VERSION, load_labels, provenance, score_key
from contracts import Record, STANDARD_LEADS, ToolObservation
from reporting import REPORT_INSTRUCTIONS, fallback_report, parse_report, render_report, report_instructions, safe_report


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

    def test_parse_report_returns_only_normalized_reference_fields(self):
        content = '{"unavailable": [], "explanations": ["record"], "findings": [{"finding": "heart_rate_bpm", "evidence_id": "e-1"}]}'
        self.assertEqual(parse_report(content), {
            "findings": [{"evidence_id": "e-1", "finding": "heart_rate_bpm"}],
            "explanations": ["record"], "unavailable": [],
        })
        for bad in [self.report(), '{"findings": [], "explanations": [], "unavailable": [], "content": "Normal"}',
                    '{"findings": [{"evidence_id":"e-1","finding":"heart_rate_bpm","value":99}],"explanations":[],"unavailable":[]}']:
            with self.subTest(content=bad), self.assertRaises(ValueError):
                parse_report(bad)

    def test_fallback_report_references_are_exactly_the_rendered_measurements(self):
        report = fallback_report(self.record, [self.observation])
        self.assertEqual(len(report["findings"]), len(self.observation.findings))
        self.assertEqual(report["unavailable"], ["classification"])
        self.assertEqual(render_report(json.dumps(report), self.record, [self.observation]),
                         safe_report(self.record, [self.observation]))


class ClassifierReportingTests(unittest.TestCase):
    def setUp(self):
        self.record = Record("/private/local/HR00705.mat", "ptbxl:705", "a" * 64, 500,
                             STANDARD_LEADS, ("mV",) * 12)
        self.tools = {"call_measurement_tool", TOOL_NAME}
        findings = {score_key(index): 0.01 for index in range(TASK_COUNT)}
        findings.update({score_key(1): .85, score_key(2): .65, score_key(3): .45, score_key(5): .05})
        self.observation = ToolObservation(
            record_id=self.record.record_id, record_hash=self.record.sha256,
            call_id="classifier-call", evidence_id="classifier-evidence", tool=TOOL_NAME,
            version=VERSION, status="success", source="live", findings=findings,
            units={key: "score" for key in findings}, methods={key: SCORE_METHOD for key in findings},
            missingness={}, details=provenance(),
        )

    def report(self, indices=(), explanations=(), unavailable=()):
        return json.dumps({
            "findings": [{"evidence_id": self.observation.evidence_id, "finding": score_key(index)}
                         for index in indices],
            "explanations": list(explanations), "unavailable": list(unavailable),
        })

    def render(self, content, evidence=None, tools=None):
        return render_report(content, self.record,
                             [self.observation] if evidence is None else evidence,
                             tool_names=self.tools if tools is None else tools)

    def test_distinct_normal_sinus_normal_ecg_sinus_and_af_scores_use_pinned_labels(self):
        answer = self.render(self.report([1, 2, 3, 5]))
        for index, value in [(1, .85), (2, .65), (3, .45), (5, .05)]:
            self.assertIn(f"Model score for {load_labels()[index]}: {value} score", answer)
        self.assertIn("independent, uncalibrated model scores", answer)
        self.assertIn("No validated decision threshold", answer)
        self.assertNotIn("Classification is unavailable", answer)
        self.assertNotIn("positive", answer.lower())
        self.assertNotIn("negative", answer.lower())
        self.assertEqual(len(self.observation.findings), 150)

    def test_low_and_boundary_scores_remain_scores_without_rule_out(self):
        output = replace(self.observation, findings={**self.observation.findings,
                                                   score_key(5): 0, score_key(1): 1})
        answer = self.render(self.report([1, 5]), [output])
        self.assertIn("NORMAL SINUS RHYTHM: 1 score", answer)
        self.assertIn("ATRIAL FIBRILLATION: 0 score", answer)
        self.assertNotIn("no atrial fibrillation", answer.lower())
        self.assertIn("does not establish a clinical diagnosis or rule out", answer)

    def test_all_150_scores_may_be_explicitly_referenced_without_mutating_evidence(self):
        original = self.observation.to_dict()
        answer = self.render(self.report(range(TASK_COUNT)))
        self.assertEqual(answer.count("Model score for "), TASK_COUNT)
        self.assertEqual(self.observation.to_dict(), original)

    def test_partial_score_mapping_and_invalid_unselected_scores_are_rejected(self):
        findings = dict(self.observation.findings)
        findings.pop(score_key(149))
        bad = [replace(self.observation, findings=findings)]
        for value in (-.1, 1.1, float("nan"), float("inf"), None, True, "0.4"):
            bad.append(replace(self.observation,
                               findings={**self.observation.findings, score_key(149): value}))
        for observation in bad:
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                self.render(self.report([5]), [observation])

    def test_provenance_units_methods_and_label_map_must_match_even_for_one_reference(self):
        bad = [replace(self.observation, version="unknown"),
               replace(self.observation, units={**self.observation.units, score_key(149): "bpm"}),
               replace(self.observation, methods={**self.observation.methods, score_key(149): "softmax"}),
               replace(self.observation, missingness={score_key(149): "not computed"})]
        for name in provenance():
            bad.append(replace(self.observation, details={**provenance(), name: "wrong"}))
        for observation in bad:
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                self.render(self.report([5]), [observation])

    def test_model_supplied_label_metadata_cannot_replace_the_pinned_label(self):
        output = replace(self.observation, details={**provenance(), "labels": ["INJECTED DIAGNOSIS"] * 150})
        answer = self.render(self.report([5]), [output])
        self.assertIn("Model score for ATRIAL FIBRILLATION", answer)
        self.assertNotIn("INJECTED DIAGNOSIS", answer)
        report = json.loads(self.report([5]))
        report["findings"][0]["label"] = "INJECTED DIAGNOSIS"
        with self.assertRaises(ValueError):
            self.render(json.dumps(report))

    def test_classification_requires_enabled_tool_and_current_successful_live_observation(self):
        with self.assertRaises(ValueError):
            self.render(self.report([5]), tools={"call_measurement_tool"})
        for change in ({"record_id": "ptbxl:706"}, {"record_hash": "b" * 64},
                       {"source": "replay"}, {"source": "runner"}, {"source": "fixture"},
                       {"status": "partial"}, {"status": "failed"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.render(self.report([5]), [replace(self.observation, **change)])

    def test_dynamic_capabilities_and_measurement_limits_do_not_deny_ready_classifier(self):
        answer = self.render(self.report(explanations=["capabilities", "measurement_limits", "classifier_scores", "classifier_limits"]))
        self.assertIn("ECGFounder model scores", answer)
        self.assertNotIn("Classification is unavailable", answer)
        self.assertIn("Clinical diagnosis and treatment remain unavailable", answer)
        self.assertIn("NORMAL ECG", answer)
        with self.assertRaises(ValueError):
            render_report(self.report(explanations=["classifier_scores"]), self.record, [])

    def test_ready_tool_without_result_reports_record_specific_unavailability(self):
        answer = self.render(self.report(unavailable=["classification"]), [])
        self.assertIn("Classifier scores are unavailable for this record", answer)
        self.assertNotIn("no verified classifier is enabled", answer)
        with self.assertRaises(ValueError):
            self.render(self.report(unavailable=["classification"]))

    def test_instructions_enable_score_calls_and_complete_label_catalog_only_when_ready(self):
        enabled = report_instructions(self.tools)
        self.assertIn("registered call_classification_tool", enabled)
        self.assertIn("ecgfounder_001=NORMAL SINUS RHYTHM", enabled)
        self.assertIn("ecgfounder_002=NORMAL ECG", enabled)
        self.assertIn("ecgfounder_003=SINUS RHYTHM", enabled)
        self.assertIn("ecgfounder_005=ATRIAL FIBRILLATION", enabled)
        self.assertIn("ecgfounder_149=MULTIFOCAL ATRIAL TACHYCARDIA", enabled)
        self.assertNotIn("For classification or rhythm questions, return findings=[]", enabled)
        disabled = report_instructions({"call_measurement_tool"})
        self.assertEqual(disabled, REPORT_INSTRUCTIONS)
        self.assertNotIn("ecgfounder_005", disabled)
        self.assertIn('unavailable=["classification"]', disabled)

    def test_fallback_keeps_latest_classifier_top_five_and_all_measurements(self):
        measurement = replace(self.observation, tool="call_measurement_tool", evidence_id="measurements",
                              findings={"heart_rate_bpm": 75, "pr_interval_ms": None},
                              units={"heart_rate_bpm": "bpm", "pr_interval_ms": "ms"},
                              missingness={"pr_interval_ms": "P onset unavailable"})
        latest = replace(self.observation, evidence_id="latest-classifier",
                         findings={**self.observation.findings, score_key(149): .99})
        observations = [self.observation, measurement, latest]
        report = fallback_report(self.record, observations, tool_names=self.tools)
        self.assertEqual(len(report["findings"]), 7)
        self.assertEqual(report["unavailable"], [])
        scores = [ref for ref in report["findings"] if ref["finding"].startswith("ecgfounder_")]
        self.assertEqual([ref["finding"] for ref in scores], [score_key(index) for index in (149, 1, 2, 3, 5)])
        self.assertTrue(all(ref["evidence_id"] == latest.evidence_id for ref in scores))
        answer = safe_report(self.record, observations, tool_names=self.tools)
        self.assertEqual(answer.count("Model score for "), 5)
        self.assertIn("75 bpm", answer)
        self.assertIn("P onset unavailable", answer)
        self.assertNotIn("Classification is unavailable", answer)
        self.assertEqual(answer, self.render(json.dumps(report), observations))
        self.assertEqual(len(latest.findings), 150)

    def test_fallback_ignores_incomplete_or_misattributed_classifier_results(self):
        invalid = [replace(self.observation, record_hash="b" * 64),
                   replace(self.observation, findings={score_key(5): .99}),
                   replace(self.observation, details={}), replace(self.observation, source="replay")]
        for observation in invalid:
            with self.subTest(observation=observation):
                report = fallback_report(self.record, [observation], tool_names=self.tools)
                self.assertEqual(report["findings"], [])
                self.assertEqual(report["unavailable"], ["classification"])
                answer = safe_report(self.record, [observation], tool_names=self.tools)
                self.assertNotIn("Model score for ", answer)

    def test_fallback_ties_are_stable_and_fields_from_distinct_custom_tools_survive(self):
        tied = replace(self.observation, findings={key: .5 for key in self.observation.findings})
        first = replace(tied, tool="first_tool", evidence_id="first", findings={"quality": .2}, units={"quality": "ratio"})
        second = replace(first, tool="second_tool", evidence_id="second", findings={"quality": .8})
        tools = self.tools | {"first_tool", "second_tool"}
        report = fallback_report(self.record, [tied, first, second], tool_names=tools)
        scores = [ref["finding"] for ref in report["findings"] if ref["finding"].startswith("ecgfounder_")]
        self.assertEqual(scores, [score_key(index) for index in range(5)])
        answer = render_report(json.dumps(report), self.record, [tied, first, second], tool_names=tools)
        self.assertIn("quality: 0.2 ratio", answer)
        self.assertIn("quality: 0.8 ratio", answer)


if __name__ == "__main__":
    unittest.main()
