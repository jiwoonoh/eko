"""Synthetic converter boundaries and benchmark label/evidence separation."""

from dataclasses import replace
from io import StringIO
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import benchmark
import classifier
from contracts import Record, STANDARD_LEADS, ToolObservation
from measurements import METHODS, UNITS, VERSION as MEASUREMENT_VERSION

SCIENCE_AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "scipy"))


def metadata_fixture():
    """Create annotation groups with distinct fake patients plus the excluded demo."""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=("ecg_id", "patient_id", "strat_fold", "filename_hr", "scp_codes"))
    writer.writeheader()
    for index in range(1, 21):
        codes = {"AFIB": 100} if index <= 5 else {"NORM": 100, "SR": 100} if index <= 10 else {"SR": 100, "LAFB": 100} if index <= 15 else {"SBRAD": 100}
        writer.writerow({"ecg_id": index, "patient_id": index + 1000, "strat_fold": 9,
                         "filename_hr": f"records500/00000/{index:05d}_hr", "scp_codes": str(codes)})
    writer.writerow({"ecg_id": 705, "patient_id": 9999, "strat_fold": 9,
                     "filename_hr": "records500/00000/00705_hr", "scp_codes": "{'AFIB': 100}"})
    registry = b",rhythm\nAFIB,1.0\nSR,1.0\nSBRAD,1.0\nLAFB,\nNORM,\n"
    return output.getvalue().encode(), registry


def selected_fixture():
    metadata, registry = metadata_fixture()
    with mock.patch.multiple(benchmark, METADATA_SHA256=hashlib.sha256(metadata).hexdigest(), REGISTRY_SHA256=hashlib.sha256(registry).hexdigest()):
        return benchmark.select_cases(metadata, registry)


def record_fixture(identifier=25):
    return Record(f"HR{identifier:05d}.mat", f"ptbxl:{identifier}", "waveform-sha", 500, STANDARD_LEADS, ("mV",) * 12)


def tool_fixture(name, record):
    if name == classifier.TOOL_NAME:
        findings = {classifier.score_key(index): .5 for index in range(150)}
        return {"record_id": record.record_id, "record_hash": record.sha256,
                "status": "success", "source": "live", "version": classifier.VERSION,
                "findings": findings, "units": dict.fromkeys(findings, "score"),
                "methods": dict.fromkeys(findings, classifier.SCORE_METHOD), "missingness": {},
                "details": classifier.provenance()}
    return {"record_id": record.record_id, "record_hash": record.sha256,
            "status": "success", "source": "live", "version": MEASUREMENT_VERSION,
            "findings": dict.fromkeys(benchmark.MEASUREMENT_KEYS, 100), "units": dict(UNITS),
            "methods": dict(METHODS), "missingness": {}, "details": {}}


def wfdb_fixture(identifier=25, names=STANDARD_LEADS):
    import numpy as np

    digital = np.tile(np.arange(12, dtype="<i2"), (5000, 1))
    digital[-1] += 2
    lines = [f"{identifier:05d}_hr 12 500 5000"]
    for column, name in enumerate(names):
        checksum = int(digital[:, column].sum(dtype=np.int64)) % 65536
        lines.append(f"{identifier:05d}_hr.dat 16 1000.0(10)/mV 16 0 {int(digital[0,column])} {checksum} 0 {name}")
    return ("\n".join(lines) + "\n").encode(), digital.tobytes()


class BenchmarkTests(unittest.TestCase):
    def test_selection_is_frozen_balanced_and_patient_disjoint(self):
        manifest = selected_fixture()
        self.assertEqual(len(manifest["cases"]), 20)
        self.assertEqual(len({case["patient_id"] for case in manifest["cases"]}), 20)
        self.assertNotIn(705, [case["ecg_id"] for case in manifest["cases"]])
        self.assertNotIn(9999, [case["patient_id"] for case in manifest["cases"]])
        self.assertEqual([case["ecg_id"] for case in manifest["cases"]], list(range(1, 21)))
        self.assertTrue(all(case["strat_fold"] == 9 for case in manifest["cases"]))

    def test_source_metadata_changes_are_rejected(self):
        metadata, registry = metadata_fixture()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            benchmark.select_cases(metadata, registry)

    def test_committed_manifest_has_verified_sources_and_distinct_patients(self):
        manifest = benchmark.load_manifest()
        self.assertEqual(len(manifest["cases"]), 20)
        self.assertEqual(manifest["excluded_patient_ids"], [4874])
        self.assertEqual([case["ecg_id"] for case in manifest["cases"][:5]], [17, 152, 717, 722, 866])

    def test_auroc_ties_and_class_absence_are_explicit(self):
        self.assertEqual(benchmark.rank_auroc([False, True], [0.1, 0.9]), 1)
        self.assertEqual(benchmark.rank_auroc([False, True], [0.9, 0.1]), 0)
        self.assertEqual(benchmark.rank_auroc([False, True], [0.5, 0.5]), 0.5)
        self.assertIsNone(benchmark.rank_auroc([True, True], [0.2, 0.8]))
        with self.assertRaises(ValueError):
            benchmark.rank_auroc([False], [float("nan")])

    def test_released_claims_do_not_include_omitted_evidence_findings(self):
        observation = ToolObservation("ptbxl:25", "waveform", "call", "ev", "call_measurement_tool", "test", "success",
                                      {"heart_rate_bpm": 70, "pr_interval_ms": 150}, {"heart_rate_bpm": "bpm", "pr_interval_ms": "ms"})
        result = SimpleNamespace(evidence=(observation,), report={"findings": [{"evidence_id": "ev", "finding": "heart_rate_bpm"}]})
        claims = benchmark.released_claims(None, 1, result)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].finding, "heart_rate_bpm")

    def test_live_check_is_pending_without_key_and_makes_no_worker(self):
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch("worker.ToolWorker") as worker:
            result = benchmark.run_dialogue({}, {}, "unused", model="explicit/model")
        self.assertEqual(result["status"], "pending")
        worker.assert_not_called()

    def test_direct_checks_both_tools_and_keeps_reference_labels_out_of_records(self):
        manifest = selected_fixture()
        calls = []

        class FakeWorker:
            classifier_provenance = {"checkpoint_sha256": "a" * 64}
            classifier_error = None
            closed = False

            def prepare_classifier(self, record, timeout):
                return True

            def call(self, name, arguments, record, timeout):
                self_record = record.public()
                if "reference" in self_record or "group" in self_record:
                    raise AssertionError("reference labels leaked into record")
                calls.append((name, arguments))
                return tool_fixture(name, record)

            def close(self):
                self.closed = True

        worker = FakeWorker()
        paths = {case["ecg_id"]: Path(f"HR{case['ecg_id']:05d}.mat") for case in manifest["cases"]}
        with mock.patch("worker.ToolWorker", return_value=worker), mock.patch("records.load_record", side_effect=lambda path: (record_fixture(int(path.stem[2:])), None)):
            report = benchmark.run_direct(manifest, paths, "unused")
        self.assertTrue(worker.closed)
        self.assertEqual(len(calls), 40)
        self.assertEqual(report["expected_records"], 20)
        self.assertEqual(report["classification_completed"], 20)
        self.assertEqual(report["metrics"]["AFIB"]["exploratory_auroc"], .5)
        self.assertTrue(all(set(arguments) == {"record_id"} for _, arguments in calls))

    def test_direct_rejects_incomplete_or_misattributed_tool_results(self):
        manifest = selected_fixture()
        paths = {case["ecg_id"]: Path(f"HR{case['ecg_id']:05d}.mat") for case in manifest["cases"]}
        cases = [
            (classifier.TOOL_NAME, lambda raw: raw.update(record_id="ptbxl:999")),
            (classifier.TOOL_NAME, lambda raw: raw.update(record_hash="wrong-waveform")),
            (classifier.TOOL_NAME, lambda raw: raw["findings"].pop("ecgfounder_149")),
            (classifier.TOOL_NAME, lambda raw: raw.update(version="unverified")),
            ("call_measurement_tool", lambda raw: raw["findings"].pop("pr_interval_ms")),
            ("call_measurement_tool", lambda raw: raw["units"].update(heart_rate_bpm="ms")),
            ("call_measurement_tool", lambda raw: raw["findings"].update(pr_interval_ms=None)),
            ("call_measurement_tool", lambda raw: raw["findings"].update(heart_rate_bpm=float("nan"))),
        ]
        for name, mutate in cases:
            worker = mock.Mock(classifier_provenance=classifier.provenance(), classifier_error=None)
            worker.prepare_classifier.return_value = True

            def result(tool, arguments, record, timeout):
                raw = tool_fixture(tool, record)
                if record.record_id == "ptbxl:1" and tool == name:
                    mutate(raw)
                return raw

            worker.call.side_effect = result
            with self.subTest(tool=name, mutation=mutate), \
                 mock.patch("worker.ToolWorker", return_value=worker), \
                 mock.patch("records.load_record", side_effect=lambda path: (record_fixture(int(path.stem[2:])), None)):
                report = benchmark.run_direct(manifest, paths, "unused")
                self.assertEqual(report["status"], "incomplete")
                self.assertEqual(report["records"][0][name]["status"], "failed")
                self.assertTrue(report["records"][0][name]["error"])
                self.assertEqual(report["classification_completed" if name == classifier.TOOL_NAME else "measurement_completed"], 19)

    def test_dialogue_failures_mark_report_incomplete_and_cli_returns_failure(self):
        manifest = selected_fixture()
        canonical = {"metadata": {}, "manifest": [], "turns": [],
                     "summary": {"expected_turns": 20, "completed_answers": 19}}
        with mock.patch.object(benchmark, "run_manifest", return_value=canonical):
            report = benchmark.run_dialogue(manifest, {}, "unused")
        with self.subTest(check="report status"):
            self.assertEqual(report.get("status"), "incomplete")
        with mock.patch.object(benchmark, "load_manifest", return_value=manifest), \
             mock.patch.object(benchmark, "prepare", return_value={}), \
             mock.patch.object(benchmark, "run_dialogue", return_value=report), \
             mock.patch("sys.stdout", new_callable=StringIO):
            with self.subTest(check="CLI exit"):
                self.assertEqual(benchmark.main(["dialogue", "--checkpoint", "unused"]), 1)

    def test_per_turn_requested_findings_validate_question_count(self):
        from evaluation import EvaluationCase, run_manifest
        case = EvaluationCase("case", "ptbxl:25", ("question",), requested_findings_by_turn=(("heart_rate_bpm",), ()))
        with self.assertRaisesRegex(ValueError, "question count"):
            run_manifest((case,), lambda _: None)

    def test_guard_rejections_are_separate_from_provider_and_tool_failures(self):
        from contracts import Decision
        from eko import Session
        from evaluation import EvaluationCase, run_manifest
        from tests.test_runtime import FakeModel, FakeWorker, RECORD

        case = EvaluationCase("blocked", RECORD.record_id, ("Interpret this ECG.",))
        result = run_manifest((case,), lambda _: Session(
            FakeModel([Decision("This is normal sinus rhythm.")]), FakeWorker(), RECORD),
            benchmark.released_claims)
        self.assertEqual(result["summary"]["answer_rejections"], 1)
        self.assertEqual(result["summary"]["answer_rejection_rate"], 1)
        self.assertEqual(result["summary"]["completed_answers"], 0)
        self.assertEqual(result["summary"]["provider_errors"], 0)
        self.assertEqual(result["summary"]["tool_errors"], 0)
        self.assertEqual(result["turns"][0]["answer_rejections"], 1)


@unittest.skipUnless(SCIENCE_AVAILABLE, "install numerical dependencies")
class ConverterTests(unittest.TestCase):
    def test_conversion_preserves_gain_baseline_and_reorders_declared_leads(self):
        names = tuple(reversed(STANDARD_LEADS))
        header, payload = wfdb_fixture(names=names)
        signal = benchmark.decode_record(header, payload, 25)
        self.assertEqual(signal.shape, (12, 5000))
        self.assertAlmostEqual(signal[0, 0], (11 - 10) / 1000)
        self.assertAlmostEqual(signal[-1, 0], (0 - 10) / 1000)

    def test_digital_corruption_frequency_format_and_units_are_rejected(self):
        header, payload = wfdb_fixture()
        for changed_header, changed_payload in (
            (header, payload[:-2]), (header.replace(b"12 500 5000", b"12 250 5000"), payload),
            (header.replace(b".dat 16 ", b".dat 24 ", 1), payload),
            (header.replace(b"/mV", b"/uV", 1), payload),
            (header, bytes([payload[0] ^ 1]) + payload[1:]),
        ):
            with self.subTest(header=changed_header[:30], length=len(changed_payload)):
                with self.assertRaises(ValueError):
                    benchmark.decode_record(changed_header, changed_payload, 25)

    def test_cache_hash_failure_does_not_redownload_or_overwrite(self):
        header, payload = wfdb_fixture()
        case = {"ecg_id": 25, "filename_hr": "records500/00000/00025_hr", "sources": {}}
        for extension, content in (("hea", header), ("dat", payload)):
            case["sources"][extension] = {"url": benchmark.BASE_URL + case["filename_hr"] + "." + extension,
                                          "sha256": hashlib.sha256(content).hexdigest()}
        with tempfile.TemporaryDirectory() as directory, mock.patch("benchmark._download", side_effect=[header, payload]):
            path, _ = benchmark._prepare_case(case, Path(directory))
            self.assertTrue(path.exists())
            digital = path.parent / "00025_hr.dat"
            digital.write_bytes(payload + b"changed")
            with mock.patch("benchmark._download") as download:
                with self.assertRaises(ValueError):
                    benchmark._prepare_case(case, Path(directory))
            download.assert_not_called()
            self.assertTrue(digital.read_bytes().endswith(b"changed"))


if __name__ == "__main__":
    unittest.main()
