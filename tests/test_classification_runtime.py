"""Classifier integration contracts; model and scientific operations stay at fixtures."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import sys
import unittest
from unittest import mock

import classifier
from contracts import Decision, STANDARD_LEADS
from eko import Session, run_turn
from tests.test_runtime import FakeModel, FakeWorker, RECORD, call, measurement, report
from worker import ToolWorker, _dispatch


ECG = replace(RECORD, lead_names=STANDARD_LEADS, waveform_units=("mV",) * 12)


def classification(**updates):
    keys = [classifier.score_key(index) for index in range(150)]
    scores = dict.fromkeys(keys, 0.1)
    scores.update(ecgfounder_001=0.92, ecgfounder_002=0.03, ecgfounder_003=0.95,
                  ecgfounder_005=0.06, ecgfounder_036=0.58)
    raw = {
        "record_id": ECG.record_id, "record_hash": ECG.sha256,
        "status": "success", "version": classifier.VERSION, "source": "live",
        "findings": scores, "units": dict.fromkeys(keys, "score"),
        "methods": dict.fromkeys(keys, classifier.SCORE_METHOD), "missingness": {},
        "details": classifier.provenance(), "elapsed": 0.01,
    }
    raw.update(updates)
    return raw


class ClassificationRuntimeTests(unittest.TestCase):
    def session(self, decisions, outputs=(), *, ready=True):
        model = FakeModel(decisions)
        worker = FakeWorker(outputs)
        worker.classifier_ready = ready
        logs = []
        session = Session(model, worker, ECG,
                          log=lambda kind, **fields: logs.append((kind, fields)))
        return session, model, worker, logs

    def test_base_session_omits_classifier_and_ready_session_advertises_both(self):
        for ready, expected in ((False, ["call_measurement_tool"]),
                                (True, ["call_measurement_tool", classifier.TOOL_NAME])):
            session, model, _, _ = self.session([report()], ready=ready)
            self.assertEqual(run_turn(session, "Which tools are available?").status, "success")
            tools = model.requests[0][1]
            self.assertEqual([tool["function"]["name"] for tool in tools], expected)
            for tool in tools:
                arguments = tool["function"]["parameters"]
                self.assertEqual(set(arguments["properties"]), {"record_id"})
                self.assertEqual(arguments["properties"]["record_id"]["enum"], [ECG.record_id])
                self.assertFalse(arguments["additionalProperties"])

    def test_measurement_and_classification_run_before_answer_and_release_only_requested_refs(self):
        session, model, worker, logs = self.session([
            Decision(tool_calls=(call("measure"), call("classify", classifier.TOOL_NAME))),
            report("ecgfounder_005"),
        ], [measurement(), classification()])
        result = run_turn(session, "Measure and classify this record.")
        self.assertEqual(result.status, "success", result.error)
        self.assertEqual((result.model_responses, result.tool_calls), (2, 2))
        self.assertEqual([entry[0] for entry in worker.calls], ["call_measurement_tool", classifier.TOOL_NAME])
        self.assertEqual(len(result.evidence), 2)
        class_evidence = next(item for item in result.evidence if item.tool == classifier.TOOL_NAME)
        classifier.validate_output(class_evidence.to_dict())
        self.assertEqual(len(class_evidence.findings), 150)
        self.assertEqual(class_evidence.details["checkpoint_sha256"], classifier.CHECKPOINT_SHA256)
        self.assertEqual(result.report, {"findings": [{"evidence_id": class_evidence.evidence_id,
                                                      "finding": "ecgfounder_005"}],
                                         "explanations": [], "unavailable": []})
        self.assertEqual(result.to_dict()["report"], result.report)
        self.assertIn("ATRIAL FIBRILLATION", result.answer)
        self.assertIn("0.06", result.answer)
        self.assertNotIn("72 bpm", result.answer)
        tool_messages = [message for message in model.requests[1][0] if message["role"] == "tool"]
        self.assertEqual([message["tool_call_id"] for message in tool_messages], ["measure", "classify"])
        released = next(fields["report"] for kind, fields in logs if kind == "report_released")
        self.assertEqual(released, result.report)

    def test_followup_reuses_classification_but_report_lists_only_newly_released_label(self):
        session, model, worker, _ = self.session([
            Decision(tool_calls=(call("classify", classifier.TOOL_NAME),)),
            report("ecgfounder_005"), report("ecgfounder_002"),
        ], [classification()])
        first = run_turn(session, "Show the atrial fibrillation score.")
        followup = run_turn(session, "Show the separate normal ECG score.")
        self.assertEqual((first.status, followup.status), ("success", "success"))
        self.assertEqual(len(worker.calls), 1)
        self.assertEqual(first.evidence[0].evidence_id, followup.evidence[0].evidence_id)
        self.assertEqual([item["finding"] for item in first.report["findings"]], ["ecgfounder_005"])
        self.assertEqual([item["finding"] for item in followup.report["findings"]], ["ecgfounder_002"])
        self.assertIn("NORMAL ECG", followup.answer)
        self.assertNotIn("ATRIAL FIBRILLATION", followup.answer)
        self.assertEqual(len([message for message in model.requests[-1][0] if message["role"] == "tool"]), 1)

    def test_invalid_classifier_version_dimension_scores_and_identity_never_become_evidence(self):
        malformed = []
        for field, value in (("version", "wrong-version"), ("status", "partial"),
                             ("record_id", "ptbxl:2"), ("record_hash", "b" * 64), ("source", "replay")):
            malformed.append((field, classification(**{field: value})))
        raw = classification()
        del raw["findings"]["ecgfounder_005"]
        malformed.append(("149 scores", raw))
        raw = classification()
        raw["findings"]["ecgfounder_150"] = 0.5
        malformed.append(("151 scores", raw))
        for value in (None, float("nan"), float("inf"), True, -0.01, 1.01):
            raw = classification()
            raw["findings"]["ecgfounder_005"] = value
            malformed.append((f"invalid score {value}", raw))
        raw = classification()
        raw["details"]["checkpoint_sha256"] = "b" * 64
        malformed.append(("wrong checkpoint", raw))
        for name, raw in malformed:
            with self.subTest(name=name):
                session, _, _, _ = self.session([
                    Decision(tool_calls=(call("classify", classifier.TOOL_NAME),)), report(),
                ], [raw])
                result = run_turn(session, "Classify this record.")
                self.assertEqual(result.status, "failed")
                self.assertEqual(session.evidence, [])
                self.assertEqual(len(result.evidence), 1)
                self.assertEqual(result.evidence[0].status, "failed")
                self.assertEqual(result.report["findings"], [])

    def test_real_classifier_observation_does_not_authorize_fabricated_diagnosis(self):
        prose = "You have atrial fibrillation and need treatment. This ECG is normal."
        session, model, worker, logs = self.session([
            Decision(tool_calls=(call("classify", classifier.TOOL_NAME),)),
            Decision(prose), report("ecgfounder_005"),
        ], [classification()])
        rejected = run_turn(session, "Is it atrial fibrillation or normal?")
        self.assertEqual(rejected.status, "failed")
        self.assertIn("unsupported final response blocked", rejected.error)
        self.assertNotIn(prose, rejected.answer)
        self.assertNotIn(prose, str(session.history))
        self.assertTrue(rejected.report["findings"])
        self.assertEqual(len(rejected.report["findings"]), 5)
        self.assertTrue(any(kind == "answer_rejected" for kind, _ in logs))
        followup = run_turn(session, "Give only the measured classifier score.")
        self.assertEqual(followup.status, "success")
        self.assertEqual(len(worker.calls), 1)
        self.assertNotIn(prose, str(model.requests[-1][0]))


class ClassifierHandshakeTests(unittest.TestCase):
    def worker(self, ready_result=None):
        ready_result = ready_result or {"status": "ready", "provenance": classifier.provenance()}
        script = f"""
import json, sys
ready = {ready_result!r}
result = {classification()!r}
for line in sys.stdin:
    request = json.loads(line)
    output = ready if request['tool'] == '__prepare_classifier__' else result
    print(json.dumps({{'id': request['id'], 'result': output}}), flush=True)
"""
        worker = ToolWorker([sys.executable, "-u", "-c", script], checkpoint="fixture-checkpoint.pth")
        self.addCleanup(worker.close)
        return worker

    def test_verified_subprocess_handshake_enables_classifier_in_session(self):
        worker = self.worker()
        self.assertFalse(worker.classifier_ready)
        self.assertTrue(worker.prepare_classifier(ECG, timeout=3))
        self.assertEqual(worker.classifier_provenance, classifier.provenance())
        self.assertIsNone(worker.classifier_error)
        model = FakeModel([Decision(tool_calls=(call("classify", classifier.TOOL_NAME),)),
                           report("ecgfounder_005")])
        result = run_turn(Session(model, worker, ECG), "Run the classifier.")
        self.assertEqual(result.status, "success", result.error)
        self.assertIn(classifier.TOOL_NAME, [tool["function"]["name"] for tool in model.requests[0][1]])
        self.assertEqual(len(result.evidence[0].findings), 150)

    def test_forged_readiness_metadata_or_status_does_not_advertise_classifier(self):
        results = [{"status": "success", "provenance": classifier.provenance()},
                   {"status": "ready", "provenance": {}}]
        for field in ("checkpoint_sha256", "label_map_sha256", "source_revision", "model_revision", "device"):
            profile = classifier.provenance()
            profile[field] = "forged"
            results.append({"status": "ready", "provenance": profile})
        for response in results:
            with self.subTest(response=response):
                worker = ToolWorker(checkpoint="fixture-checkpoint.pth")
                with mock.patch.object(worker, "_invoke", return_value=deepcopy(response)):
                    self.assertFalse(worker.prepare_classifier(ECG))
                self.assertFalse(worker.classifier_ready)
                self.assertIsNone(worker.classifier_provenance)
                session = Session(FakeModel([]), worker, ECG)
                self.assertEqual([tool["function"]["name"] for tool in session.tools], ["call_measurement_tool"])

    def test_readiness_transport_rejects_wrong_response_id(self):
        script = "import json,sys\nfor line in sys.stdin:\n print(json.dumps({'id':'forged','result':{'status':'ready'}}),flush=True)"
        worker = ToolWorker([sys.executable, "-u", "-c", script], checkpoint="fixture.pth")
        self.addCleanup(worker.close)
        self.assertFalse(worker.prepare_classifier(ECG, timeout=3))
        self.assertIsNone(worker.process)
        self.assertIn("does not match", worker.classifier_error)

    def test_unconfigured_or_failed_classifier_preserves_measurement_capability(self):
        for checkpoint in (None, "invalid-checkpoint.pth"):
            worker = ToolWorker(checkpoint=checkpoint)
            with mock.patch.object(worker, "_invoke", return_value={"status": "failed", "details": {"message": "bad checksum"}}):
                self.assertFalse(worker.prepare_classifier(ECG))
            failure = worker.call(classifier.TOOL_NAME, {"record_id": ECG.record_id}, ECG)
            self.assertEqual(failure["details"]["reason"], "classifier_unavailable")
            with mock.patch.object(worker, "_invoke", return_value=measurement()) as invoke:
                result = worker.call("call_measurement_tool", {"record_id": ECG.record_id}, ECG)
            self.assertEqual(result["status"], "success")
            self.assertEqual(invoke.call_args.args[0], "call_measurement_tool")


class ChildClassifierCacheTests(unittest.TestCase):
    def request(self, name, checkpoint="fixture.pth"):
        return {"id": "fixture-request", "tool": name, "arguments": {"record_id": ECG.record_id},
                "record": asdict(ECG), "checkpoint": checkpoint}

    def test_child_reuses_one_loaded_classifier_across_prepare_and_multiple_calls(self):
        backend = mock.Mock()
        backend.provenance.return_value = classifier.provenance()
        backend.classify.return_value = classification()
        cache = {}
        with mock.patch.object(classifier, "Classifier", return_value=backend) as factory, \
             mock.patch.object(classifier, "preprocess") as preprocess:
            ready = _dispatch(self.request("__prepare_classifier__"), cache)
            first = _dispatch(self.request(classifier.TOOL_NAME), cache)
            second = _dispatch(self.request(classifier.TOOL_NAME), cache)
        self.assertEqual(ready, {"status": "ready", "provenance": classifier.provenance()})
        factory.assert_called_once_with("fixture.pth")
        preprocess.assert_called_once_with(ECG)
        self.assertEqual(backend.classify.call_args_list, [mock.call(ECG), mock.call(ECG)])
        self.assertEqual((first["status"], second["status"]), ("success", "success"))
        self.assertIs(cache["classifier"], backend)

    def test_failed_checkpoint_does_not_cache_model_or_disable_measurements(self):
        cache = {}
        with mock.patch.object(classifier, "Classifier", side_effect=ValueError("checkpoint checksum mismatch")), \
             mock.patch("measurements.measure_record", return_value=measurement()) as measure:
            ready = _dispatch(self.request("__prepare_classifier__"), cache)
            measured = _dispatch(self.request("call_measurement_tool"), cache)
        self.assertEqual(ready["status"], "failed")
        self.assertEqual(ready["details"]["reason"], "classifier_validation_failed")
        self.assertEqual(cache, {})
        self.assertEqual(measured["status"], "success")
        measure.assert_called_once_with(ECG)

    def test_child_refuses_checkpoint_changes_without_replacing_cached_model(self):
        backend = mock.Mock()
        cache = {"classifier": backend, "checkpoint": "first.pth"}
        with mock.patch.object(classifier, "Classifier") as factory:
            result = _dispatch(self.request(classifier.TOOL_NAME, checkpoint="second.pth"), cache)
        self.assertEqual(result["status"], "failed")
        self.assertIn("restart", result["details"]["message"])
        factory.assert_not_called()
        backend.classify.assert_not_called()
        self.assertIs(cache["classifier"], backend)


if __name__ == "__main__":
    unittest.main()
