"""One schema correction reuses evidence and never executes scientific tools."""

import json
import unittest

from contracts import Decision
from eko import Session, run_turn
from tests.test_runtime import FakeModel, FakeWorker, RECORD, call, measurement, report


class CorrectionTests(unittest.TestCase):
    def test_evaluation_separates_first_pass_corrected_and_unscored_tasks(self):
        from evaluation import EvaluationCase, run_manifest
        cases = (EvaluationCase("task", RECORD.record_id, ("Explain what is unavailable.",),
                                required_explanations_by_turn=(("capabilities",),)),)
        result = run_manifest(cases, lambda case: Session(FakeModel([
            Decision("bad report"), report(explanations=("capabilities",))]), FakeWorker(), RECORD))
        self.assertEqual(result["summary"]["completed_tasks"], 1)
        self.assertEqual(result["summary"]["first_pass_tasks"], 0)
        self.assertEqual(result["summary"]["corrected_tasks"], 1)
        self.assertEqual(result["summary"]["report_corrections"], 1)
        unscored = run_manifest((EvaluationCase("unspecified", RECORD.record_id, ("Hello",)),),
            lambda case: Session(FakeModel([report()]), FakeWorker(), RECORD))
        self.assertEqual(unscored["summary"]["task_scored_turns"], 0)
        self.assertIsNone(unscored["summary"]["task_completion_rate"])

    def test_bad_report_is_corrected_once_without_new_evidence_or_history_leak(self):
        bad = "Unsupported interpretation that must stay out of history."
        model = FakeModel([Decision(tool_calls=(call(),)), Decision(bad), report("heart_rate")])
        worker = FakeWorker([measurement()]);events=[]
        session = Session(model, worker, RECORD, log=lambda kind, **fields: events.append((kind, fields)))
        result = run_turn(session, "Report heart rate.")
        self.assertEqual(result.status, "success", result.error)
        self.assertEqual((result.tool_calls, result.model_responses), (1, 3))
        self.assertEqual(len(worker.calls), 1)
        self.assertNotIn(bad, str(session.history))
        self.assertNotIn(bad, str(model.requests[-1][0]))
        self.assertIn("Report validation", model.requests[-1][0][0]["content"])
        self.assertEqual(sum(kind=='answer_rejected' for kind, _ in events), 1)
        self.assertEqual(sum(kind=='report_correction' for kind, _ in events), 1)

    def test_second_bad_report_stops_and_response_cap_is_preserved(self):
        for cap, expected in ((8, 2), (1, 1)):
            model = FakeModel([Decision("bad report"), Decision("still bad")])
            result = run_turn(Session(model, FakeWorker(), RECORD, max_responses=cap), "Explain limits.")
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.model_responses, expected)
            self.assertEqual(len(model.requests), expected)

    def test_correction_cannot_repeat_a_scientific_tool(self):
        model = FakeModel([Decision(tool_calls=(call(),)), Decision("bad report"),
                           Decision(tool_calls=(call("repeat"),))])
        worker = FakeWorker([measurement()])
        result = run_turn(Session(model, worker, RECORD), "Measure heart rate.")
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(worker.calls), 1)
        self.assertIn("correction", result.error)

    def test_provider_failure_during_correction_preserves_safe_partial_answer(self):
        model = FakeModel([Decision(tool_calls=(call(),)), Decision("bad report"), RuntimeError("offline")])
        result = run_turn(Session(model, FakeWorker([measurement()]), RECORD), "Measure heart rate.")
        self.assertEqual(result.status, "failed")
        self.assertIn("72", result.answer)
        self.assertIn("offline", result.error)
