"""Native terminal reports share the existing evidence validator."""

import json
import unittest

from contracts import Decision, ToolCall
from eko import Session, run_turn
from tests.test_runtime import FakeModel, FakeWorker, RECORD, call, measurement, report


def submit(messages):
    observation = next(json.loads(message["content"]) for message in reversed(messages)
                       if message["role"] == "tool")
    return Decision("This ignored prose must never be displayed.", tool_calls=(ToolCall(
        "final", "submit_report", {"findings": [{"evidence_id": observation["evidence_id"], "finding": "heart_rate"}],
                                     "explanations": [], "unavailable": []}),))


class SubmissionTests(unittest.TestCase):
    def test_followup_history_uses_validated_report_json_while_display_remains_rendered(self):
        for final, retries in ((report("heart_rate"), ()),
                               (Decision("Unvalidated prose"), (Decision("Still unvalidated prose"),))):
            model = FakeModel([Decision(tool_calls=(call(),)), final, *retries, report("pr_interval")])
            session = Session(model, FakeWorker([measurement()]), RECORD)
            first = run_turn(session, "Measure heart rate.")
            history_report = json.loads(session.history[-1]["content"])
            self.assertEqual(history_report, first.report)
            self.assertIn("72", first.answer)
            self.assertNotEqual(first.answer, session.history[-1]["content"])
            followup = run_turn(session, "Repeat the PR interval from that evidence.")
            self.assertEqual(followup.status, "success")
            previous = [item for item in model.requests[-1][0] if item["role"] == "assistant" and item.get("content")]
            self.assertEqual(json.loads(previous[-1]["content"]), first.report)
            self.assertNotIn("Unvalidated prose", str(model.requests[-1][0]))

    def test_terminal_submission_does_not_execute_a_scientific_tool_or_leak_prose(self):
        model = FakeModel([Decision(tool_calls=(call(),)), submit])
        worker = FakeWorker([measurement()])
        session = Session(model, worker, RECORD)
        result = run_turn(session, "Measure heart rate.")
        self.assertEqual(result.status, "success", result.error)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(len(worker.calls), 1)
        self.assertIn("72", result.answer)
        self.assertNotIn("ignored prose", str(session.history))
        self.assertNotIn('"name": "submit_report"', json.dumps(session.history))
        schema = next(tool for tool in model.requests[1][1] if tool["function"]["name"] == "submit_report")
        finding = schema["function"]["parameters"]["properties"]["findings"]["items"]["properties"]
        self.assertEqual(finding["evidence_id"]["enum"], [result.evidence[0].evidence_id])
        self.assertEqual(set(finding["finding"]["enum"]), {"heart_rate", "pr_interval"})

    def test_values_and_unknown_finding_keys_in_native_reports_are_rejected(self):
        for reference in ({"evidence_id": "fake", "finding": "heart_rate"},
                          {"evidence_id": "fake", "finding": "heart_rate", "value": 999}):
            model = FakeModel([Decision(tool_calls=(ToolCall("final", "submit_report", {
                "findings": [reference], "explanations": [], "unavailable": []}),))])
            worker = FakeWorker()
            result = run_turn(Session(model, worker, RECORD), "Report heart rate.")
            self.assertEqual(result.status, "failed")
            self.assertIn("unsupported final response", result.error)
            self.assertEqual(worker.calls, [])
            self.assertNotIn("999", result.answer)

    def test_submission_mixed_with_acquisition_is_rejected_without_partial_execution(self):
        model = FakeModel([Decision(tool_calls=(call(), ToolCall("final", "submit_report", {
            "findings": [], "explanations": ["capabilities"], "unavailable": []})))])
        worker = FakeWorker()
        result = run_turn(Session(model, worker, RECORD), "Measure heart rate.")
        self.assertEqual(result.status, "failed")
        self.assertIn("alone", result.error)
        self.assertEqual(worker.calls, [])
