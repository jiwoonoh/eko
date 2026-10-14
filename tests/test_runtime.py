"""Runtime tests use real contracts and fake model/tool boundaries."""

from copy import deepcopy
from dataclasses import replace
import math
import unittest
from unittest import mock

from contracts import Decision, Record, ToolCall
from eko import Session, measurement_schema, run_turn


RECORD = Record("local-only.mat", "ptbxl:1", "a" * 64, 500.0, ("II",), ("mV",))


def call(call_id="call-1", name="call_measurement_tool", **args):
    return ToolCall(call_id, name, args or {"record_id": RECORD.record_id})


def measurement(**updates):
    result = {
        "record_id": RECORD.record_id, "record_hash": RECORD.sha256,
        "status": "success", "version": "fixture-1", "source": "fixture",
        "findings": {"heart_rate": 72.0, "pr_interval": 160.0},
        "units": {"heart_rate": "bpm", "pr_interval": "ms"},
        "methods": {"heart_rate": "R-peak intervals"}, "missingness": {},
        "details": {}, "elapsed": 0.01,
    }
    result.update(updates)
    return result


class FakeModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.requests = []
        self.deadlines = []
        self.interrupted = False

    def complete(self, messages, tools):
        self.requests.append((messages, tools))
        decision = self.decisions.pop(0)
        if isinstance(decision, BaseException):
            raise decision
        return decision

    def set_deadline(self, deadline):
        self.deadlines.append(deadline)

    def interrupt(self):
        self.interrupted = True

    def close(self):
        pass


class FakeWorker:
    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])
        self.calls = []
        self.closed = 0

    def call(self, name, arguments, record, timeout):
        self.calls.append((name, arguments, record, timeout))
        result = self.outputs.pop(0) if self.outputs else measurement()
        if isinstance(result, BaseException):
            raise result
        return deepcopy(result)

    def close(self):
        self.closed += 1


class RuntimeTests(unittest.TestCase):
    def session(self, decisions, outputs=None, **options):
        model, worker = FakeModel(decisions), FakeWorker(outputs)
        logs = []
        session = Session(model, worker, RECORD,
                          log=lambda kind, **fields: logs.append((kind, fields)), **options)
        return session, model, worker, logs

    def test_measurements_are_actual_tool_results_and_followup_reuses_evidence(self):
        session, model, worker, logs = self.session([
            Decision("Tool_Output: heart_rate=999", (call(),), "fixture-model", {"total_tokens": 8}),
            Decision("Heart rate is 72 bpm."),
            Decision("PR is 160 ms."),
        ])
        result = run_turn(session, "Measure this ECG.")
        followup = run_turn(session, "And its PR interval?")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.evidence[0].findings["heart_rate"], 72)
        self.assertEqual(len(worker.calls), 1)
        self.assertEqual(followup.evidence[0].evidence_id, result.evidence[0].evidence_id)
        tool_messages = [message for message in model.requests[1][0] if message["role"] == "tool"]
        self.assertEqual(tool_messages[0]["tool_call_id"], "call-1")
        self.assertNotIn("local-only.mat", model.requests[0][0][0]["content"])
        logged = next(fields for kind, fields in logs if kind == "model_response")
        self.assertEqual(logged["model"], "fixture-model")
        self.assertEqual(logged["usage"], {"total_tokens": 8})

    def test_two_tools_in_same_response_are_both_executed_and_attributed(self):
        tools = [measurement_schema(RECORD.record_id), measurement_schema(RECORD.record_id)]
        tools[1]["function"]["name"] = "fixture_second_tool"
        session, model, worker, _ = self.session([
            Decision(tool_calls=(call("a"), call("b", "fixture_second_tool"))),
            Decision("Both requested analyses completed."),
        ], tools=tools)
        result = run_turn(session, "Use both fixtures.")
        self.assertEqual(result.status, "success")
        self.assertEqual([observation.call_id for observation in result.evidence], ["a", "b"])
        self.assertEqual(len(worker.calls), 2)
        messages = [message for message in model.requests[1][0] if message["role"] == "tool"]
        self.assertEqual([message["tool_call_id"] for message in messages], ["a", "b"])

    def test_repeated_model_tool_model_loop_is_not_limited_to_one_tool(self):
        session, _, worker, _ = self.session([
            Decision(tool_calls=(call("a"),)), Decision(tool_calls=(call("b"),)),
            Decision("Both measurements are available."),
        ])
        result = run_turn(session, "Check again.")
        self.assertEqual((result.model_responses, result.tool_calls), (3, 2))
        self.assertEqual(len(worker.calls), 2)

    def test_tool_cap_pairs_every_requested_id_and_prevents_more_execution(self):
        session, _, worker, _ = self.session([
            Decision(tool_calls=tuple(call(str(index)) for index in range(5))),
        ])
        result = run_turn(session, "Repeat measurements.")
        self.assertEqual(result.status, "limit")
        self.assertEqual(len(worker.calls), 4)
        self.assertEqual(result.tool_calls, 4)
        messages = [message for message in session.history if message["role"] == "tool"]
        self.assertEqual([message["tool_call_id"] for message in messages], list(map(str, range(5))))
        self.assertEqual(result.evidence[-1].status, "failed")
        self.assertEqual(len(session.evidence), 4)

    def test_model_response_cap_produces_no_fabricated_final_answer(self):
        session, _, _, _ = self.session([Decision(tool_calls=(call(),))], max_responses=1)
        result = run_turn(session, "Measure.")
        self.assertEqual(result.status, "limit")
        self.assertEqual(result.answer, "")
        self.assertEqual(result.model_responses, 1)

    def test_classification_unavailable_and_not_advertised(self):
        session, model, worker, _ = self.session([
            Decision(tool_calls=(call(name="call_classification_tool"),)),
            Decision("Classification is unavailable."),
        ])
        result = run_turn(session, "Classify.")
        self.assertEqual(worker.calls, [])
        self.assertEqual(result.status, "failed")
        self.assertIn("unavailable tool", result.error)
        self.assertEqual([item["function"]["name"] for item in model.requests[0][1]],
                         ["call_measurement_tool"])
        self.assertEqual(session.evidence, [])

    def test_model_paths_and_wrong_record_ids_never_reach_worker(self):
        for args in ({"record_id": "ptbxl:2"}, {"record_id": RECORD.record_id, "ecg_path": "/secret.mat"}):
            with self.subTest(args=args):
                session, _, worker, _ = self.session([
                    Decision(tool_calls=(ToolCall("a", "call_measurement_tool", args),)),
                    Decision("I could not execute that request."),
                ])
                result = run_turn(session, "Measure.")
                self.assertEqual(result.status, "failed")
                self.assertEqual(worker.calls, [])

    def test_corrected_same_tool_request_succeeds_and_keeps_failed_attempt(self):
        invalid = ToolCall("invalid", "call_measurement_tool", {"record_id": "ptbxl:2"})
        session, _, worker, _ = self.session([
            Decision(tool_calls=(invalid,)), Decision(tool_calls=(call("corrected"),)),
            Decision("Heart rate is 72 bpm."),
        ])
        result = run_turn(session, "Measure.")
        self.assertEqual(result.status, "success")
        self.assertIsNone(result.error)
        self.assertEqual(len(worker.calls), 1)
        self.assertEqual(len(session.evidence), 1)
        self.assertEqual(sorted(item.status for item in result.evidence), ["failed", "success"])

    def test_repaired_tool_failure_succeeds_but_unavailable_other_tool_remains_failed(self):
        session, _, _, _ = self.session([
            Decision(tool_calls=(call("first"),)), Decision(tool_calls=(call("retry"),)),
            Decision("Heart rate is 72 bpm."),
        ], [measurement(status="failed", findings={}, details={"reason": "no R peaks"}), measurement()])
        result = run_turn(session, "Measure.")
        self.assertEqual(result.status, "success")
        self.assertEqual(sorted(item.status for item in result.evidence), ["failed", "success"])

        session, _, _, _ = self.session([
            Decision(tool_calls=(call("classify", "call_classification_tool"),)),
            Decision(tool_calls=(call("measure"),)), Decision("Measurement succeeded; classification unavailable."),
        ])
        result = run_turn(session, "Classify and measure.")
        self.assertEqual(result.status, "failed")
        self.assertIn("unavailable tool", result.error)

    def test_wrong_hash_nonfinite_and_unexplained_missing_values_rejected(self):
        bad_outputs = [
            measurement(record_hash="b" * 64),
            measurement(findings={"heart_rate": math.nan}),
            measurement(findings={"heart_rate": math.inf}),
            measurement(findings={"heart_rate": None}, missingness={}),
            measurement(units={}),
        ]
        for output in bad_outputs:
            with self.subTest(output=output):
                session, _, _, _ = self.session([
                    Decision(tool_calls=(call(),)), Decision("The measurement was rejected."),
                ], [output])
                result = run_turn(session, "Measure.")
                self.assertEqual(result.status, "failed")
                self.assertEqual(session.evidence, [])
                self.assertEqual(result.evidence[0].status, "failed")

    def test_partial_observation_keeps_missingness_and_finite_results(self):
        output = measurement(status="partial", findings={"heart_rate": 72, "pr_interval": None},
                             missingness={"pr_interval": "P onset was not detected"})
        session, _, _, _ = self.session([
            Decision(tool_calls=(call(),)), Decision("PR could not be measured."),
        ], [output])
        result = run_turn(session, "Measure.")
        self.assertEqual(result.status, "success")
        self.assertIsNone(result.evidence[0].findings["pr_interval"])
        self.assertIn("P onset", result.evidence[0].missingness["pr_interval"])

    def test_failed_tool_observation_is_auditable_but_not_reusable_evidence(self):
        output = measurement(status="failed", findings={}, details={"error": "no R peaks"})
        session, _, _, _ = self.session([
            Decision(tool_calls=(call(),)), Decision("There were no usable R peaks."),
        ], [output])
        result = run_turn(session, "Measure.")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.evidence[0].status, "failed")
        self.assertEqual(session.evidence, [])
        self.assertEqual(result.error, "tool: no R peaks")

    def test_failed_observation_preserves_supplied_reason(self):
        for fields in ({"reason": "too few usable peaks"}, {"message": "signal is empty"},
                       {"details": {"reason": "lead unavailable"}}):
            with self.subTest(fields=fields):
                output = measurement(status="failed", findings={}, **fields)
                session, _, _, _ = self.session([
                    Decision(tool_calls=(call(),)), Decision("Measurement unavailable."),
                ], [output])
                result = run_turn(session, "Measure.")
                reason = fields.get("reason") or fields.get("message") or fields["details"]["reason"]
                self.assertEqual(result.error, f"tool: {reason}")

    def test_timeout_stops_worker_and_completes_all_native_call_ids(self):
        session, _, worker, _ = self.session([
            Decision(tool_calls=(call("a"), call("b"))),
        ], [TimeoutError("fixture exceeded 30 seconds")])
        result = run_turn(session, "Measure.")
        self.assertEqual(result.status, "timeout")
        self.assertEqual(worker.closed, 1)
        self.assertEqual([message["tool_call_id"] for message in session.history
                          if message["role"] == "tool"], ["a", "b"])

    def test_interrupt_during_tools_leaves_history_valid_for_next_turn(self):
        session, model, worker, _ = self.session([
            Decision(tool_calls=(call("a"), call("b"))), Decision("Ready for a new request."),
        ], [KeyboardInterrupt()])
        result = run_turn(session, "Measure.")
        next_result = run_turn(session, "Try a different question.")
        self.assertEqual(result.status, "interrupted")
        self.assertTrue(model.interrupted)
        self.assertEqual(worker.closed, 1)
        self.assertEqual(next_result.status, "success")
        self.assertEqual([message["tool_call_id"] for message in model.requests[1][0]
                          if message["role"] == "tool"], ["a", "b"])

    def test_provider_errors_and_empty_responses_fail_cleanly(self):
        for response in (RuntimeError("bad credentials"), TimeoutError("deadline"), Decision()):
            session, _, worker, _ = self.session([response])
            result = run_turn(session, "Measure.")
            self.assertIn(result.status, ("failed", "timeout"))
            self.assertTrue(result.error.startswith("provider:"))
            self.assertEqual(worker.calls, [])
            self.assertEqual(result.answer, "")

    def test_request_deadline_is_per_model_request_not_per_turn(self):
        session, model, worker, _ = self.session([
            Decision(tool_calls=(call(),)), Decision("Done."),
        ])
        clock = iter(range(0, 1000, 10))
        with mock.patch("eko.time.monotonic", side_effect=lambda: next(clock)):
            result = run_turn(session, "Measure.")
        self.assertEqual(result.status, "success")
        deadlines = [value for value in model.deadlines if value is not None]
        self.assertEqual(len(deadlines), 2)
        self.assertGreater(deadlines[1], deadlines[0])
        self.assertEqual(model.deadlines[-1], None)
        self.assertEqual(worker.calls[0][3], 30)

    def test_record_change_requires_explicit_reset(self):
        session, model, _, _ = self.session([Decision("Ready."), Decision("New record.")])
        run_turn(session, "Hello.")
        new = replace(RECORD, record_id="ptbxl:2", sha256="b" * 64)
        session.record = new
        result = run_turn(session, "Use the new file.")
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(model.requests), 1)
        session.set_record(new)
        self.assertEqual(len(session.history), 1)
        self.assertEqual(session.evidence, [])
        result = run_turn(session, "Use the new file.")
        self.assertEqual(result.status, "success")
        self.assertNotIn("ptbxl:1", model.requests[-1][0][0]["content"])


if __name__ == "__main__":
    unittest.main()
