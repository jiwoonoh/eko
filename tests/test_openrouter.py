"""HTTP contract tests never send requests to OpenRouter."""

import io
import json
import time
import unittest
from unittest import mock

from eko import measurement_schema
from providers.openrouter import OpenRouter


def response(content="Done.", calls=None, finish=None, **extra):
    message = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    return {
        "model": "actual/provider-model", "usage": {"total_tokens": 17},
        "choices": [{"finish_reason": finish or ("tool_calls" if calls else "stop"),
                     "message": message}], **extra,
    }


def native_call(call_id="a", arguments='{"record_id":"ptbxl:1"}'):
    return {"id": call_id, "type": "function",
            "function": {"name": "call_measurement_tool", "arguments": arguments}}


class FakeResponse(io.BytesIO):
    def __init__(self, body, status=200):
        super().__init__(body if isinstance(body, bytes) else json.dumps(body).encode())
        self.status = status


class FakeConnection:
    def __init__(self, body, status=200):
        self.response = FakeResponse(body, status)
        self.sock = mock.Mock()
        self.requests = []
        self.closed = False

    def request(self, method, path, body, headers):
        self.requests.append((method, path, json.loads(body), headers))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class OpenRouterTests(unittest.TestCase):
    def test_required_decisions_are_an_explicit_provider_setting(self):
        model = OpenRouter("test/model", api_key="test-key", tool_choice="required")
        connection = FakeConnection(response(content=None, calls=[native_call()]))
        with mock.patch("providers.openrouter.http.client.HTTPSConnection", return_value=connection):
            decision = model.complete([{"role": "user", "content": "Measure."}], [measurement_schema("ptbxl:1")])
        self.assertEqual(connection.requests[0][2]["tool_choice"], "required")
        self.assertEqual(decision.tool_calls[0].name, "call_measurement_tool")
        with self.assertRaises(ValueError):
            OpenRouter("test/model", api_key="test-key", tool_choice="unknown")

    def complete(self, connections, messages=None, tools=None):
        model = OpenRouter("requested/model", "test-secret")
        factory = mock.Mock(side_effect=connections)
        with mock.patch("providers.openrouter.http.client.HTTPSConnection", factory):
            decision = model.complete(messages or [{"role": "user", "content": "Measure."}],
                                      tools or [measurement_schema("ptbxl:1")])
        return model, decision, factory

    def test_native_tool_calls_usage_and_actual_model_preserved(self):
        connection = FakeConnection(response(content=None, calls=[native_call()]))
        _, decision, _ = self.complete([connection])
        self.assertEqual(decision.model, "actual/provider-model")
        self.assertEqual(decision.usage, {"total_tokens": 17})
        self.assertEqual(decision.tool_calls[0].arguments, {"record_id": "ptbxl:1"})
        _, _, payload, headers = connection.requests[0]
        self.assertEqual(payload["model"], "requested/model")
        self.assertEqual(payload["provider"], {"require_parameters": True})
        self.assertNotIn("parallel_tool_calls", payload)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(headers["Authorization"], "Bearer test-secret")
        self.assertTrue(connection.closed)

    def test_multiple_native_tool_calls_are_preserved_for_local_dispatch(self):
        connection = FakeConnection(response(content=None, calls=[native_call("a"), native_call("b")]))
        _, decision, _ = self.complete([connection])
        self.assertEqual([call.id for call in decision.tool_calls], ["a", "b"])

    def test_free_router_accepts_only_supported_tool_parameters(self):
        # The free endpoints advertise tools/tool_choice/max_tokens, but not
        # parallel_tool_calls. Strict parameter routing rejects that extra hint.
        class FreeEndpoint(FakeConnection):
            def request(self, method, path, body, headers):
                super().request(method, path, body, headers)
                payload = json.loads(body)
                tool_parameters = {"tools", "tool_choice", "max_tokens", "parallel_tool_calls"}
                supported = {"tools", "tool_choice", "max_tokens"}
                unsupported = (set(payload) & tool_parameters) - supported
                if payload["provider"]["require_parameters"] and unsupported:
                    self.response = FakeResponse({"error": {
                        "code": 404,
                        "message": "No endpoints found that can handle the requested parameters",
                        "metadata": {"failed_routing_step": "Filter by Parameters"},
                    }}, 404)

        connection = FreeEndpoint(response(content=None, calls=[native_call()]))
        model = OpenRouter("openrouter/free", "test-secret")
        with mock.patch("providers.openrouter.http.client.HTTPSConnection", return_value=connection):
            decision = model.complete([{"role": "user", "content": "hi"}], [measurement_schema("ptbxl:1")])
        self.assertEqual(decision.tool_calls[0].name, "call_measurement_tool")
        self.assertEqual(connection.requests[0][2]["provider"], {"require_parameters": True})

    def test_adapter_does_not_keep_private_history(self):
        first, second = FakeConnection(response()), FakeConnection(response("Again."))
        model = OpenRouter("requested/model", "test-secret")
        with mock.patch("providers.openrouter.http.client.HTTPSConnection", side_effect=[first, second]):
            model.complete([{"role": "user", "content": "First."}], [])
            model.complete([{"role": "user", "content": "Second."}], [])
        self.assertEqual(second.requests[0][2]["messages"], [{"role": "user", "content": "Second."}])

    def test_empty_response_gets_one_retry(self):
        first, second = FakeConnection(response("")), FakeConnection(response())
        _, decision, factory = self.complete([first, second])
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(decision.content, "Done.")
        self.assertEqual(first.requests[0][2], second.requests[0][2])

    def test_retry_retains_actual_model_and_usage_for_each_attempt(self):
        first, second = FakeConnection(response("")), FakeConnection(response())
        model, _, _ = self.complete([first, second])
        self.assertEqual([entry["status"] for entry in model.last_attempts], ["failed", "success"])
        self.assertEqual([entry["usage"]["total_tokens"] for entry in model.last_attempts], [17, 17])

    def test_watchdog_expiry_cannot_return_late_success_or_retry(self):
        connection = FakeConnection(response())
        model = OpenRouter("requested/model", "test-secret")
        read = connection.response.read1

        def expire_during_read(size):
            model._expire()
            return read(size)

        connection.response.read1 = expire_during_read
        factory = mock.Mock(return_value=connection)
        with mock.patch("providers.openrouter.http.client.HTTPSConnection", factory):
            with self.assertRaises(TimeoutError):
                model.complete([], [])
        self.assertEqual(factory.call_count, 1)
        self.assertTrue(connection.closed)

    def test_nonfinite_usage_rejected_without_unserializable_attempt_log(self):
        connection = FakeConnection(response(usage={"total_tokens": float("nan")}))
        model = OpenRouter("requested/model", "test-secret")
        with mock.patch("providers.openrouter.http.client.HTTPSConnection", return_value=connection):
            with self.assertRaises(RuntimeError):
                model.complete([], [])
        json.dumps(model.last_attempts, allow_nan=False)

    def test_two_empty_responses_stop_after_one_retry(self):
        connections = [FakeConnection(response("")), FakeConnection(response(""))]
        factory = mock.Mock(side_effect=connections)
        model = OpenRouter("requested/model", "test-secret")
        with mock.patch("providers.openrouter.http.client.HTTPSConnection", factory):
            with self.assertRaisesRegex(RuntimeError, "after one retry"):
                model.complete([], [])
        self.assertEqual(factory.call_count, 2)

    def test_transient_http_failure_gets_one_retry(self):
        first, second = FakeConnection({"error": "busy"}, 429), FakeConnection(response())
        _, decision, factory = self.complete([first, second])
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(decision.content, "Done.")

    def test_bad_authentication_does_not_retry_or_expose_key(self):
        connection = FakeConnection({"error": "invalid key test-secret"}, 401)
        model = OpenRouter("requested/model", "test-secret")
        factory = mock.Mock(return_value=connection)
        with mock.patch("providers.openrouter.http.client.HTTPSConnection", factory):
            with self.assertRaises(RuntimeError) as caught:
                model.complete([], [])
        self.assertNotIn("test-secret", str(caught.exception))
        self.assertIn("[redacted]", str(caught.exception))
        self.assertEqual(factory.call_count, 1)

    def test_truncated_completion_is_rejected_before_tool_execution(self):
        connection = FakeConnection(response(content=None, calls=[native_call()], finish="length"))
        model = OpenRouter("requested/model", "test-secret")
        with mock.patch("providers.openrouter.http.client.HTTPSConnection", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "length"):
                model.complete([], [])

    def test_malformed_arguments_duplicate_ids_and_missing_calls_are_rejected(self):
        bodies = [
            response(None, [native_call(arguments='{"record_id": NaN}')]),
            response(None, [native_call(arguments='{"record_id": Infinity}')]),
            response(None, [native_call(arguments="[]")]),
            response(None, [native_call(arguments="{")]),
            response(None, [native_call("a"), native_call("a")]),
            response(None, [], finish="tool_calls"),
        ]
        for body in bodies:
            with self.subTest(body=body):
                model = OpenRouter("requested/model", "test-secret")
                with mock.patch("providers.openrouter.http.client.HTTPSConnection",
                                return_value=FakeConnection(body)):
                    with self.assertRaises(RuntimeError):
                        model.complete([], [])

    def test_expired_request_deadline_does_not_connect_or_retry(self):
        model = OpenRouter("requested/model", "test-secret")
        model.set_deadline(time.monotonic() - 1)
        with mock.patch("providers.openrouter.http.client.HTTPSConnection") as factory:
            with self.assertRaises(TimeoutError):
                model.complete([], [])
        factory.assert_not_called()

    def test_interrupt_closes_active_socket_without_retry_and_next_request_works(self):
        first = FakeConnection(response())
        second = FakeConnection(response("Recovered."))
        model = OpenRouter("requested/model", "test-secret")
        first.response.read1 = lambda _size: (model.interrupt(), b"")[1]
        factory = mock.Mock(side_effect=[first, second])
        with mock.patch("providers.openrouter.http.client.HTTPSConnection", factory):
            with self.assertRaises(InterruptedError):
                model.complete([], [])
            decision = model.complete([], [])
        self.assertEqual(decision.content, "Recovered.")
        first.sock.shutdown.assert_called_once()
        self.assertEqual(factory.call_count, 2)

    def test_model_and_key_are_required(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                OpenRouter("requested/model")
            with self.assertRaises(ValueError):
                OpenRouter("", "test-secret")


if __name__ == "__main__":
    unittest.main()
