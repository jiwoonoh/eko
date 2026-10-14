"""OpenRouter native tool calls over the Python standard library."""

from __future__ import annotations

import http.client
import json
import math
import os
import socket
import threading
import time
from typing import Any

from contracts import Decision, ToolCall


class _RetryableError(RuntimeError):
    pass


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


class OpenRouter:
    """Stateless message adapter with one bounded transient/empty retry."""

    def __init__(self, model: str, api_key: str | None = None, *,
                 max_tokens: int = 4096, timeout: float = 60.0) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("an OpenRouter model ID is required")
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("Set OPENROUTER_API_KEY for OpenRouter")
        if max_tokens < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("max_tokens and timeout must be positive")
        self.model = model.strip()
        self.api_key = key
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._deadline: float | None = None
        self._connection: http.client.HTTPSConnection | None = None
        self._interrupted = threading.Event()
        self._expired = threading.Event()
        self.last_attempts: list[dict[str, Any]] = []

    def set_deadline(self, deadline: float | None) -> None:
        self._deadline = deadline

    def _safe(self, error: object) -> str:
        return str(error).replace(self.api_key, "[redacted]")

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Decision:
        self._interrupted.clear()
        self._expired.clear()
        self.last_attempts = []
        deadline = self._deadline or (time.monotonic() + self.timeout)
        payload = {
            "model": self.model, "messages": messages, "tools": tools,
            "tool_choice": "auto", "parallel_tool_calls": False,
            "provider": {"require_parameters": True},
            "max_tokens": self.max_tokens, "stream": False,
        }
        body = json.dumps(payload, allow_nan=False).encode()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("OpenRouter request deadline exceeded")
        timer = threading.Timer(remaining, self._expire)
        timer.daemon = True
        timer.start()
        try:
            for attempt in range(2):
                entry: dict[str, Any] = {"attempt": attempt + 1}
                self.last_attempts.append(entry)
                try:
                    data = self._request(body, deadline)
                    entry["model"] = data.get("model") if isinstance(data.get("model"), str) else None
                    usage = data.get("usage") or {}
                    try:
                        json.dumps(usage, allow_nan=False)
                        entry["usage"] = usage
                    except (ValueError, TypeError):
                        entry["usage_invalid"] = True
                    decision = self._decision(data)
                    if self._expired.is_set() or time.monotonic() >= deadline:
                        raise TimeoutError("OpenRouter request deadline exceeded")
                    entry["status"] = "success"
                    return decision
                except Exception as error:
                    entry.update(status="failed", error=self._safe(error))
                    if self._interrupted.is_set():
                        raise InterruptedError("OpenRouter request interrupted") from None
                    if self._expired.is_set() or time.monotonic() >= deadline:
                        raise TimeoutError("OpenRouter request deadline exceeded") from None
                    retryable = isinstance(error, (_RetryableError, http.client.HTTPException, OSError))
                    if retryable and attempt == 0:
                        continue
                    if isinstance(error, TimeoutError):
                        raise TimeoutError("OpenRouter request timed out after one retry") from None
                    detail = "request failed after one retry: " if retryable else ""
                    raise RuntimeError(f"OpenRouter {detail}{self._safe(error)}") from None
            raise AssertionError("unreachable")
        finally:
            timer.cancel()
            timer.join()

    def _request(self, body: bytes, deadline: float) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("OpenRouter request deadline exceeded")
        connection = http.client.HTTPSConnection("openrouter.ai", timeout=remaining)
        self._connection = connection
        try:
            connection.request("POST", "/api/v1/chat/completions", body, {
                "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            if self._interrupted.is_set():
                raise InterruptedError("OpenRouter request interrupted")
            self._set_socket_timeout(connection, deadline)
            response = connection.getresponse()
            chunks = []
            while True:
                self._set_socket_timeout(connection, deadline)
                chunk = response.read1(64 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
            if self._interrupted.is_set():
                raise InterruptedError("OpenRouter request interrupted")
            if response.status in (408, 429) or response.status >= 500:
                raise _RetryableError(f"OpenRouter HTTP {response.status}")
            if response.status != 200:
                raise RuntimeError(f"OpenRouter HTTP {response.status}: {raw.decode(errors='replace')}")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise RuntimeError("OpenRouter returned an invalid response object")
            if data.get("error"):
                error = data["error"]
                code = error.get("code") if isinstance(error, dict) else None
                if code in (408, 429) or isinstance(code, int) and code >= 500:
                    raise _RetryableError(f"OpenRouter: {error}")
                raise RuntimeError(f"OpenRouter: {error}")
            if self._interrupted.is_set():
                raise InterruptedError("OpenRouter request interrupted")
            return data
        except socket.timeout as error:
            raise TimeoutError("OpenRouter request timed out") from error
        finally:
            self._connection = None
            connection.close()

    @staticmethod
    def _set_socket_timeout(connection: http.client.HTTPSConnection, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("OpenRouter request deadline exceeded")
        if connection.sock is not None:
            connection.sock.settimeout(remaining)

    def _decision(self, data: dict[str, Any]) -> Decision:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise _RetryableError("OpenRouter produced no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError("OpenRouter produced an invalid choice")
        finish = choice.get("finish_reason")
        if finish not in ("stop", "tool_calls"):
            raise RuntimeError(f"OpenRouter completion ended with {finish!r}")
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
            raise RuntimeError("OpenRouter produced an invalid assistant message")
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise RuntimeError("OpenRouter assistant content must be text")
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise RuntimeError("OpenRouter tool calls must be a list")
        calls = []
        seen = set()
        for raw in raw_calls:
            if not isinstance(raw, dict) or not isinstance(raw.get("function"), dict):
                raise RuntimeError("OpenRouter produced an invalid tool call")
            function = raw["function"]
            call_id, name = raw.get("id"), function.get("name")
            if raw.get("type") != "function" or not isinstance(call_id, str) or not call_id or call_id in seen:
                raise RuntimeError("OpenRouter produced invalid or duplicate tool-call IDs")
            if not isinstance(name, str) or not name:
                raise RuntimeError("OpenRouter produced an invalid tool name")
            encoded = function.get("arguments")
            if not isinstance(encoded, str):
                raise RuntimeError("OpenRouter tool arguments must be JSON text")
            try:
                args = json.loads(encoded, parse_constant=_reject_constant)
                json.dumps(args, allow_nan=False)
            except (ValueError, TypeError) as error:
                raise RuntimeError("OpenRouter tool arguments are invalid JSON") from error
            if not isinstance(args, dict):
                raise RuntimeError("OpenRouter tool arguments must be an object")
            calls.append(ToolCall(call_id, name, args))
            seen.add(call_id)
        if finish == "tool_calls" and not calls:
            raise RuntimeError("OpenRouter tool completion omitted tool calls")
        if not content.strip() and not calls:
            raise _RetryableError("OpenRouter produced an empty response")
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            raise RuntimeError("OpenRouter usage must be an object")
        json.dumps(usage, allow_nan=False)
        returned_model = data.get("model") or self.model
        if not isinstance(returned_model, str):
            raise RuntimeError("OpenRouter returned an invalid model identity")
        return Decision(content, tuple(calls), returned_model, usage)

    def _expire(self) -> None:
        self._expired.set()
        self._abort_socket()

    def _abort_socket(self) -> None:
        connection = self._connection
        if connection is not None and connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def interrupt(self) -> None:
        self._interrupted.set()
        self._abort_socket()

    def close(self) -> None:
        self.interrupt()
