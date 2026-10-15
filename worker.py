"""Persistent, bounded subprocess transport for registered scientific tools."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import threading
import time
import uuid

from contracts import Record
from measurements import empty_result

MAX_REQUEST_BYTES = 64 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024
MAX_STDERR_BYTES = 32 * 1024
SECRET_NAME = re.compile(r"API_?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


class WorkerError(RuntimeError):
    pass


class ToolWorker:
    def __init__(self, command=None, *, checkpoint=None):
        self.command = list(command) if command is not None else [sys.executable, str(Path(__file__).resolve()), "--serve"]
        self.process = None
        self.stderr_tail = b""
        self._lock = threading.Lock()
        self.checkpoint = str(Path(checkpoint).expanduser().resolve()) if checkpoint is not None else None
        self.classifier_ready = False
        self.classifier_error = None
        self.classifier_provenance = None

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        environment = {name: value for name, value in os.environ.items() if not SECRET_NAME.search(name)}
        environment["PYTHONUNBUFFERED"] = "1"
        environment["MPLBACKEND"] = "Agg"
        environment.setdefault("OPENBLAS_NUM_THREADS", "1")
        environment.setdefault("OMP_NUM_THREADS", "1")
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0, env=environment, start_new_session=True)
        self.stderr_tail = b""

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Some macOS sandboxes deny killpg after its leader has exited.
            # Reap that leader; a live process still receives a direct kill.
            if process.poll() is None:
                process.kill()
        process.wait(timeout=2)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()

    def _exchange(self, payload, deadline):
        process = self.process
        output = bytearray()
        position = 0
        with selectors.DefaultSelector() as selector:
            for stream in (process.stdin, process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE, "input")
            selector.register(process.stdout, selectors.EVENT_READ, "output")
            selector.register(process.stderr, selectors.EVENT_READ, "error")
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("ECG tool exceeded its hard timeout")
                for key, _ in selector.select(remaining):
                    if key.data == "input":
                        try:
                            position += os.write(key.fd, payload[position:])
                        except BlockingIOError:
                            continue
                        if position == len(payload):
                            selector.unregister(key.fileobj)
                        continue
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        if key.data == "output":
                            raise WorkerError("ECG worker exited without a result")
                        continue
                    if key.data == "error":
                        self.stderr_tail = (self.stderr_tail + chunk)[-MAX_STDERR_BYTES:]
                        continue
                    output.extend(chunk)
                    if len(output) > MAX_RESULT_BYTES:
                        raise WorkerError("ECG worker result exceeds the output limit")
                    if b"\n" in output:
                        line, extra = output.split(b"\n", 1)
                        if extra.strip():
                            raise WorkerError("ECG worker emitted unexpected extra output")
                        try:
                            return json.loads(line)
                        except (ValueError, UnicodeError) as error:
                            raise WorkerError("ECG worker returned invalid JSON") from error

    def prepare_classifier(self, record, timeout=30.0):
        """Verify configured weights, input compatibility, and CPU inference in the worker."""
        self.classifier_ready = False
        self.classifier_provenance = None
        if self.checkpoint is None:
            self.classifier_error = "no classifier checkpoint configured"
            return False
        result = self._invoke("__prepare_classifier__", {"record_id": record.record_id}, record, timeout)
        from classifier import provenance
        self.classifier_ready = result.get("status") == "ready" and result.get("provenance") == provenance()
        self.classifier_provenance = result.get("provenance") if self.classifier_ready else None
        self.classifier_error = None if self.classifier_ready else result.get("details", {}).get("message", "classifier readiness failed")
        return self.classifier_ready

    def call(self, name, arguments, record, timeout=30.0):
        if name not in ("call_measurement_tool", "call_classification_tool"):
            return empty_result(record, "unavailable_tool", f"Tool is unavailable: {name}")
        if name == "call_classification_tool" and not self.classifier_ready:
            return _failure(record, name, "classifier_unavailable", self.classifier_error or "classifier readiness has not been checked")
        if not isinstance(arguments, dict) or set(arguments) != {"record_id"} or arguments["record_id"] != record.record_id:
            return _failure(record, name, "invalid_arguments", "Tool arguments must name only the attached record_id")
        return self._invoke(name, arguments, record, timeout)

    def _invoke(self, name, arguments, record, timeout):
        if not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ValueError("tool timeout must be positive and finite")
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        request = {"id": request_id, "tool": name, "arguments": arguments, "record": asdict(record)}
        if name in ("__prepare_classifier__", "call_classification_tool"):
            request["checkpoint"] = self.checkpoint
        payload = (json.dumps(request, allow_nan=False) + "\n").encode()
        if len(payload) > MAX_REQUEST_BYTES:
            return _failure(record, name, "invalid_request", "ECG request exceeds the input limit")
        with self._lock:
            try:
                self._start()
                response = self._exchange(payload, started + timeout)
                if not isinstance(response, dict) or response.get("id") != request_id or not isinstance(response.get("result"), dict):
                    raise WorkerError("ECG worker response does not match the request")
                result = response["result"]
                result["elapsed"] = time.monotonic() - started
                return result
            except (TimeoutError, KeyboardInterrupt):
                self.close()
                raise
            except (WorkerError, OSError, ValueError) as error:
                self.close()
                result = _failure(record, name, "worker_failed", str(error))
                result["elapsed"] = time.monotonic() - started
                return result

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _failure(record, name, reason, message):
    if name == "call_measurement_tool":
        return empty_result(record, reason, message)
    return {"record_id": record.record_id, "record_hash": record.sha256,
            "status": "failed", "version": "worker", "source": "runner", "findings": {},
            "units": {}, "methods": {}, "missingness": {},
            "details": {"reason": reason, "message": message}, "elapsed": 0.0}


def _dispatch(request, cache=None):
    from measurements import measure_record

    fields = dict(request["record"])
    fields["lead_names"] = tuple(fields["lead_names"])
    fields["waveform_units"] = tuple(fields["waveform_units"])
    record = Record(**fields)
    name = request.get("tool")
    if name not in ("call_measurement_tool", "call_classification_tool", "__prepare_classifier__") or request.get("arguments") != {"record_id": record.record_id}:
        return _failure(record, name, "invalid_arguments", "Only registered tools on the attached record are accepted")
    try:
        if name in ("call_classification_tool", "__prepare_classifier__"):
            from classifier import Classifier, preprocess
            if not request.get("checkpoint"):
                raise ValueError("no classifier checkpoint configured")
            if cache is None:
                cache = {}
            if "classifier" not in cache:
                cache["classifier"] = Classifier(request["checkpoint"])
                cache["checkpoint"] = request["checkpoint"]
            if cache["checkpoint"] != request["checkpoint"]:
                raise ValueError("restart the worker to change classifier checkpoints")
            if name == "__prepare_classifier__":
                preprocess(record)
                return {"status": "ready", "provenance": cache["classifier"].provenance()}
            return cache["classifier"].classify(record)
        return measure_record(record)
    except (ImportError, ModuleNotFoundError) as error:
        reason = "measurement_dependencies_missing" if name == "call_measurement_tool" else "classifier_dependencies_missing"
        return _failure(record, name, reason, str(error))
    except Exception as error:
        reason = "record_validation_failed" if name == "call_measurement_tool" else "classifier_validation_failed"
        return _failure(record, name, reason, str(error))


def serve():
    cache = {}
    while True:
        line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            return
        if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
            raise WorkerError("ECG worker input exceeds the input limit")
        request = json.loads(line)
        with redirect_stdout(sys.stderr):
            result = _dispatch(request, cache)
        response = {"id": request["id"], "result": result}
        sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    if sys.argv[1:] != ["--serve"]:
        raise SystemExit("Run worker through ToolWorker")
    serve()
