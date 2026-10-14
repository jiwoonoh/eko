"""Persistent, bounded subprocess transport for the named ECG measurement tool."""

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
    def __init__(self, command=None):
        self.command = list(command) if command is not None else [sys.executable, str(Path(__file__).resolve()), "--serve"]
        self.process = None
        self.stderr_tail = b""
        self._lock = threading.Lock()

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

    def call(self, name, arguments, record, timeout=30.0):
        if name != "call_measurement_tool":
            return empty_result(record, "unavailable_tool", f"Tool is unavailable: {name}")
        if not isinstance(arguments, dict) or set(arguments) != {"record_id"} or arguments["record_id"] != record.record_id:
            return empty_result(record, "invalid_arguments", "Tool arguments must name only the attached record_id")
        if not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ValueError("tool timeout must be positive and finite")
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        request = {"id": request_id, "tool": name, "arguments": arguments, "record": asdict(record)}
        payload = (json.dumps(request, allow_nan=False) + "\n").encode()
        if len(payload) > MAX_REQUEST_BYTES:
            return empty_result(record, "invalid_request", "ECG request exceeds the input limit")
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
                result = empty_result(record, "worker_failed", str(error))
                result["elapsed"] = time.monotonic() - started
                return result

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _dispatch(request):
    from measurements import measure_record

    fields = dict(request["record"])
    fields["lead_names"] = tuple(fields["lead_names"])
    fields["waveform_units"] = tuple(fields["waveform_units"])
    record = Record(**fields)
    if request.get("tool") != "call_measurement_tool" or request.get("arguments") != {"record_id": record.record_id}:
        return empty_result(record, "invalid_arguments", "Only the attached record can be measured")
    try:
        return measure_record(record)
    except (ImportError, ModuleNotFoundError) as error:
        return empty_result(record, "measurement_dependencies_missing", str(error))
    except Exception as error:
        return empty_result(record, "record_validation_failed", str(error))


def serve():
    while True:
        line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            return
        if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
            raise WorkerError("ECG worker input exceeds the input limit")
        request = json.loads(line)
        with redirect_stdout(sys.stderr):
            result = _dispatch(request)
        response = {"id": request["id"], "result": result}
        sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    if sys.argv[1:] != ["--serve"]:
        raise SystemExit("Run worker through ToolWorker")
    serve()
