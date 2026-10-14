import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from contracts import Record, STANDARD_LEADS
from worker import MAX_RESULT_BYTES, MAX_STDERR_BYTES, ToolWorker

FIXTURE = """
import json, os, sys
count = 0
for line in sys.stdin:
    request = json.loads(line)
    count += 1
    result = {'record_id': request['record']['record_id'], 'record_hash': request['record']['sha256'],
              'status': 'success', 'findings': {}, 'details': {'pid': os.getpid(), 'calls': count,
              'secret_present': any(x in os.environ for x in ['OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'HF_TOKEN'])}}
    print(json.dumps({'id': request['id'], 'result': result}), flush=True)
"""


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.record = Record("/unopened/HR25.mat", "ptbxl:25", "a" * 64, 500, STANDARD_LEADS, ("unknown",) * 12)
        self.arguments = {"record_id": self.record.record_id}

    def worker(self, script=FIXTURE):
        worker = ToolWorker([sys.executable, "-u", "-c", script])
        self.addCleanup(worker.close)
        return worker

    def call(self, worker, timeout=2):
        return worker.call("call_measurement_tool", self.arguments, self.record, timeout)

    def test_worker_reuses_process_and_close_allows_lazy_restart(self):
        worker = self.worker()
        first, second = self.call(worker), self.call(worker)
        self.assertEqual(first["details"]["pid"], second["details"]["pid"])
        self.assertEqual(second["details"]["calls"], 2)
        old_process = worker.process
        worker.close()
        self.assertIsNotNone(old_process.poll())
        third = self.call(worker)
        self.assertEqual(third["details"]["calls"], 1)
        self.assertNotEqual(first["details"]["pid"], third["details"]["pid"])

    def test_api_keys_are_removed_from_child_environment(self):
        worker = self.worker()
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "fixture", "OPENAI_API_KEY": "fixture", "HF_TOKEN": "fixture"}):
            self.assertFalse(self.call(worker)["details"]["secret_present"])

    def test_unknown_tool_or_record_arguments_never_start_worker(self):
        worker = self.worker()
        for name, arguments in [("python", self.arguments), ("call_measurement_tool", {"record_id": "ptbxl:26"}),
                                ("call_measurement_tool", {**self.arguments, "path": "/etc/passwd"})]:
            result = worker.call(name, arguments, self.record, 1)
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(worker.process)

    def test_timeout_kills_and_reaps_process_group(self):
        worker = self.worker("import time\ntime.sleep(60)")
        worker._start()
        process = worker.process
        started = time.monotonic()
        with mock.patch("worker.os.killpg", wraps=os.killpg) as kill_group:
            with self.assertRaises(TimeoutError):
                self.call(worker, timeout=.15)
        kill_group.assert_called_once()
        self.assertEqual(kill_group.call_args.args[0], process.pid)
        self.assertIsNotNone(process.poll())
        self.assertIsNone(worker.process)
        self.assertLess(time.monotonic() - started, 2)
        worker.command = [sys.executable, "-u", "-c", FIXTURE]
        self.assertEqual(self.call(worker)["status"], "success")

    def test_keyboard_interrupt_reaps_worker_and_can_restart(self):
        worker = self.worker()
        worker._start()
        process = worker.process
        with mock.patch.object(worker, "_exchange", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.call(worker)
        self.assertIsNotNone(process.poll())
        self.assertIsNone(worker.process)
        self.assertEqual(self.call(worker)["status"], "success")

    def test_crash_returns_failure_and_next_call_restarts(self):
        worker = self.worker(FIXTURE.replace("count += 1", "count += 1\n    if count == 2: os._exit(7)"))
        self.assertEqual(self.call(worker)["status"], "success")
        self.assertEqual(self.call(worker)["status"], "failed")
        self.assertIsNone(worker.process)
        self.assertEqual(self.call(worker)["details"]["calls"], 1)

    def test_stderr_and_stdout_have_limits(self):
        worker = self.worker(FIXTURE.replace("count += 1", f"count += 1\n    sys.stderr.write('x' * {MAX_STDERR_BYTES * 8})\n    sys.stderr.flush()"))
        self.assertEqual(self.call(worker)["status"], "success")
        self.assertLessEqual(len(worker.stderr_tail), MAX_STDERR_BYTES)
        noisy = self.worker(f"import sys\nsys.stdout.write('x' * {MAX_RESULT_BYTES + 1})\nsys.stdout.flush()")
        result = self.call(noisy)
        self.assertEqual(result["status"], "failed")
        self.assertIn("output limit", result["details"]["message"])
        self.assertIsNone(noisy.process)

    def test_real_worker_measures_selected_record_and_detects_changed_bytes(self):
        try:
            import numpy as np
            import neurokit2 as nk
            from scipy.io import savemat
        except ImportError:
            self.skipTest("install the measurement extra")
        from records import load_record

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "HR25.mat"
        lead = nk.ecg_simulate(duration=10, sampling_rate=250, heart_rate=80, random_state=42)
        savemat(path, {"feats": np.tile(lead, (12, 1)), "curr_sample_rate": 250})
        record, _ = load_record(path, lead_order=STANDARD_LEADS)
        worker = ToolWorker()
        self.addCleanup(worker.close)
        result = worker.call("call_measurement_tool", {"record_id": record.record_id}, record, 30)
        self.assertIn(result["status"], {"success", "partial"}, result)
        self.assertEqual(result["record_hash"], record.sha256)
        self.assertAlmostEqual(result["findings"]["heart_rate_bpm"], 80, delta=5)
        pid = worker.process.pid
        savemat(path, {"feats": np.zeros((12, 2500)), "curr_sample_rate": 250})
        changed = worker.call("call_measurement_tool", {"record_id": record.record_id}, record, 30)
        self.assertEqual(changed["status"], "failed")
        self.assertEqual(changed["details"]["reason"], "record_validation_failed")
        self.assertEqual(worker.process.pid, pid)


if __name__ == "__main__":
    unittest.main()
