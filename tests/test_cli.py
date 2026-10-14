import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from contracts import Record, STANDARD_LEADS
from host import lead_order
from traces import EventLog


class CliTests(unittest.TestCase):
    def test_help_runs_without_site_packages(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-S", str(root / "host.py"), "--help"],
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--ecg", result.stdout)
        self.assertNotIn("--sandbox", result.stdout)

    def test_layout_requires_explicit_confirmation(self):
        self.assertIsNone(lead_order(None))
        self.assertEqual(lead_order("standard"), STANDARD_LEADS)
        self.assertEqual(lead_order("II, I"), ("II", "I"))

    def test_colliding_timestamps_do_not_overwrite_logs(self):
        record = Record("/private/example/HR25.mat", "ptbxl:25", "abc", 500,
                        STANDARD_LEADS, ("unknown",) * 12)
        with tempfile.TemporaryDirectory() as directory:
            with EventLog(directory, record, {"model": "test"}) as first:
                first("turn_finished", status="failed")
            with EventLog(directory, record, {"model": "test"}) as second:
                second("turn_finished", status="success")
            self.assertNotEqual(first.path, second.path)
            events = [json.loads(line) for line in first.path.read_text().splitlines()]
            self.assertEqual([event["sequence"] for event in events], [1, 2])
            self.assertEqual(events[-1]["status"], "failed")
            self.assertNotIn("path", events[0]["record"])
            self.assertEqual(first.path.stat().st_mode & 0o777, 0o600)

    def test_nonfinite_events_are_not_written(self):
        record = Record("HR1.mat", "ptbxl:1", "abc", 500,
                        STANDARD_LEADS, ("unknown",) * 12)
        with tempfile.TemporaryDirectory() as directory:
            with EventLog(directory, record, {}) as log:
                with self.assertRaises(ValueError):
                    log("tool_result", value=float("nan"))
            events = log.path.read_text().splitlines()
            self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
