"""Experimental controls and local credential boundaries, without API requests."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiment import load_config, load_key, run, summarize


class ExperimentTests(unittest.TestCase):
    def test_resource_accounting_includes_failed_and_retried_provider_attempts(self):
        summary = dict.fromkeys(("expected_turns", "completed_answers", "completed_tasks", "answer_rejections",
                                 "provider_errors", "tool_errors", "attempted_tool_calls", "model_responses"), 1)
        summary["total_elapsed_seconds"] = 2
        turn = {"model_responses_trace": [{"usage": {"cost": .2, "prompt_tokens": 20}}],
                "provider_attempts": [{"status": "failed", "usage": {"cost": .1, "prompt_tokens": 10}},
                                      {"status": "success", "usage": {"cost": .2, "prompt_tokens": 20}}]}
        result = summarize([{"summary": summary, "turns": [turn]}])
        self.assertAlmostEqual(result["reported_cost_usd"], .3)
        self.assertEqual(result["prompt_tokens"], 30)

    def test_config_requires_all_explicit_controls_and_no_secrets(self):
        config = load_config(Path("experiments/reporting.json"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for update in ({"repeats": True}, {"api_key": "not-permitted"},
                           {"limits": {**config["limits"], "request_timeout": -1}},
                           {"limits": {**config["limits"], "max_responses": 1.5}}):
                path.write_text(json.dumps({**config, **update}))
                with self.subTest(update=update), self.assertRaises(ValueError):
                    load_config(path)

    def test_key_file_is_data_and_existing_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / ".env"
            path.write_text("UNRELATED=$(do-not-execute)\nexport OPENROUTER_API_KEY='local-test-key'\n")
            load_key(path)
            self.assertEqual(os.environ["OPENROUTER_API_KEY"], "local-test-key")
            path.write_text("OPENROUTER_API_KEY=other-key\n")
            load_key(path)
            self.assertEqual(os.environ["OPENROUTER_API_KEY"], "local-test-key")

    def test_repeat_outputs_are_exclusive_and_record_configuration(self):
        config = load_config(Path("experiments/reporting.json"))
        config["repeats"] = 2
        summary = {key: 0 for key in ("answer_rejections", "provider_errors", "tool_errors", "attempted_tool_calls", "model_responses")}
        summary.update(expected_turns=20, completed_answers=19, completed_tasks=18, total_elapsed_seconds=1)
        report = {"summary": summary, "turns": []}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch("experiment.load_manifest", return_value={}), \
             mock.patch("experiment.prepare", return_value={}), \
             mock.patch("experiment.run_dialogue", return_value=report) as dialogue, \
             mock.patch("builtins.print"):
            first = run(config, "controlled", Path(directory))
            second = run(config, "controlled", Path(directory))
            self.assertNotEqual(first, second)
            self.assertEqual(dialogue.call_count, 4)
            saved = json.loads((first / "summary.json").read_text())
            self.assertEqual(saved["results"]["tasks_by_repeat"], [18, 18])
            self.assertEqual(saved["results"]["task_completion_rate"], .9)
            self.assertEqual(saved["config"], config)
            self.assertTrue((first / "repeat-2.json").is_file())
        with self.assertRaises(ValueError):
            run(config, "../invalid", Path("unused"))
