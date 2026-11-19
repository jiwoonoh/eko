from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from toolbox.measurements import ecg_analysis
from toolbox.measurements.preprocessing import Beats
from toolbox.registry import Toolbox


class MeasurementToolsTest(TestCase):
    def test_individual_selection(self):
        tools = Toolbox(Namespace(tools=["heart_rate", "qtc_interval"], leads="single-lead-I"))
        with TemporaryDirectory() as directory, ExitStack() as resources:
            root = Path(directory)
            tools.start(resources, root, root, sandbox=False)
            modules = root / "toolbox" / "measurements"
            self.assertIn("using lead I", tools.prompt)
            self.assertTrue((modules / "heart_rate.py").is_file())
            self.assertTrue((modules / "qtc_interval.py").is_file())
            self.assertFalse((modules / "pr_interval.py").exists())
            self.assertFalse((modules / "ecg_analysis.py").exists())

    def test_fixed_panel_shares_beats_and_preserves_precision(self):
        peaks = np.arange(500, 4000, 500)
        beats = Beats(np.zeros(4000), 500, peaks)
        beats.waves = {
            "ECG_P_Onsets": peaks - 100, "ECG_R_Onsets": peaks - 20,
            "ECG_Q_Peaks": peaks - 16, "ECG_S_Peaks": peaks + 25, "ECG_T_Offsets": peaks + 180,
        }
        with patch("toolbox.measurements.preprocessing.prepare_beats", return_value=beats) as prepare:
            results = ecg_analysis("record.mat")
        prepare.assert_called_once_with("record.mat")
        self.assertEqual(results, {
            "heart_rate": "60.00", "pr_interval": "162", "qrs_duration": "84", "qtc_interval": "394.00",
        })
