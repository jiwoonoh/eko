import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from contracts import STANDARD_LEADS
from measurements import _estimate, measure_record
from records import load_record

try:
    import numpy as np
    import neurokit2 as nk
    from scipy.io import savemat
    SCIENCE_AVAILABLE = True
except ImportError:
    SCIENCE_AVAILABLE = False


@unittest.skipUnless(SCIENCE_AVAILABLE, "install the measurement extra")
class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "HR00025.mat"

    def save(self, signal=None, frequency=500, **metadata):
        if signal is None:
            signal = np.tile(np.sin(np.arange(5000) / 25), (12, 1))
        savemat(self.path, {"feats": signal, "curr_sample_rate": frequency, **metadata})
        return load_record(self.path, lead_order=STANDARD_LEADS)[0]

    def test_metadata_hash_and_explicit_lead_confirmation(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "confirm"):
            load_record(self.path)
        record, array = load_record(self.path, lead_order=STANDARD_LEADS)
        self.assertEqual(record.record_id, "ptbxl:25")
        self.assertEqual(record.sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(array.shape, (12, 5000))
        self.assertEqual(record.waveform_units, ("unknown",) * 12)
        self.save(lead_names=np.array(STANDARD_LEADS, dtype=object), units="mV")
        record, _ = load_record(self.path)
        self.assertEqual(record.lead_names, STANDARD_LEADS)
        self.assertEqual(record.waveform_units, ("mV",) * 12)

    def test_frequency_and_shape_are_not_guessed(self):
        for frequency in (0, -1, float("nan"), [250, 500]):
            with self.subTest(frequency=frequency):
                savemat(self.path, {"feats": np.zeros((12, 20)), "curr_sample_rate": frequency})
                with self.assertRaises(ValueError):
                    load_record(self.path, lead_order=STANDARD_LEADS)
        savemat(self.path, {"feats": np.zeros((5000, 12)), "curr_sample_rate": 500})
        with self.assertRaisesRegex(ValueError, "shape"):
            load_record(self.path, lead_order=STANDARD_LEADS)

    def test_changed_record_and_nonfinite_samples_are_rejected(self):
        record = self.save()
        self.save(frequency=250)
        with self.assertRaisesRegex(ValueError, "changed"):
            measure_record(record)
        samples = np.zeros((12, 100))
        samples[1, 5] = np.nan
        savemat(self.path, {"feats": samples, "curr_sample_rate": 500})
        with self.assertRaisesRegex(ValueError, "finite"):
            load_record(self.path, lead_order=STANDARD_LEADS)

    def test_lead_metadata_cannot_conflict_with_confirmed_order(self):
        reordered = list(STANDARD_LEADS)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        savemat(self.path, {"feats": np.zeros((12, 100)), "curr_sample_rate": 500,
                            "lead_names": np.array(reordered, dtype=object)})
        with self.assertRaisesRegex(ValueError, "disagrees"):
            load_record(self.path, lead_order=STANDARD_LEADS)
        record, _ = load_record(self.path)
        self.assertEqual(record.lead_names.index("II"), 0)

    def test_explicit_reordered_leads_select_lead_ii(self):
        reordered = list(STANDARD_LEADS)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        signal = np.zeros((12, 5000))
        signal[0] = np.sin(np.arange(5000) / 25)
        savemat(self.path, {"feats": signal, "curr_sample_rate": 500})
        record, _ = load_record(self.path, lead_order=reordered)
        peaks = np.arange(1, 9) * 500
        with mock.patch.object(nk, "ecg_clean", side_effect=lambda x, **_: x) as clean, \
             mock.patch.object(nk, "ecg_peaks", return_value=(None, {"ECG_R_Peaks": peaks})), \
             mock.patch.object(nk, "ecg_delineate", return_value=(None, {})):
            result = measure_record(record)
        np.testing.assert_equal(clean.call_args.args[0], signal[0])
        self.assertEqual(result["findings"]["heart_rate_bpm"], 60)
        self.assertEqual(result["details"]["lead_used"], "II")
        with self.assertRaises(ValueError):
            load_record(self.path, lead_order=reordered[:-1])

    def test_flat_signal_and_insufficient_peaks_keep_failure_status(self):
        record = self.save(signal=np.ones((12, 1000)))
        result = measure_record(record)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["missingness"]["heart_rate_bpm"], "flat_signal")
        record = self.save()
        with mock.patch.object(nk, "ecg_clean", side_effect=lambda x, **_: x), \
             mock.patch.object(nk, "ecg_peaks", return_value=(None, {"ECG_R_Peaks": [10, 510]})):
            result = measure_record(record)
        self.assertEqual(result["status"], "partial")
        self.assertTrue(all(value is None for value in result["findings"].values()))
        json.dumps(result, allow_nan=False)

    def test_sampling_rate_controls_intervals(self):
        for frequency in (250, 500):
            with self.subTest(frequency=frequency):
                peaks = np.arange(1, 9) * frequency
                waves = {"ECG_P_Onsets": peaks - .15 * frequency, "ECG_R_Onsets": peaks,
                         "ECG_Q_Peaks": peaks - .04 * frequency, "ECG_S_Peaks": peaks + .06 * frequency,
                         "ECG_T_Offsets": peaks + .36 * frequency}
                findings, missing, _, _ = _estimate(peaks, waves, frequency)
                self.assertEqual(findings["rr_interval_ms"], 1000)
                self.assertEqual(findings["heart_rate_bpm"], 60)
                self.assertAlmostEqual(findings["pr_interval_ms"], 150 + 1000 / frequency)
                self.assertFalse(missing)

    def test_missing_middle_qt_does_not_shift_rr_pairing(self):
        frequency = 500
        peaks = np.array([0, 500, 1250, 1750, 2250, 2750, 3250, 3750])
        waves = {"ECG_P_Onsets": peaks - 75, "ECG_R_Onsets": peaks,
                 "ECG_Q_Peaks": peaks - 20, "ECG_S_Peaks": peaks + 30,
                 "ECG_T_Offsets": peaks + 180.0}
        waves["ECG_T_Offsets"][1] = np.nan
        _, _, estimates, _ = _estimate(peaks, waves, frequency)
        qtc = estimates["qtc_bazett_ms"]
        self.assertEqual(qtc[1]["reason"], "missing_landmarks")
        self.assertAlmostEqual(qtc[2]["value"], 402 / np.sqrt(1.5))
        self.assertAlmostEqual(qtc[3]["value"], 402)

    def test_filters_preserve_raw_estimates_and_partial_status(self):
        record = self.save()
        peaks = np.arange(1, 9) * 500
        waves = {"ECG_P_Onsets": peaks - 75, "ECG_R_Onsets": peaks,
                 "ECG_Q_Peaks": peaks - 10, "ECG_S_Peaks": peaks + 10,
                 "ECG_T_Offsets": peaks + 180}
        with mock.patch.object(nk, "ecg_clean", side_effect=lambda x, **_: x), \
             mock.patch.object(nk, "ecg_peaks", return_value=(None, {"ECG_R_Peaks": peaks})), \
             mock.patch.object(nk, "ecg_delineate", return_value=(None, waves)):
            result = measure_record(record)
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["findings"]["q_s_peak_span_ms"])
        self.assertEqual(result["details"]["per_beat"]["q_s_peak_span_ms"][0]["value"], 42)
        self.assertEqual(result["details"]["filtering"]["q_s_peak_span_ms"]["excluded"], 8)
        self.assertNotIn("Interpretation", json.dumps(result))
        json.dumps(result, allow_nan=False)

    def test_real_synthetic_signal_runs_at_250_and_500_hz(self):
        for frequency in (250, 500):
            with self.subTest(frequency=frequency):
                lead = nk.ecg_simulate(duration=10, sampling_rate=frequency, heart_rate=80, random_state=42)
                record = self.save(signal=np.tile(lead, (12, 1)), frequency=frequency)
                result = measure_record(record)
                self.assertIn(result["status"], {"success", "partial"})
                self.assertAlmostEqual(result["findings"]["heart_rate_bpm"], 80, delta=5)
                self.assertEqual(result["details"]["sampling_rate_hz"], frequency)
                json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
