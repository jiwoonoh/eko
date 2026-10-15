from contextlib import nullcontext
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import classifier
from contracts import STANDARD_LEADS

try:
    import numpy as np
    from scipy.io import savemat
    SCIENCE_AVAILABLE = True
except ImportError:
    SCIENCE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def observation():
    keys = [classifier.score_key(index) for index in range(150)]
    return {
        "record_id": "ptbxl:705", "record_hash": "a" * 64,
        "version": classifier.VERSION, "status": "success", "source": "live",
        "findings": {key: index / 149 for index, key in enumerate(keys)},
        "units": dict.fromkeys(keys, "score"),
        "methods": dict.fromkeys(keys, classifier.SCORE_METHOD),
        "missingness": {}, "details": classifier.provenance(), "elapsed": 0,
    }


class ManifestTests(unittest.TestCase):
    def test_core_import_does_not_import_scientific_packages(self):
        code = "import classifier,sys; assert not ({'torch','numpy','scipy','ecgfounder'} & set(sys.modules))"
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(classifier.__file__).parent,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_manifest_identity_and_exact_distinct_label_meanings(self):
        labels = classifier.load_labels()
        self.assertEqual(len(labels), 150)
        self.assertEqual(len(set(labels)), 150)
        self.assertEqual(classifier.label_for_key("ecgfounder_001"), "NORMAL SINUS RHYTHM")
        self.assertEqual(classifier.label_for_key("ecgfounder_002"), "NORMAL ECG")
        self.assertEqual(classifier.label_for_key("ecgfounder_003"), "SINUS RHYTHM")
        self.assertEqual(classifier.label_for_key("ecgfounder_005"), "ATRIAL FIBRILLATION")
        self.assertEqual(classifier.label_for_key("ecgfounder_036"), "LEFT ANTERIOR FASCICULAR BLOCK")

    def test_manifest_corruption_is_rejected(self):
        with mock.patch.object(classifier, "LABEL_BYTES", b"ATRIAL FIBRILLATION\n"):
            with self.assertRaisesRegex(ValueError, "checksum"):
                classifier.load_labels()

    def test_invalid_label_keys_and_indices_fail_closed(self):
        for index in (-1, 150, True, 1.0, "5"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                classifier.score_key(index)
        for key in ("ecgfounder_150", "ecgfounder_5", "ecgfounder_005x", "ecgfounder_-01", 5):
            with self.subTest(key=key), self.assertRaises(ValueError):
                classifier.label_for_key(key)

    def test_complete_numeric_output_is_json_safe(self):
        raw = observation()
        self.assertIs(classifier.validate_output(raw), raw)
        json.dumps(raw, allow_nan=False)

    def test_missing_extra_nonfinite_and_out_of_range_scores_rejected(self):
        changes = [("remove", None), ("extra", None), ("score", None), ("score", True),
                   ("score", float("nan")), ("score", float("inf")), ("score", -0.1),
                   ("score", 1.1), ("score", "0.2")]
        for change, value in changes:
            raw = observation()
            if change == "remove":
                del raw["findings"]["ecgfounder_005"]
            elif change == "extra":
                raw["findings"]["afib"] = 0.3
            else:
                raw["findings"]["ecgfounder_005"] = value
            with self.subTest(change=change, value=value), self.assertRaises(ValueError):
                classifier.validate_output(raw)

    def test_metadata_units_methods_and_missingness_must_match(self):
        for field in ("checkpoint_sha256", "source_revision", "model_revision", "label_map_sha256",
                      "preprocessing_version", "backend", "device", "task_count", "output_transform"):
            raw = observation()
            raw["details"][field] = "changed"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "provenance"):
                classifier.validate_output(raw)
        for field in ("units", "methods"):
            raw = observation()
            raw[field]["ecgfounder_005"] = "invented"
            with self.subTest(field=field), self.assertRaises(ValueError):
                classifier.validate_output(raw)
            del raw[field]["ecgfounder_005"]
            with self.assertRaises(ValueError):
                classifier.validate_output(raw)
        for field, value in (("version", "legacy"), ("status", "partial"), ("source", "replay"),
                             ("missingness", {"ecgfounder_005": "unknown"})):
            raw = observation()
            raw[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                classifier.validate_output(raw)


@unittest.skipUnless(SCIENCE_AVAILABLE, "install the ecg extra")
class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "HR00705.mat"
        self.waveform = np.arange(60000, dtype=np.float64).reshape(12, 5000) / 1000

    def save(self, waveform=None, frequency=500, leads=STANDARD_LEADS, units="mV"):
        from records import load_record

        savemat(self.path, {"feats": self.waveform if waveform is None else waveform,
                            "curr_sample_rate": frequency, "lead_names": np.array(leads, dtype=object),
                            "waveform_units": units})
        return load_record(self.path)[0]

    def test_global_z_score_float32_and_exact_500hz_window(self):
        result = classifier.preprocess(self.save())
        expected = ((self.waveform - self.waveform.mean()) / (self.waveform.std() + 1e-8)).astype(np.float32)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (12, 5000))
        self.assertTrue(result.flags.c_contiguous)
        self.assertGreater(abs(float(result[0].mean())), 0.5)

    def test_reordered_metadata_is_canonicalized_before_inference(self):
        order = [1, 0, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]
        record = self.save(self.waveform[order], leads=tuple(STANDARD_LEADS[i] for i in order))
        actual = classifier.preprocess(record)
        expected = classifier.preprocess(self.save())
        np.testing.assert_array_equal(actual, expected)

    def test_wrong_frequency_length_units_or_flat_signal_are_not_guessed(self):
        cases = [dict(frequency=250), dict(waveform=self.waveform[:, :4999]),
                 dict(waveform=self.waveform[:, :500]), dict(units="unknown"), dict(units="uV"),
                 dict(waveform=np.zeros((12, 5000)))]
        for kwargs in cases:
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                classifier.preprocess(self.save(**kwargs))

    def test_changed_record_descriptors_and_bytes_are_rejected(self):
        from dataclasses import replace

        record = self.save()
        for altered in (replace(record, sha256="b" * 64), replace(record, record_id="ptbxl:1"),
                        replace(record, sampling_rate_hz=250), replace(record, waveform_units=("unknown",) * 12)):
            with self.subTest(altered=altered), self.assertRaisesRegex(ValueError, "changed"):
                classifier.preprocess(altered)
        self.save(waveform=self.waveform + 1)
        with self.assertRaisesRegex(ValueError, "changed"):
            classifier.preprocess(record)

    def test_nonfinite_source_is_rejected_before_model_execution(self):
        waveform = self.waveform.copy()
        waveform[1, 5] = np.nan
        savemat(self.path, {"feats": waveform, "curr_sample_rate": 500,
                            "lead_names": np.array(STANDARD_LEADS, dtype=object), "waveform_units": "mV"})
        from contracts import Record

        record = Record(str(self.path), "ptbxl:705", "a" * 64, 500, STANDARD_LEADS, ("mV",) * 12)
        with self.assertRaisesRegex(ValueError, "finite"):
            classifier.preprocess(record)


class CheckpointTests(unittest.TestCase):
    def torch_mock(self, checkpoint):
        return types.SimpleNamespace(load=mock.Mock(return_value=checkpoint),
                                     serialization=types.SimpleNamespace(safe_globals=mock.Mock(return_value=nullcontext())))

    def test_only_verified_same_bytes_reach_torch(self):
        payload = b"verified checkpoint fixture"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            path.write_bytes(payload)
            with mock.patch.object(classifier, "CHECKPOINT_BYTES", len(payload)), \
                 mock.patch.object(classifier, "CHECKPOINT_SHA256", hashlib.sha256(payload).hexdigest()):
                self.assertEqual(classifier._checkpoint_bytes(path), payload)
                path.write_bytes(payload + b"extra")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    classifier._checkpoint_bytes(path)
                path.write_bytes(b"x" * len(payload))
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    classifier._checkpoint_bytes(path)

    @unittest.skipUnless(SCIENCE_AVAILABLE, "install the classifier extra")
    def test_model_loading_is_cpu_weights_only_and_strict(self):
        model = mock.Mock()
        torch_mock = self.torch_mock({"state_dict": {"dense.weight": "tensor"}})
        factory = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {"torch": torch_mock, "ecgfounder": types.SimpleNamespace(Net1D=factory)}):
            actual = classifier._load_model(b"verified fixture")
        self.assertIs(actual, model)
        self.assertEqual(torch_mock.load.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
        self.assertEqual(torch_mock.load.call_args.args[0].getvalue(), b"verified fixture")
        model.load_state_dict.assert_called_once_with({"dense.weight": "tensor"}, strict=True)
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()
        model.requires_grad_.assert_called_once_with(False)
        self.assertEqual(factory.call_args.kwargs["n_classes"], 150)
        self.assertEqual(torch_mock.serialization.safe_globals.call_args.args[0],
                         [np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))])

    @unittest.skipUnless(SCIENCE_AVAILABLE, "install the classifier extra")
    def test_missing_state_dict_and_strict_load_failure_are_not_hidden(self):
        for value in ({}, {"state_dict": []}, "checkpoint"):
            torch_mock = self.torch_mock(value)
            with mock.patch.dict(sys.modules, {"torch": torch_mock, "ecgfounder": types.SimpleNamespace(Net1D=mock.Mock())}):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    classifier._load_model(b"fixture")
        model = mock.Mock()
        model.load_state_dict.side_effect = RuntimeError("missing classifier head")
        torch_mock = self.torch_mock({"state_dict": {}})
        with mock.patch.dict(sys.modules, {"torch": torch_mock, "ecgfounder": types.SimpleNamespace(Net1D=mock.Mock(return_value=model))}):
            with self.assertRaisesRegex(RuntimeError, "classifier head"):
                classifier._load_model(b"fixture")

    def test_readiness_requires_a_successful_forward(self):
        torch_mock = types.SimpleNamespace(zeros=mock.Mock(return_value="zeros"), float32="float32")
        with mock.patch.object(classifier, "_checkpoint_bytes", return_value=b"fixture"), \
             mock.patch.object(classifier, "_load_model", return_value="model"), \
             mock.patch.object(classifier, "_scores", return_value=[0.5] * 150) as scores, \
             mock.patch.dict(sys.modules, {"torch": torch_mock}):
            backend = classifier.Classifier("fixture.pth")
            self.assertTrue(backend.ready)
            scores.assert_called_once_with("model", "zeros")
            torch_mock.zeros.assert_called_once_with((1, 12, 5000), dtype="float32", device="cpu")
        with mock.patch.object(classifier, "_checkpoint_bytes", return_value=b"fixture"), \
             mock.patch.object(classifier, "_load_model", return_value="model"), \
             mock.patch.object(classifier, "_scores", side_effect=ValueError("149 logits")), \
             mock.patch.dict(sys.modules, {"torch": torch_mock}):
            with self.assertRaisesRegex(ValueError, "149 logits"):
                classifier.Classifier("fixture.pth")


@unittest.skipUnless(TORCH_AVAILABLE, "install the classifier extra")
class TorchBoundaryTests(unittest.TestCase):
    def test_output_shape_finiteness_and_sigmoid_are_checked(self):
        batch = torch.zeros((1, 12, 5000))
        for logits in (torch.zeros(1, 149), torch.zeros(1, 151), torch.zeros(2, 150),
                       torch.full((1, 150), float("nan")), torch.full((1, 150), float("inf"))):
            with self.subTest(shape=tuple(logits.shape)), self.assertRaises(ValueError):
                classifier._scores(lambda _: logits, batch)
        values = classifier._scores(lambda _: torch.linspace(-2, 2, 150).reshape(1, 150), batch)
        self.assertEqual(len(values), 150)
        self.assertGreater(values[-1], values[0])
        self.assertTrue(all(0 < value < 1 for value in values))

    def test_attributed_architecture_runs_without_training_dependencies(self):
        from ecgfounder import Net1D

        model = Net1D(in_channels=12, base_filters=4, ratio=1, filter_list=[4], m_blocks_list=[1],
                      kernel_size=3, stride=2, groups_width=2, n_classes=150, use_bn=False, use_do=False)
        model.eval()
        self.assertEqual(tuple(model(torch.zeros(1, 12, 5000)).shape), (1, 150))

    @unittest.skipUnless(SCIENCE_AVAILABLE, "install the ecg extra")
    def test_classify_binds_current_record_and_preserves_all_scores(self):
        from contracts import Record

        backend = object.__new__(classifier.Classifier)
        backend.ready = True
        backend.model = lambda _: torch.linspace(-2, 2, 150).reshape(1, 150)
        record = Record("fixture.mat", "ptbxl:705", "a" * 64, 500, STANDARD_LEADS, ("mV",) * 12)
        with mock.patch.object(classifier, "preprocess", return_value=np.zeros((12, 5000), dtype=np.float32)):
            raw = backend.classify(record)
        self.assertEqual(raw["record_id"], record.record_id)
        self.assertEqual(raw["record_hash"], record.sha256)
        self.assertEqual(len(raw["findings"]), 150)
        self.assertEqual(raw["details"]["lead_order"], list(STANDARD_LEADS))
        self.assertEqual(raw["details"]["filtering"], "none")
        classifier.validate_output(raw)
        backend.ready = False
        with self.assertRaisesRegex(ValueError, "not ready"):
            backend.classify(record)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.payload = b"small pinned checkpoint fixture"
        self.size = mock.patch.object(classifier, "CHECKPOINT_BYTES", len(self.payload))
        self.digest = mock.patch.object(classifier, "CHECKPOINT_SHA256", hashlib.sha256(self.payload).hexdigest())
        self.size.start()
        self.digest.start()
        self.addCleanup(self.size.stop)
        self.addCleanup(self.digest.stop)
        self.path = Path(self.directory.name) / classifier.CHECKPOINT_NAME

    def response(self, payload=None, header=True):
        result = BytesIO(self.payload if payload is None else payload)
        result.headers = {"Content-Length": str(len(self.payload))} if header else {}
        return result

    def test_download_verifies_before_atomic_install_and_reuses_good_file(self):
        with mock.patch.object(classifier.urllib.request, "urlopen", return_value=self.response()) as network:
            self.assertEqual(classifier.download(self.directory.name), self.path)
            network.assert_called_once()
            self.assertEqual(network.call_args.kwargs["timeout"], 30)
        self.assertEqual(self.path.read_bytes(), self.payload)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [self.path])
        with mock.patch.object(classifier.urllib.request, "urlopen") as network:
            self.assertEqual(classifier.download(self.directory.name), self.path)
            network.assert_not_called()

    def test_existing_bad_file_is_preserved_without_network(self):
        self.path.write_bytes(b"do not replace")
        with mock.patch.object(classifier.urllib.request, "urlopen") as network:
            with self.assertRaisesRegex(ValueError, "preserved"):
                classifier.download(self.directory.name)
            network.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b"do not replace")

    def test_bad_hash_truncation_and_oversized_download_leave_no_file(self):
        for payload in (b"x" * len(self.payload), self.payload[:-1], self.payload + b"extra"):
            with mock.patch.object(classifier.urllib.request, "urlopen", return_value=self.response(payload, header=False)):
                with self.subTest(payload=len(payload)), self.assertRaises(ValueError):
                    classifier.download(self.directory.name)
            self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_wrong_content_length_and_network_failure_do_not_install(self):
        response = self.response()
        response.headers["Content-Length"] = "1"
        with mock.patch.object(classifier.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "length"):
                classifier.download(self.directory.name)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])
        with mock.patch.object(classifier.urllib.request, "urlopen", side_effect=OSError("network failed")):
            with self.assertRaises(OSError):
                classifier.download(self.directory.name)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
