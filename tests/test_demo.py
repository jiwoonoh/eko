import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import demo
from contracts import STANDARD_LEADS

try:
    import numpy as np
    from scipy.io import loadmat
    SCIENCE_AVAILABLE = True
except ImportError:
    SCIENCE_AVAILABLE = False


@unittest.skipUnless(SCIENCE_AVAILABLE, "install the measurement extra")
class DemoTests(unittest.TestCase):
    def fixture(self):
        lines = ["00705_hr 12 500 5000"]
        samples = np.zeros((5000, 12), dtype="<i2")
        for index, (initial, checksum, lead) in enumerate(zip(demo.INITIAL, demo.CHECKSUMS, demo.SOURCE_LEADS)):
            samples[0, index] = initial
            correction = (checksum - initial) % 65536
            samples[-1, index] = correction if correction < 32768 else correction - 65536
            lines.append(f"00705_hr.dat 16 1000.0(0)/mV 16 0 {initial} {checksum} 0 {lead}")
        return ("\n".join(lines) + "\n").encode(), samples.tobytes()

    def test_narrow_converter_preserves_channel_order_and_physical_units(self):
        header, payload = self.fixture()
        result = demo.convert_record(header, payload)
        self.assertEqual(result.shape, (12, 5000))
        np.testing.assert_equal(result[:, 0], np.array(demo.INITIAL) / 1000)
        self.assertEqual(result[0, 1], 0)

    def test_header_format_frequency_and_lead_changes_are_rejected(self):
        header, payload = self.fixture()
        for old, new in [(b"12 500 5000", b"12 250 5000"), (b".dat 16 ", b".dat 24 "),
                         (b"1000.0(0)/mV", b"1000.0(0)/uV"), (b"0 II\n", b"0 I\n")]:
            with self.subTest(old=old):
                with self.assertRaisesRegex(ValueError, "header"):
                    demo.convert_record(header.replace(old, new, 1), payload)

    def test_initial_sample_checksum_and_data_length_are_checked(self):
        header, payload = self.fixture()
        with self.assertRaisesRegex(ValueError, "120000"):
            demo.convert_record(header, payload[:-2])
        initial = np.frombuffer(payload, dtype="<i2").copy()
        initial[0] += 1
        with self.assertRaisesRegex(ValueError, "initial"):
            demo.convert_record(header, initial.tobytes())
        checksum = np.frombuffer(payload, dtype="<i2").copy()
        checksum[12] += 1
        with self.assertRaisesRegex(ValueError, "checksums"):
            demo.convert_record(header, checksum.tobytes())

    def test_source_hash_rejects_changed_public_bytes(self):
        header, payload = self.fixture()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            demo._verify_source(header, payload)

    def test_bundle_metadata_reuse_and_no_overwrite(self):
        header, payload = self.fixture()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        question = "Give the ECG measurements."
        published = {"source_url": demo.PUBLISHED_URL, "source_revision": demo.UPSTREAM_REVISION,
                     "first_line_sha256": demo.PUBLISHED_LINE_SHA256, "ecg_id": "00705", "batch_idx": 283,
                     "question": question, "stored_tool_output": demo.STORED_MEASUREMENTS}
        hashes = {"HEADER_SHA256": hashlib.sha256(header).hexdigest(), "DATA_SHA256": hashlib.sha256(payload).hexdigest(),
                  "PUBLISHED_QUESTION_SHA256": hashlib.sha256(question.encode()).hexdigest()}
        with mock.patch.multiple(demo, **hashes), mock.patch.object(demo, "_published_case", return_value=published), \
             mock.patch.object(demo, "_download", side_effect=[header, payload]) as download:
            path = demo.prepare_demo(directory.name)
            saved = path.read_bytes()
            mat = loadmat(path)
            self.assertEqual(mat["feats"].shape, (12, 5000))
            self.assertEqual(mat["curr_sample_rate"].item(), 500)
            self.assertEqual(len(mat["lead_names"][0]), 12)
            manifest = json.loads((Path(directory.name) / "source.json").read_text())
            self.assertEqual(manifest["lead_names"], list(STANDARD_LEADS))
            self.assertEqual(manifest["waveform_units"], ["mV"] * 12)
            self.assertEqual(manifest["mat_sha256"], hashlib.sha256(saved).hexdigest())
            self.assertEqual(demo.prepare_demo(directory.name), path)
            self.assertEqual(download.call_count, 2)
            self.assertEqual(path.read_bytes(), saved)
            path.write_bytes(saved + b"changed")
            with self.assertRaisesRegex(ValueError, "MAT hash"):
                demo.prepare_demo(directory.name)
            self.assertEqual(path.read_bytes(), saved + b"changed")

    def test_incomplete_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / demo.MAT_NAME
            path.write_bytes(b"existing")
            with mock.patch.object(demo, "_download") as download:
                with self.assertRaises(FileExistsError):
                    demo.prepare_demo(directory)
            download.assert_not_called()
            self.assertEqual(path.read_bytes(), b"existing")


class DemoDownloadTests(unittest.TestCase):
    def published_line(self, ecg_id="00705", question="Give the ECG measurements."):
        example = {"ecg_id": ecg_id, "batch_idx": 283,
                   "dialogue": [{"role": "user", "content": "Classify the ECG."}, {}, {},
                                {"role": "user", "content": question},
                                {"role": "assistant", "action": "call_measurement_tool", "tool_output": demo.STORED_MEASUREMENTS}]}
        return (json.dumps(example) + "\n").encode(), question

    def test_published_first_line_selects_only_measurement_question(self):
        line, question = self.published_line()
        with mock.patch.multiple(demo, PUBLISHED_LINE_SHA256=hashlib.sha256(line).hexdigest(),
                                 PUBLISHED_QUESTION_SHA256=hashlib.sha256(question.encode()).hexdigest()), \
             mock.patch.object(demo.urllib.request, "urlopen", return_value=BytesIO(line)):
            result = demo._published_case()
        self.assertEqual(result["ecg_id"], "00705")
        self.assertEqual(result["question"], question)
        self.assertEqual(result["measurement_user_turn"], 2)
        self.assertNotIn("dialogue", result)
        self.assertNotIn("classification", result)

    def test_published_record_identity_and_first_line_limit_are_checked(self):
        line, _ = self.published_line(ecg_id="00001")
        with mock.patch.object(demo, "PUBLISHED_LINE_SHA256", hashlib.sha256(line).hexdigest()), \
             mock.patch.object(demo.urllib.request, "urlopen", return_value=BytesIO(line)):
            with self.assertRaisesRegex(ValueError, "00705"):
                demo._published_case()
        with mock.patch.object(demo.urllib.request, "urlopen", return_value=BytesIO(b"x" * (demo.PUBLISHED_LINE_LIMIT + 1))):
            with self.assertRaisesRegex(ValueError, "byte limit"):
                demo._published_case()

    def test_download_checks_declared_and_actual_byte_limits(self):
        for data, length in [(b"short", "100"), (b"too-long", None)]:
            response = BytesIO(data)
            response.headers = {} if length is None else {"Content-Length": length}
            with self.subTest(length=length), mock.patch.object(demo.urllib.request, "urlopen", return_value=response) as download:
                with self.assertRaisesRegex(ValueError, "byte limit"):
                    demo._download(demo.HEADER_URL, 4)
                self.assertEqual(download.call_args.kwargs["timeout"], 30)


if __name__ == "__main__":
    unittest.main()
