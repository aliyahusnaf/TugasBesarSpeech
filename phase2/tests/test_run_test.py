import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from phase2.scripts import run_test
from phase2.scripts.dataset import digest, write_csv
from phase2.src.model import DEFAULT_CONFIG


class TestRunMetadata(unittest.TestCase):
    def test_run_logs_exact_inputs_outputs_and_runtime(self):
        asr = Mock()
        asr.artifact_hashes = {"model.bin": "weight-hash"}
        asr.preprocess_config = {"_sha256": "preprocessing-hash"}
        asr.backend.model.device = "cpu"
        asr.backend.model.compute_type = "int8_float32"
        asr.transcribe_sample.return_value = "apo kabar"
        row = {"sample_id": "test_001", "transcript": "apo kabar"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "test.csv"
            write_csv(manifest, [row], list(row))
            with (patch.object(run_test, "PHASE2", root), patch.object(run_test, "MANIFESTS", root),
                  patch.object(run_test, "validate") as validate,
                  patch.object(run_test, "check_no_leakage"),
                  patch.object(run_test.model, "WhisperASR", return_value=asr),
                  patch.object(run_test.evaluate, "run", return_value={"primary": {"wer": 0}}),
                  patch.object(run_test, "version", return_value="installed")):
                self.assertEqual(run_test.run(DEFAULT_CONFIG, "matched"), {"primary": {"wer": 0}})

            validate.assert_called_once_with()
            out = root / "results/test/matched"
            metadata = json.loads((out / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["manifest_sha256"], digest(manifest))
            self.assertEqual(metadata["predictions_sha256"], digest(out / "predictions.csv"))
            self.assertEqual(metadata["preprocess_config_sha256"], "preprocessing-hash")
            self.assertEqual(metadata["actual_compute_type"], "int8_float32")
            self.assertEqual(metadata["packages"]["jiwer"], "installed")
