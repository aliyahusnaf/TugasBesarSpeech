import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from phase2.scripts.dataset import PHASE2, read_csv
from phase2.src import infer, model


def write_config(tmp, config):
    path = Path(tmp) / "model.json"
    path.write_text(json.dumps(config))
    return path


class LocalCheckpointConfigTests(unittest.TestCase):
    def local_config(self):
        config = model.load_config()
        del config["revision"]
        config.update(model_id="local-finetuned", model_path="phase2/models/example/ct2",
                      base_model_id="openai/whisper-small", base_revision="a" * 40,
                      model_sha256="b" * 64)
        return config

    def test_local_checkpoint_config_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = model.load_config(write_config(tmp, self.local_config()))
        self.assertEqual(loaded["model_path"], "phase2/models/example/ct2")

    def test_local_checkpoint_requires_base_revision_and_hash(self):
        for key in ("base_revision", "model_sha256"):
            config = self.local_config()
            config[key] = "main"
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "immutable"):
                    model.load_config(write_config(tmp, config))

    def test_missing_local_checkpoint_explains_download(self):
        config = self.local_config()
        config["model_path"] = "phase2/models/does_not_exist/ct2"
        with patch.dict("sys.modules", {"faster_whisper": Mock(), "faster_whisper.utils": Mock()}):
            with self.assertRaisesRegex(FileNotFoundError, "unduh checkpoint"):
                model.WhisperASR(config)


class OutputRootTests(unittest.TestCase):
    def test_runs_can_be_written_outside_baseline_directory(self):
        fake = Mock()
        fake.preprocess_config = {"_sha256": "test", "version": "preprocess_v1"}
        fake.artifact_hashes = {"model.bin": "test"}
        fake.backend.model.device = "cpu"
        fake.backend.model.compute_type = "int8_float32"
        fake.transcribe_sample.return_value = "apo"
        with tempfile.TemporaryDirectory() as tmp, patch.object(infer.model, "WhisperASR", return_value=fake):
            out = infer.run("exp_smoke", limit=1, output_root=Path(tmp) / "experiments")
            self.assertEqual(out, Path(tmp) / "experiments" / "exp_smoke")
            rows = read_csv(out / "predictions.csv")
            self.assertEqual(rows[0]["reference"], read_csv(PHASE2 / "data/manifests/validation.csv")[0]["transcript"])
            self.assertFalse((PHASE2 / "results/baseline/exp_smoke").exists())


if __name__ == "__main__":
    unittest.main()
