import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from phase2.scripts.dataset import PHASE2, read_csv
from phase2.src import infer, model


class ModelTests(unittest.TestCase):
    def adapter(self):
        asr = model.WhisperASR.__new__(model.WhisperASR)
        asr.config = model.load_config()
        asr.backend = Mock()
        return asr

    def test_lazy_segments_and_exact_decoding(self):
        asr = self.adapter()
        asr.backend.transcribe.return_value = (iter([
            SimpleNamespace(text=" Dak, apo"), SimpleNamespace(text=" nian!")]), None)
        signal = np.zeros(16000, dtype=np.float32)
        self.assertEqual(asr.transcribe_signal(signal), "Dak, apo nian!")
        asr.backend.transcribe.assert_called_once_with(signal, **asr.config["decoding"])

    def test_invalid_waveforms_rejected(self):
        asr = self.adapter()
        for signal in [np.array([], dtype=np.float32), np.zeros((2, 10), dtype=np.float32),
                       np.array([np.nan], dtype=np.float32), np.array([np.inf], dtype=np.float32),
                       np.array([1.1], dtype=np.float32), np.zeros(10, dtype=np.int16)]:
            with self.subTest(signal=signal), self.assertRaises(ValueError):
                asr.transcribe_signal(signal)
        asr.backend.transcribe.assert_not_called()

    def test_lazy_decode_failure_propagates(self):
        def failing_segments():
            yield SimpleNamespace(text="partial")
            raise RuntimeError("decode failed")
        asr = self.adapter()
        asr.backend.transcribe.return_value = (failing_segments(), None)
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            asr.transcribe_signal(np.zeros(16000, dtype=np.float32))

    def test_config_requires_immutable_revision(self):
        config = model.load_config()
        config["revision"] = "main"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "immutable"):
                model.load_config(path)

    def test_experiments_change_only_target_factor(self):
        base = model.load_config()
        beam = model.load_config(PHASE2 / "configs/experiments/whisper_small_beam5.json")
        beam["decoding"]["beam_size"] = base["decoding"]["beam_size"]
        self.assertEqual(beam, base)
        preprocessing = model.load_config(PHASE2 / "configs/experiments/whisper_small_rms_trim.json")
        preprocessing["preprocess_config"] = base["preprocess_config"]
        self.assertEqual(preprocessing, base)


class PredictionRunTests(unittest.TestCase):
    def test_predictions_preserve_references_and_runs_are_immutable(self):
        fake = Mock()
        fake.preprocess_config = {"_sha256": "test", "version": "preprocess_v1"}
        fake.artifact_hashes = {"model.bin": "test"}
        fake.backend.model.device = "cpu"
        fake.backend.model.compute_type = "int8_float32"
        fake.transcribe_sample.return_value = 'Dak, "apo"!'
        with tempfile.TemporaryDirectory() as tmp, patch.object(infer, "PHASE2", Path(tmp)):
            # Kopi input kecil supaya tes juga jalan di Windows tanpa izin symlink.
            shutil.copytree(PHASE2 / "configs", Path(tmp) / "configs")
            (Path(tmp) / "data/manifests").mkdir(parents=True)
            shutil.copy2(PHASE2 / "data/source_snapshot.json", Path(tmp) / "data/source_snapshot.json")
            shutil.copy2(PHASE2 / "data/manifests/validation.csv", Path(tmp) / "data/manifests/validation.csv")
            with patch.object(infer.model, "WhisperASR", return_value=fake) as loader:
                out = infer.run("smoke", limit=2)
                rows = read_csv(out / "predictions.csv")
                original = read_csv(PHASE2 / "data/manifests/validation.csv")[:2]
                self.assertEqual([r["reference"] for r in rows], [r["transcript"] for r in original])
                self.assertEqual([r["prediction"] for r in rows], ['Dak, "apo"!'] * 2)
                self.assertEqual(json.loads((out / "run.json").read_text())["status"], "complete")
                loader.assert_called_once()
                with self.assertRaises(FileExistsError):
                    infer.run("smoke", limit=2)
                fake.transcribe_sample.side_effect = RuntimeError("backend unavailable")
                with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
                    infer.run("failed", limit=1)
                failure = json.loads((Path(tmp) / "results/baseline/failed/run.json").read_text())
                self.assertEqual(failure["status"], "failed")
                self.assertIn("backend unavailable", failure["error"])

    def test_invalid_run_id_cannot_escape_output_directory(self):
        with self.assertRaises(ValueError):
            infer.run("../outside")

    def test_provenance_rejects_content_changes(self):
        current = infer.source_snapshot()
        current["files"]["Korpus_Bahasa_Palembang/annotation/annotation.csv"] = "0" * 64
        with patch.object(infer, "source_snapshot", return_value=current), patch.object(
                Path, "read_bytes", return_value=b"changed transcript"):
            with self.assertRaisesRegex(ValueError, "Source content changed"):
                infer.provenance()
