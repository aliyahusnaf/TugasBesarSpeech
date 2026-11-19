import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from phase2.scripts.dataset import PHASE2, ROOT
from phase2.src import preprocess, selection_metrics, train

TRAIN_CONFIGS = PHASE2 / "configs/train"
REFERENCE_CONFIG = TRAIN_CONFIGS / "whisper_small_ft_lr1e-5.json"


class SelectionMetricTests(unittest.TestCase):
    def test_identity_after_normalization(self):
        s = selection_metrics.utterance_scores("Halo, kenalin namo aku [nama]!", "halo kenalin namo aku")
        self.assertEqual((s["wer"], s["cer"]), (0.0, 0.0))

    def test_substitution_deletion_insertion(self):
        s = selection_metrics.utterance_scores("dak apo apo", "dak apa")
        self.assertEqual((s["substitutions"], s["deletions"], s["insertions"]), (1, 1, 0))
        self.assertAlmostEqual(s["wer"], 2 / 3)
        s = selection_metrics.utterance_scores("dak", "dak apo")
        self.assertEqual((s["insertions"], s["wer"]), (1, 1.0))

    def test_empty_prediction_is_all_deletions(self):
        s = selection_metrics.utterance_scores("wong kito galo", "")
        self.assertEqual((s["deletions"], s["wer"], s["cer"]), (3, 1.0, 1.0))

    def test_corpus_level_is_weighted_by_reference_length(self):
        scores = selection_metrics.corpus_scores(["a b c d", "e"], ["a b c d", "x"])
        self.assertAlmostEqual(scores["wer"], 1 / 5)  # bukan rata-rata (0 + 1) / 2

    def test_empty_reference_rejected(self):
        with self.assertRaises(ValueError):
            selection_metrics.utterance_scores("[nama]", "budi")


class TrainingConfigTests(unittest.TestCase):
    def test_experiments_change_only_one_factor(self):
        reference = train.load_train_config(REFERENCE_CONFIG)
        expected = {"whisper_small_ft_lr5e-6.json": {"learning_rate"},
                    "whisper_small_ft_lr3e-5.json": {"learning_rate"},
                    "whisper_small_ft_lr1e-5_freeze_encoder.json": {"freeze_encoder"}}
        for name, factors in expected.items():
            config = train.load_train_config(TRAIN_CONFIGS / name)
            changed = {k for k in reference if reference[k] != config[k]} | (set(config) - set(reference))
            self.assertEqual(changed, factors, name)

    def test_kaggle_job_points_to_existing_unique_runs(self):
        job = json.loads((TRAIN_CONFIGS / "kaggle_job_v1.json").read_text(encoding="utf-8"))
        ids = [r["run_id"] for r in job["runs"]]
        self.assertEqual(len(ids), len(set(ids)))
        for item in job["runs"]:
            train.load_train_config(ROOT / item["config"])

    def test_mutable_revision_rejected(self):
        config = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
        config["base_revision"] = "main"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "commit SHA"):
                train.load_train_config(path)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.pre = preprocess.load_config()

    def test_test_split_is_never_loaded(self):
        with self.assertRaisesRegex(ValueError, "test set"):
            train.load_rows("test", self.pre)

    def test_train_rows_are_eligible_and_speaker_disjoint(self):
        rows, _ = train.load_rows("train", self.pre)
        val, _ = train.load_rows("validation", self.pre)
        self.assertTrue(rows and all(r["train_eligible"] == "True" for r in rows))
        self.assertFalse({r["speaker_id"] for r in rows} & {r["speaker_id"] for r in val})
        self.assertFalse(any("[nama]" in r["transcript"] for r in rows))

    def test_signal_matches_preprocess_loader(self):
        row = train.load_rows("validation", self.pre, limit=1)[0][0]
        np.testing.assert_array_equal(train.load_signal(row, self.pre), preprocess.load_sample(row, self.pre))

    def test_collate_pads_labels_and_drops_start_token(self):
        start = 50258
        batch = [{"features": np.zeros((80, 3000), np.float32), "labels": [start, 1, 2, 3]},
                 {"features": np.zeros((80, 3000), np.float32), "labels": [start, 4]}]
        features, labels = train.collate(batch, start)
        self.assertEqual(tuple(features.shape), (2, 80, 3000))
        self.assertEqual(labels.tolist(), [[1, 2, 3], [4, -100, -100]])


if __name__ == "__main__":
    unittest.main()
