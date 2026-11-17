from copy import deepcopy
import unittest
from unittest.mock import patch

from phase2.scripts.check_model_input import check
from phase2.scripts.dataset import PHASE2, read_csv


class ModelInputTests(unittest.TestCase):
    def test_real_validation_handoff(self):
        result = check()
        self.assertEqual(result["status"], "input_contract_passed")
        self.assertEqual(result["utterances"], 235)
        self.assertEqual(result["sample_rate"], 16000)

    def test_mismatched_preprocessing_is_rejected(self):
        rows = read_csv(PHASE2 / "data/manifests/validation.csv")[:1]
        prepared = read_csv(PHASE2 / "data/processed/preprocess_v1/validation.csv")[:1]
        for field, value in [("split", "test"), ("sample_id", "other"),
                             ("transcript", "changed"), ("text", "changed"),
                             ("pcm_sha256", "0" * 64), ("duration_sec", "999")]:
            changed = deepcopy(prepared)
            changed[0][field] = value
            with self.subTest(field=field), patch(
                    "phase2.scripts.check_model_input.read_csv", side_effect=[rows, changed]):
                with self.assertRaises(ValueError):
                    check()

    def test_empty_manifest_is_rejected(self):
        with patch("phase2.scripts.check_model_input.read_csv", side_effect=[[], []]):
            with self.assertRaises(ValueError):
                check()
