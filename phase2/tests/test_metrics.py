import unittest

from phase2.src.evaluate import evaluate, score, total


def rows(pairs):
    manifest = [{"sample_id": f"S05_{i:04d}", "speaker_id": "S05", "transcript": ref}
                for i, (ref, _) in enumerate(pairs)]
    predictions = [{"sample_id": m["sample_id"], "reference": m["transcript"], "prediction": hyp}
                   for m, (_, hyp) in zip(manifest, pairs)]
    return predictions, manifest


class MetricTests(unittest.TestCase):
    def test_identity(self):
        s = score("aku pegi ke pasar", "Aku pegi ke pasar.")
        self.assertEqual((s["wer"], s["cer"]), (0, 0))

    def test_substitution(self):
        s = score("aku pegi", "aku pigi")
        self.assertEqual((s["substitutions"], s["wer"], s["cer"]), (1, 0.5, 1 / 8))

    def test_deletion(self):
        s = score("aku pegi ke pasar", "aku pasar")
        self.assertEqual((s["deletions"], s["wer"]), (2, 0.5))

    def test_insertion(self):
        s = score("aku pegi", "aku pegi bae")
        self.assertEqual((s["insertions"], s["wer"]), (1, 0.5))

    def test_empty_prediction(self):
        s = score("aku pegi", "")
        self.assertEqual((s["deletions"], s["wer"], s["cer"]), (2, 1.0, 1.0))

    def test_corpus_level_not_mean(self):
        self.assertAlmostEqual(total([score("a b c d", "a b c d"), score("x", "y")])["wer"], 1 / 5)

    def test_masked_name_excluded_from_primary(self):
        result, _ = evaluate(*rows([("aku pegi", "aku pegi"), ("namo aku [nama]", "namo aku budi")]))
        self.assertEqual((result["primary"]["utterances"], result["primary"]["wer"]), (1, 0))
        self.assertEqual((result["all"]["utterances"], result["all"]["insertions"]), (2, 1))

    def test_mismatch_with_manifest_rejected(self):
        predictions, manifest = rows([("aku", "aku"), ("kito", "kito")])
        with self.assertRaises(ValueError):
            evaluate(predictions[:1], manifest)
        predictions[0]["reference"] = "kami"
        with self.assertRaises(ValueError):
            evaluate(predictions, manifest)


if __name__ == "__main__":
    unittest.main()
