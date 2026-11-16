from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from phase2.scripts.dataset import PHASE2, ROOT, read_csv, segment_pcm, write_csv
from phase2.scripts.prepare_manifest import choose_split, speaker_summary, validate_rows
from phase2.scripts.validate_manifest import validate


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((PHASE2 / "configs/split_v1.json").read_text(encoding="utf-8"))
        cls.rows = [r for s in ["train", "validation", "test"]
                    for r in read_csv(PHASE2 / f"data/manifests/{s}.csv")]

    def test_actual_manifest_matches_sources(self):
        result = validate()
        self.assertTrue(result["valid"])
        self.assertEqual(result["accepted"] + result["excluded"], 1313)

    def test_speaker_balance_and_selection_reproducible(self):
        speakers, _ = choose_split(speaker_summary(self.rows))
        self.assertEqual(speakers, self.config["speakers"])
        self.assertEqual([len(speakers[s]) for s in ["train", "validation", "test"]], [7, 2, 2])
        stats = speaker_summary(self.rows)
        for group in speakers.values():
            self.assertEqual({stats[s]["native"] for s in group}, {"Ya", "Tidak"})
            self.assertEqual({stats[s]["gender"] for s in group}, {"L", "P"})

    def test_segment_handoff_reads_only_utterance(self):
        for split in ["train", "validation", "test"]:
            row = next(r for r in self.rows if r["split"] == split)
            payload = segment_pcm(ROOT / row["audio_path"], float(row["start_sec"]), float(row["end_sec"]))
            self.assertEqual(len(payload), round(float(row["duration_sec"]) * 16000) * 2)

    def test_speaker_leak_is_rejected(self):
        config = deepcopy(self.config)
        config["speakers"]["test"].append(config["speakers"]["train"][0])
        self.assertTrue(any("lintas split" in e for e in validate_rows(self.rows, config, False)))

    def test_duplicate_id_and_audio_rejected(self):
        rows = self.rows + [dict(self.rows[0])]
        errors = validate_rows(rows, self.config, False)
        self.assertTrue(any("ID dobel" in e for e in errors))
        self.assertTrue(any("PCM dobel" in e for e in errors))

    def test_invalid_bounds_and_paths_rejected(self):
        for changes in [{"end_sec": "999999"}, {"start_sec": "nan"},
                        {"start_sec": "oops"}, {"duration_sec": "0"},
                        {"audio_path": "../outside.wav"}]:
            rows = deepcopy(self.rows)
            rows[0].update(changes)
            with self.subTest(changes=changes):
                self.assertTrue(validate_rows(rows, self.config, False))

    def test_modified_pcm_rejected(self):
        rows = deepcopy(self.rows)
        rows[0]["pcm_sha256"] = "0" * 64
        self.assertTrue(any("Hash segmen" in e for e in validate_rows(rows, self.config)))

    def test_transcript_csv_roundtrip(self):
        row = {"transcript": 'apo, "iyo"\n[nama]'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            write_csv(path, [row], ["transcript"])
            self.assertEqual(read_csv(path), [row])

    def test_exclusions_do_not_enter_manifests(self):
        excluded = read_csv(PHASE2 / "data/manifests/excluded.csv")
        self.assertFalse({r["sample_id"] for r in self.rows} & {r["sample_id"] for r in excluded})
        self.assertFalse(any("S04_T04_001.wav" in r["audio_path"] for r in self.rows))
        self.assertFalse(any(r["speaker_id"] == "OTHER" or "[unclear]" in r["transcript"] for r in self.rows))


if __name__ == "__main__":
    unittest.main()
