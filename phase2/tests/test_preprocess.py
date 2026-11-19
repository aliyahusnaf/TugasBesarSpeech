import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
import wave

import numpy as np

from phase2.scripts.dataset import PHASE2, ROOT, read_csv
from phase2.src import audio
from phase2.src.preprocess import (MANIFEST_FIELDS, SPLITS, build, load_config, load_sample,
                                   output_dir, prepare_input, prepare_text)
from phase2.src.text_normalization import normalize_text, text_flags

QC = load_config()["qc"]


def tone(seconds, amplitude=0.5, freq=440.0):
    t = np.arange(round(seconds * audio.SAMPLE_RATE)) / audio.SAMPLE_RATE
    return audio.to_pcm(amplitude * np.sin(2 * np.pi * freq * t))


class TextNormalizationTests(unittest.TestCase):
    def test_tags_removed_words_kept(self):
        self.assertEqual(normalize_text("namo aku [nama] [cough] dak tau"), "namo aku dak tau")

    def test_dialect_forms_not_replaced(self):
        text = "apo dak nian samo galo-galo kenalke pegi bae"
        self.assertEqual(normalize_text(text), text)

    def test_prediction_formatting(self):
        self.assertEqual(normalize_text("  Dak, APO?!  Kawan–kawan “iyo”. "), "dak apo kawan-kawan iyo")
        self.assertEqual(normalize_text("jum'at 2 kali"), "jumat 2 kali")

    def test_hyphen_only_inside_words(self):
        self.assertEqual(normalize_text("kawan - kawan -ee- galo-galo"), "kawan kawan ee galo-galo")
        self.assertEqual(normalize_text("galo-galo", keep_hyphen=False), "galo galo")

    def test_empty_and_tag_only(self):
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text("[noise]"), "")

    def test_flags(self):
        self.assertEqual(text_flags("namo [nama] [Cough]", "cough;code-switch"),
                         ["masked_name", "event_cough", "code_switch"])
        self.assertEqual(text_flags("dak [unclear]"), ["unclear"])

    def test_corpus_only_loses_tags(self):
        rows = [r for s in SPLITS for r in read_csv(PHASE2 / f"data/manifests/{s}.csv")]
        for row in rows:
            text = normalize_text(row["transcript"])
            self.assertEqual(normalize_text(text), text)
            if "[" not in row["transcript"]:
                self.assertEqual(text, row["transcript"], row["sample_id"])
            else:
                self.assertEqual(text.split(), [w for w in row["transcript"].split() if not w.startswith("[")])


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write_raw(self, path, signal, rate, channels):
        data = np.repeat(audio.to_pcm(signal)[:, None], channels, axis=1)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(data.astype("<i2").tobytes())

    def test_wav_roundtrip_and_segment_bounds(self):
        pcm = tone(1.0)
        audio.write_wav(self.tmp / "a.wav", pcm)
        np.testing.assert_array_equal(audio.read_wav(self.tmp / "a.wav"), pcm)
        np.testing.assert_array_equal(audio.read_segment(self.tmp / "a.wav", 0.25, 0.5), pcm[4000:8000])
        with self.assertRaises(ValueError):
            audio.read_segment(self.tmp / "a.wav", 0.5, 2.0)

    def test_nonstandard_wav_rejected_then_converted(self):
        t = np.arange(44100) / 44100
        self.write_raw(self.tmp / "src.wav", 0.3 * np.sin(2 * np.pi * 300 * t), 44100, 2)
        with self.assertRaises(ValueError):
            audio.read_wav(self.tmp / "src.wav")
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg tidak tersedia")
        before = (self.tmp / "src.wav").read_bytes()
        info = audio.convert_to_standard(self.tmp / "src.wav", self.tmp / "out/dst.wav")
        self.assertTrue(audio.is_standard_format(info))
        self.assertAlmostEqual(info["frames"], 16000, delta=16)
        self.assertEqual((self.tmp / "src.wav").read_bytes(), before)
        self.assertAlmostEqual(len(audio.load_audio_file(self.tmp / "src.wav")), 16000, delta=16)
        with self.assertRaises(ValueError):
            audio.convert_to_standard(self.tmp / "src.wav", self.tmp / "src.wav")

    def test_to_pcm_clips(self):
        np.testing.assert_array_equal(audio.to_pcm(np.array([2.0, -2.0, 0.0])), [32767, -32768, 0])

    def test_metrics_detect_silence_and_clipping(self):
        pcm = np.concatenate([np.zeros(8000, "<i2"), tone(1.0), np.zeros(16000, "<i2")])
        m = audio.audio_metrics(pcm, QC)
        self.assertAlmostEqual(m["leading_silence_sec"], 0.5, delta=QC["frame_sec"])
        self.assertAlmostEqual(m["trailing_silence_sec"], 1.0, delta=QC["frame_sec"])
        self.assertAlmostEqual(m["active_ratio"], 0.4, delta=0.02)
        self.assertEqual(m["longest_clip_run"], 0)
        clipped = tone(0.1, amplitude=3.0)
        self.assertGreaterEqual(audio.audio_metrics(clipped, QC)["longest_clip_run"], QC["min_clip_run"])
        self.assertEqual(audio.audio_metrics(np.zeros(1600, "<i2"), QC)["active_ratio"], 0)

    def test_default_variant_is_identity(self):
        pcm = tone(0.5, 0.1)
        result = audio.apply_variant(pcm, load_config()["audio"], QC)
        np.testing.assert_array_equal(result, pcm)

    def test_rms_and_trim_variant(self):
        cfg = load_config(PHASE2 / "configs/preprocess_rms_trim_v1.json")["audio"]
        pcm = np.concatenate([np.zeros(16000, "<i2"), tone(1.0, 0.01), np.zeros(16000, "<i2")])
        result = audio.apply_variant(pcm, cfg, QC)
        self.assertAlmostEqual(len(result) / audio.SAMPLE_RATE, 1.2, delta=0.03)
        rms = audio.db(np.sqrt(np.mean(np.square(audio.to_float(result), dtype=np.float64))))
        peak = audio.db(np.max(np.abs(audio.to_float(result))))
        self.assertLessEqual(peak, cfg["peak_limit_dbfs"] + 0.01)
        self.assertAlmostEqual(rms, cfg["target_rms_dbfs"], delta=1.0)
        loud = audio.apply_variant(tone(1.0, 0.9), cfg, QC)
        self.assertLessEqual(audio.db(np.max(np.abs(audio.to_float(loud)))), cfg["peak_limit_dbfs"] + 0.01)
        silent = np.zeros(800, "<i2")
        np.testing.assert_array_equal(audio.apply_variant(silent, cfg, QC), silent)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.manifests, cls.qc_rows, cls.sources, cls.changes = build(cls.config)
        cls.out = output_dir(cls.config)

    def test_committed_outputs_reproducible(self):
        for split in SPLITS:
            saved = read_csv(self.out / f"{split}.csv")
            fresh = [{k: str(r[k]) for k in MANIFEST_FIELDS} for r in self.manifests[split]]
            self.assertEqual(saved, fresh, split)

    def test_sources_standard_and_unchanged(self):
        self.assertTrue(all(r["standard_format"] and r["action"] == "none" for r in self.sources))
        for split in SPLITS:
            source = read_csv(PHASE2 / f"data/manifests/{split}.csv")
            processed = self.manifests[split]
            self.assertEqual([r["sample_id"] for r in source], [r["sample_id"] for r in processed])
            for a, b in zip(source, processed):
                self.assertEqual(a["transcript"], b["transcript"])
                self.assertEqual(a["pcm_sha256"], b["pcm_sha256"])
                self.assertEqual(a["speaker_id"], b["speaker_id"])
                self.assertEqual(b["text"], prepare_text(a["transcript"], self.config))

    def test_training_policy(self):
        rows = [r for s in SPLITS for r in self.manifests[s]]
        for row in rows:
            masked = "masked_name" in row["text_flags"].split(";")
            if masked or not row["text"]:
                self.assertFalse(row["train_eligible"], row["sample_id"])
            if row["train_eligible"]:
                self.assertTrue(row["text"])
                self.assertNotIn("[", row["text"])

    def test_load_sample_matches_manifest_hash(self):
        for split in SPLITS:
            row = self.manifests[split][0]
            signal = load_sample(row, self.config)
            self.assertEqual(hashlib.sha256(audio.to_pcm(signal).tobytes()).hexdigest(), row["pcm_sha256"])
        with self.assertRaises(ValueError):
            load_sample({**self.manifests["train"][0], "preprocess_version": "other"}, self.config)

    def test_prepare_input_matches_file(self):
        row = self.manifests["test"][0]
        path = ROOT / row["source_audio_path"]
        np.testing.assert_array_equal(prepare_input(path, self.config), audio.to_float(audio.read_wav(path)))


if __name__ == "__main__":
    unittest.main()
