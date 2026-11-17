import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from phase2.scripts.dataset import PHASE2, ROOT, verify_source_snapshot


def snapshot(files):
    hashes = {p: hashlib.sha256(value).hexdigest() for p, value in files.items()}
    return {"commit": "original-commit", "files": hashes,
            "fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


class SnapshotTests(unittest.TestCase):
    def check(self, before, after):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in after.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            saved = snapshot(before)
            original = json.dumps(saved, sort_keys=True)
            current = snapshot(after)
            current["commit"] = "another-code-commit"
            result = verify_source_snapshot(saved, current, root)
            self.assertEqual(json.dumps(saved, sort_keys=True), original)
            return result

    def test_lf_crlf_both_directions_and_json(self):
        lf = {"metadata.csv": b'File,Text\na.wav,"apo, kabar"\n',
              "qc.json": b'{\n  "ok": true\n}\n', "audio.wav": b"RIFF\r\n\x00"}
        crlf = {p: value.replace(b"\n", b"\r\n") if p.endswith((".csv", ".json")) else value
                for p, value in lf.items()}
        for before, after in [(lf, crlf), (crlf, lf)]:
            with self.subTest(before=before):
                result = self.check(before, after)
                self.assertEqual(set(result), {"metadata.csv", "qc.json"})

    def test_changed_transcript_and_whitespace_rejected(self):
        before = {"annotation.csv": b"speaker,text\nS01,apo\n"}
        for changed in [b"speaker,text\nS01,iyo\n", b"speaker,text\nS01,apo \n"]:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "Source content changed"):
                self.check(before, {"annotation.csv": changed})

    def test_audio_bytes_never_normalized(self):
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.check({"audio.wav": b"RIFF\r\n"}, {"audio.wav": b"RIFF\n"})

    def test_inventory_addition_and_removal_rejected(self):
        for after in [{}, {"a.csv": b"x\n", "new.csv": b"x\n"}]:
            with self.subTest(after=after), self.assertRaisesRegex(ValueError, "inventory changed"):
                self.check({"a.csv": b"x\n"}, after)

    def test_corrupt_saved_fingerprint_rejected(self):
        saved = snapshot({"a.csv": b"x\n"})
        saved["files"]["a.csv"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
            verify_source_snapshot(saved, snapshot({"a.csv": b"x\n"}))

    def test_actual_corpus_portable_without_rewriting_frozen_files(self):
        saved = json.loads((PHASE2 / "data/source_snapshot.json").read_text(encoding="utf-8"))
        # Kopi teks saja ke dua simulasi checkout; hash audio tetap persis snapshot.
        for ending in [b"\n", b"\r\n"]:
            with self.subTest(ending=ending), tempfile.TemporaryDirectory() as tmp:
                current = json.loads(json.dumps(saved))
                for name in saved["files"]:
                    if Path(name).suffix.lower() not in {".csv", ".json"}:
                        continue
                    payload = (ROOT / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", ending)
                    dest = Path(tmp) / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(payload)
                    current["files"][name] = hashlib.sha256(payload).hexdigest()
                current["fingerprint"] = hashlib.sha256(json.dumps(current["files"], sort_keys=True).encode()).hexdigest()
                verify_source_snapshot(saved, current, Path(tmp))


if __name__ == "__main__":
    unittest.main()
