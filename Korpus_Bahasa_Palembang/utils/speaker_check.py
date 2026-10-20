"""Cek kesesuaian Speaker ID per segmen dengan speaker embedding (WavLM x-vector).

Referensi suara penutur diambil dari segmen panjang (>= 2 detik) pada file referensi
(misal file perkenalan yang hanya berisi penutur). Setiap segmen pada file target diberi
skor cosine similarity terhadap referensi; skor rendah = kemungkinan suara orang lain.

Pemakaian:
    python utils/speaker_check.py --segments <tsv> --ref <tsv_ref> [--ref ...] --out <csv>
TSV segmen: kolom unit_id, start, end (kolom lain diabaikan); baris '#' diabaikan.
"""
import argparse
import csv
import os
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio"
MODEL = "microsoft/wavlm-base-plus-sv"


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def segments(tsv):
    out = []
    for line in Path(tsv).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        out.append((f[0], float(f[1]), float(f[2]), "\t".join(f[3:])))
    return out


class Embedder:
    def __init__(self):
        self.dev = os.environ.get("DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL)
        self.model = WavLMForXVector.from_pretrained(MODEL).to(self.dev).eval()

    @torch.inference_mode()
    def __call__(self, x):
        inp = self.fe([x], sampling_rate=16000, return_tensors="pt", padding=True).to(self.dev)
        e = self.model(**inp).embeddings[0]
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def stem_of(tsv):
    return Path(tsv).name.split(".")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    emb = Embedder()
    refs = []
    for r in args.ref:
        st = stem_of(r)
        x = read_wav(AUDIO / st[:3] / f"{st}.wav")
        for _, a, b, _ in segments(r):
            if b - a >= 2.0:
                refs.append(emb(x[int(a * 16000): int(b * 16000)]))
    ref = np.mean(refs, 0)
    ref /= np.linalg.norm(ref)
    st = stem_of(args.segments)
    x = read_wav(AUDIO / st[:3] / f"{st}.wav")
    rows = []
    for uid, a, b, txt in segments(args.segments):
        seg = x[int(a * 16000): int(b * 16000)]
        if len(seg) < 8000:  # < 0.5 detik: terlalu pendek untuk embedding andal
            rows.append(dict(unit=uid, start=a, end=b, sim="", text=txt[:80]))
            continue
        rows.append(dict(unit=uid, start=a, end=b, sim=round(float(emb(seg) @ ref), 3), text=txt[:80]))
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["unit", "start", "end", "sim", "text"])
        w.writeheader()
        w.writerows(rows)
    print(f"{st}: {len(refs)} segmen referensi, {len(rows)} segmen dinilai")


if __name__ == "__main__":
    main()
