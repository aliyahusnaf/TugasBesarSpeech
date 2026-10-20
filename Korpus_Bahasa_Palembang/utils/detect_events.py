"""Deteksi kejadian khusus (tawa, batuk, noise) untuk label Events anotasi.

Model: MIT/ast-finetuned-audioset-10-10-0.4593 (Audio Spectrogram Transformer, AudioSet).
Audio dipindai dengan jendela 1 detik (hop 0.5 detik). Sebuah jendela diberi label
jika skor kelas terkait melewati ambang. Jendela berurutan dengan label sama digabung.
Hasil adalah KANDIDAT event; tetap diperiksa saat review.

Pemakaian:
    python utils/detect_events.py --out events.json S01 S02 ...
"""
import argparse
import json
import os
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio"
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

GROUPS = {
    "laugh": (["Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Baby laughter"], 0.30),
    "cough": (["Cough", "Throat clearing", "Sneeze"], 0.30),
    "noise": (["Vehicle horn, car horn, honking", "Music", "Dog", "Bark", "Door", "Slam", "Knock",
               "Bang", "Clapping", "Motorcycle", "Siren", "Telephone bell ringing", "Ringtone",
               "Dishes, pots, and pans", "Cutlery, silverware", "Crumpling, crinkling", "Bird",
               "Rooster", "Chicken, rooster", "Cat", "Baby cry, infant cry", "Television", "Beep, bleep",
               "Alarm", "Hammer", "Wind noise (microphone)", "Microphone"], 0.25),
}
# noise latar kontinu (dilaporkan per file untuk QC, bukan tag per kata)
BACKGROUND = ["Vehicle", "Car", "Motor vehicle (road)", "Traffic noise, roadway noise", "Engine",
              "Mechanical fan", "Air conditioning", "Wind", "Rain", "Hubbub, speech noise, speech babble",
              "Music"]
WIN, HOP = 2.0, 1.0


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


@torch.inference_mode()
def scan(x, fe, model, dev):
    sr = 16000
    starts = np.arange(0, max(len(x) / sr - WIN, 0) + 1e-9, HOP)
    probs = []
    for i in range(0, len(starts), 32):
        chunk = [x[int(s * sr): int((s + WIN) * sr)] for s in starts[i:i + 32]]
        feats = fe(chunk, sampling_rate=sr, return_tensors="pt").input_values.to(dev).to(next(model.parameters()).dtype)
        probs.append(torch.sigmoid(model(feats).logits).float().cpu().numpy())
    return starts, np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speakers", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = os.environ.get("DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
    fe = ASTFeatureExtractor.from_pretrained(MODEL)
    model = ASTForAudioClassification.from_pretrained(MODEL).to(dev).eval()
    if dev == "cuda":
        model = model.half()
    lab2id = model.config.label2id
    res, bg = {}, {}
    for spk in args.speakers:
        for wav in sorted((AUDIO / spk).glob(f"{spk}_T*.wav")):
            x = read_wav(wav)
            if len(x) < 16000 * WIN:
                x = np.pad(x, (0, int(16000 * WIN) - len(x)))
            starts, P = scan(x, fe, model, dev)
            evs = []
            for name, (classes, thr) in GROUPS.items():
                ids = [lab2id[c] for c in classes if c in lab2id]
                score = P[:, ids].max(1)
                top = [classes[j] for j in P[:, ids].argmax(1)]
                cur = None
                for s, sc, tp in zip(starts, score, top):
                    if sc >= thr:
                        if cur and s <= cur["end"]:
                            cur["end"] = round(s + WIN, 2)
                            cur["score"] = max(cur["score"], round(float(sc), 3))
                        else:
                            cur = dict(start=round(float(s), 2), end=round(float(s + WIN), 2), label=name,
                                       cls=tp, score=round(float(sc), 3))
                            evs.append(cur)
            res[wav.stem] = sorted(evs, key=lambda e: e["start"])
            bids = [lab2id[c] for c in BACKGROUND if c in lab2id]
            mean = P[:, bids].mean(0)
            bg[wav.stem] = {BACKGROUND[i]: round(float(m), 3) for i, m in enumerate(mean) if m >= 0.05}
            print(wav.stem, len(evs), "events", bg[wav.stem], flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    Path(args.out).with_name("background_noise.json").write_text(json.dumps(bg, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
