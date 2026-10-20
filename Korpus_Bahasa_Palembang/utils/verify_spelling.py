"""Verifikasi akustik ejaan pada transkrip hasil review.

Whisper cenderung "meng-Indonesia-kan" kata Palembang (apo -> apa, dak -> gak, ...).
Untuk setiap kata yang punya varian (a<->o di akhir kata, -nya<->-nyo, gak<->dak, dst.)
dihitung CTC log-likelihood kalimat dengan kata asli vs dengan varian, memakai model
akustik MMS (tanpa language model). Varian yang skornya jauh lebih tinggi dilaporkan
(atau langsung diterapkan dengan --apply).

Pemakaian:
    python utils/verify_spelling.py --reviewed <dir_tsv> [--apply] [--margin 2.0] S01_T02_001 ...
"""
import argparse
import json
import os
import re
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio"
M = "MahmoudAshraf/mms-300m-1130-forced-aligner"

# Catatan: pasangan yang hanya beda huruf h di akhir (sih/si, tuh/tu, nih/ni) TIDAK diuji,
# karena h akhir hampir tak terdengar sehingga skor akustik selalu memihak tanpa-h.
PAIRS = [("gak", "dak"), ("nggak", "dak"), ("enggak", "idak"), ("tidak", "idak"), ("aja", "bae"),
         ("be", "bae"), ("kenapa", "ngapo"), ("kalau", "kalu"),
         ("kalo", "kalu"), ("karna", "karno"), ("karena", "kareno"), ("pergi", "pegi"),
         ("emang", "memang"), ("ama", "samo")]


def variants(w):
    out = set()
    if w.startswith("["):
        return out
    for a, b in PAIRS:
        if w == a:
            out.add(b)
        if w == b:
            out.add(a)
    if re.search(r"[bcdfghjklmnpqrstvwxyz]a$", w) and len(w) >= 2:
        out.add(w[:-1] + "o")
    if re.search(r"[bcdfghjklmnpqrstvwxyz]o$", w) and len(w) >= 2:
        out.add(w[:-1] + "a")
    if w.endswith("nya"):
        out.add(w[:-3] + "nyo")
    if w.endswith("nyo"):
        out.add(w[:-3] + "nya")
    # reduplikasi: kawan-kawan dsb tidak diubah sebagian
    out.discard(w)
    return {v for v in out if v}


class Scorer:
    def __init__(self):
        self.dev = os.environ.get("DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Wav2Vec2ForCTC.from_pretrained(M).to(self.dev).eval()
        vocab = json.loads(Path(hf_hub_download(M, "vocab.json")).read_text())
        self.vocab = {k: v for k, v in vocab.items() if len(k) == 1}

    @torch.inference_mode()
    def logprobs(self, x):
        x = (x - x.mean()) / (x.std() + 1e-7)
        lg = self.model(torch.from_numpy(x).to(self.dev)[None]).logits[0].float()
        return torch.log_softmax(lg, -1)

    def ids(self, words):
        return [self.vocab[c] for w in words if not w.startswith("[") for c in w.replace("-", "") if c in self.vocab]

    def score(self, lp, sentences):
        tg = [torch.tensor(self.ids(s), dtype=torch.long) for s in sentences]
        T = lp.shape[0]
        lps = lp[:, None, :].expand(T, len(tg), lp.shape[1])
        loss = F.ctc_loss(lps, torch.cat(tg).to(self.dev), torch.full((len(tg),), T, dtype=torch.long),
                          torch.tensor([len(t) for t in tg]), blank=0, reduction="none", zero_infinity=True)
        return (-loss).cpu().numpy()


BRACE = re.compile(r"\{([^{}]+)\}")


def resolve_braces(sc, lp, text):
    """Teks review boleh berisi alternatif {a b|c|d}. Pilih kombinasi dengan skor CTC tertinggi."""
    import itertools
    opts = [m.split("|") for m in BRACE.findall(text)]
    if not opts:
        return text, None
    parts = BRACE.split(text)  # teks, opsi, teks, opsi, ...
    combos = list(itertools.product(*opts))[:256]
    sents = []
    for c in combos:
        s = []
        for k, p in enumerate(parts):
            s.append(c[k // 2] if k % 2 else p)
        sents.append(" ".join("".join(s).split()))
    scores = sc.score(lp, [s.split() for s in sents])
    order = np.argsort(-scores)
    best = sents[order[0]]
    gap = scores[order[0]] - scores[order[1]] if len(order) > 1 else 99
    return best, (combos[order[0]], round(float(gap), 1))


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--reviewed", required=True)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--exclude", default="", help="kata (dipisah koma) yang tidak diuji")
    args = ap.parse_args()
    excl = set(w for w in args.exclude.split(",") if w)
    sc = Scorer()
    for stem in args.stems:
        path = Path(args.reviewed) / f"{stem}.tsv"
        x = read_wav(AUDIO / stem[:3] / f"{stem}.wav")
        lines = path.read_text(encoding="utf-8").splitlines()
        new, braces = [], False
        for line in lines:
            if not line.strip() or line.startswith("#"):
                new.append(line)
                continue
            uid, st, en, spk, text, *rest = line.split("\t")
            a, b = max(0, float(st) - 0.3), float(en) + 0.3
            if "{" in text:
                lp = sc.logprobs(x[int(a * 16000): int(b * 16000)])
                text, info = resolve_braces(sc, lp, text)
                braces = True
                print(f"{stem} {uid}: pilih {info[0]} (selisih {info[1]})")
            words = text.split()
            cands = [(i, v) for i, w in enumerate(words) if w not in excl for v in sorted(variants(w))]
            if not cands or not sc.ids(words):
                new.append("\t".join([uid, st, en, spk, " ".join(words)] + rest))
                continue
            lp = sc.logprobs(x[int(a * 16000): int(b * 16000)])
            sents = [words] + [words[:i] + [v] + words[i + 1:] for i, v in cands]
            s = sc.score(lp, sents)
            base = s[0]
            best = {}
            for (i, v), sv in zip(cands, s[1:]):
                if sv - base > args.margin and sv > best.get(i, (None, -1e9))[1]:
                    best[i] = (v, sv)
            for i, (v, sv) in sorted(best.items()):
                print(f"{stem} {uid}: {words[i]} -> {v}  (+{sv - base:.1f})  | {text}")
                if args.apply:
                    words[i] = v
            new.append("\t".join([uid, st, en, spk, " ".join(words)] + rest))
        if args.apply or braces:
            # pilihan alternatif {..|..} selalu disimpan; varian ejaan hanya dengan --apply
            path.write_text("\n".join(new) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
