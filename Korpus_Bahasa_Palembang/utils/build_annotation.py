"""Tahap akhir anotasi: forced alignment + ekspor annotation.csv, word-level CSV, TextGrid.

Input  : transkrip hasil review per file topik (TSV: unit_id, start, end, speaker, text),
         dengan start/end = jendela kasar dari draf ASR. Teks sudah mengikuti panduan
         transkripsi (huruf kecil, tanpa tanda baca, angka ditulis sebagai kata, tag [..]).
Proses : 1) setiap unit di-align ulang terhadap audio dengan model CTC MMS
            (MahmoudAshraf/mms-300m-1130-forced-aligner) memakai Viterbi CTC
            -> batas waktu per kata.
         2) kata disusun ulang menjadi utterance: pindah utterance jika speaker
            berganti atau jeda antar kata >= 0.5 detik; utterance > 15 detik dipecah
            pada jeda terpanjang.
         3) event (tawa, batuk, noise) dari utils/detect_events.py disisipkan.
Output : annotation/annotation.csv          (utterance level)
         annotation/annotation_words.csv    (word level)
         annotation/textgrid/<file>.TextGrid (Praat; tier utterance, speaker, word, event)

Tag yang di-align sebagai kata (wildcard): [nama] [unclear]
Tag non-kata (tidak punya timestamp kata): [noise] [laugh] [cough] [overlap]

Pemakaian:
    python utils/build_annotation.py --reviewed <dir_tsv> [--events events.json] S01 S02 ...
"""
import argparse
import csv
import json
import os
import re
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio"
ANN = ROOT / "annotation"
ALIGN_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
WILDCARD_TAGS = {"[nama]", "[unclear]"}
EVENT_TAGS = {"[noise]", "[laugh]", "[cough]", "[overlap]"}
FRAME_SEC = 0.02
MAX_GAP = 0.5
MAX_UTT = 15.0
BLOCK_GAP = 1.5     # detik; unit dengan jeda lebih kecil di-align bersama
MAX_BLOCK = 90.0    # detik; panjang maksimum satu blok alignment
CHUNK = 20.0        # detik; audio panjang diproses model per potongan
CTX = 2.0           # detik konteks kiri/kanan tiap potongan
WILD_PEN = 0.3
EXTEND_MAX = 0.25   # detik; perluasan batas kata maksimum berdasarkan energi
# Pelafalan akronim untuk alignment saja (transkripsi tetap memakai akronim)
ALIGN_PRON = {"itb": "itebe", "sma": "esema", "smp": "esempe", "sd": "esde", "sbm": "esbeem",
              "obc": "obese", "fsrd": "efesere", "stei": "stei", "ui": "ui", "ugm": "ugeem",
              "if": "if", "hp": "hape", "ipk": "ipeka", "sti": "esteii", "kfc": "kaefce",
              "mrt": "emerte", "pc": "pece", "ac": "ase", "dm": "deem", "ktm": "kateem",
              "utbk": "uteabeka", "snbt": "esenbete", "snbp": "esenbepe", "ppdb": "pepedebe",
              "tpb": "tepebe", "unpad": "unpad", "its": "iteeres", "bca": "beceaa"}
LANG = "Palembang"
SPK_OVERRIDE = {}


# ---------------------------------------------------------------- audio / model
def read_wav(path):
    with wave.open(str(path), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return x


class Aligner:
    def __init__(self):
        self.dev = os.environ.get("DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Wav2Vec2ForCTC.from_pretrained(ALIGN_MODEL).to(self.dev).eval()
        from huggingface_hub import hf_hub_download
        vocab = json.loads(Path(hf_hub_download(ALIGN_MODEL, "vocab.json")).read_text())
        self.vocab = {k: v for k, v in vocab.items() if len(k) == 1}
        self.blank = vocab["<blank>"]
        self.letters = np.array(sorted(self.vocab.values()))

    @torch.inference_mode()
    def _lp(self, x):
        t = torch.from_numpy(np.ascontiguousarray(x)).to(self.dev)[None]
        return torch.log_softmax(self.model(t).logits[0].float(), -1).cpu().numpy()

    def logprobs(self, x):
        """Log-prob per frame 20 ms; audio panjang diproses per potongan dengan konteks."""
        x = (x - x.mean()) / (x.std() + 1e-7)
        sr, n = 16000, len(x)
        if n <= (CHUNK + 2 * CTX) * sr:
            return self._lp(x)
        T = n // 320
        out = []
        for s in np.arange(0, n / sr, CHUNK):
            a, b = max(0.0, s - CTX), min(n / sr, s + CHUNK + CTX)
            lp = self._lp(x[int(a * sr): int(b * sr)])
            f0 = int(round((s - a) / FRAME_SEC))
            f1 = f0 + int(round(CHUNK / FRAME_SEC))
            out.append(lp[f0:f1])
        return np.concatenate(out)[:T + 1]

    def tokens(self, word):
        if word in WILDCARD_TAGS:
            return [-1]
        w = ALIGN_PRON.get(word, word).replace("-", "")
        toks = [self.vocab[c] for c in w if c in self.vocab]
        if not toks:
            raise ValueError(f"kata tanpa huruf valid: {word!r}")
        return toks

    def align(self, x, words):
        """Viterbi CTC. Return list of (start_sec, end_sec, score) per kata, relatif ke x."""
        lp = self.logprobs(x)
        T = lp.shape[0]
        # wildcard ([nama]/[unclear]) boleh memancarkan huruf apa pun, atau "blank" dengan
        # penalti kecil, agar bisa menutup seluruh ucapan yang disamarkan
        wild = np.maximum(lp[:, self.letters].max(1), lp[:, self.blank] - WILD_PEN)
        labels, owner = [], []
        for wi, w in enumerate(words):
            for tk in self.tokens(w):
                labels.append(tk)
                owner.append(wi)
        L = len(labels)
        S = 2 * L + 1
        lab = np.full(S, self.blank)
        lab[1::2] = labels
        em = np.where(lab[None, :] == -1, wild[:, None], lp[:, np.clip(lab, 0, None)])
        # izin lompat s-2 -> s: state label, berbeda dari label 2 state sebelumnya
        skip = np.zeros(S, bool)
        skip[3::2] = lab[3::2] != lab[1:-2:2]
        skip[3::2] |= lab[3::2] == -1
        if T < L:
            raise ValueError("jendela audio terlalu pendek untuk teks")
        NEG = -1e9
        dp = np.full(S, NEG)
        dp[0], dp[1] = em[0, 0], em[0, 1]
        bp = np.zeros((T, S), np.int8)
        for t in range(1, T):
            a = dp
            b = np.concatenate([[NEG], dp[:-1]])
            c = np.where(skip, np.concatenate([[NEG, NEG], dp[:-2]]), NEG)
            stack = np.stack([a, b, c])
            arg = stack.argmax(0)
            dp = stack[arg, np.arange(S)] + em[t]
            bp[t] = arg
        s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
        if dp[s] < NEG / 2:
            raise ValueError("alignment gagal (jendela terlalu pendek)")
        path = np.empty(T, int)
        for t in range(T - 1, -1, -1):
            path[t] = s
            s -= int(bp[t, s])
        res = [[None, None, []] for _ in words]
        for t, st in enumerate(path):
            if st % 2 == 1:
                wi = owner[st // 2]
                if res[wi][0] is None:
                    res[wi][0] = t
                res[wi][1] = t + 1
                res[wi][2].append(em[t, st])
        out = []
        for a, b, sc in res:
            out.append((a * FRAME_SEC, b * FRAME_SEC, float(np.exp(np.mean(sc)))))
        return out


# ---------------------------------------------------------------- io helpers
def load_reviewed(path):
    units = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        uid, st, en, spk, text, *rest = line.split("\t")
        text = text.strip()
        if not text:
            continue
        units.append(dict(id=uid, start=float(st), end=float(en), spk=spk, text=text,
                          topic=rest[0].strip() if rest else ""))
    return units


def check_text(text):
    for tok in text.split():
        if tok.startswith("["):
            if tok not in WILDCARD_TAGS | EVENT_TAGS:
                raise ValueError(f"tag tidak dikenal: {tok}")
        elif not re.fullmatch(r"[a-z']+(-[a-z']+)*", tok):
            raise ValueError(f"token tidak sesuai panduan: {tok!r} dalam: {text}")


def topic_map():
    m = {}
    with open(ROOT / "metadata" / "metadata.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m[r["File"]] = r["Topic"]
    return m


def write_textgrid(path, dur, tiers):
    def esc(s):
        return s.replace('"', '""')
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
             "xmin = 0", f"xmax = {dur:.3f}", "tiers? <exists>", f"size = {len(tiers)}", "item []:"]
    for ti, (name, ivs) in enumerate(tiers, 1):
        # isi celah dengan interval kosong
        full, t = [], 0.0
        for a, b, lab in sorted(ivs):
            a, b = max(a, t), min(b, dur)
            if b <= a:
                continue
            if a > t + 1e-6:
                full.append((t, a, ""))
            full.append((a, b, lab))
            t = b
        if t < dur - 1e-6:
            full.append((t, dur, ""))
        lines += [f"    item [{ti}]:", '        class = "IntervalTier"', f'        name = "{name}"',
                  "        xmin = 0", f"        xmax = {dur:.3f}", f"        intervals: size = {len(full)}"]
        for k, (a, b, lab) in enumerate(full, 1):
            lines += [f"        intervals [{k}]:", f"            xmin = {a:.3f}", f"            xmax = {b:.3f}",
                      f'            text = "{esc(lab)}"']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main build
def split_long(ws):
    if ws[-1]["end"] - ws[0]["start"] <= MAX_UTT or len(ws) < 2:
        return [ws]
    gaps = [ws[i + 1]["start"] - ws[i]["end"] for i in range(len(ws) - 1)]
    # utamakan jeda terpanjang di bagian tengah (hindari potongan sangat pendek)
    n = len(ws)
    cand = sorted(range(n - 1), key=lambda i: (-(gaps[i] + 1e-3 * min(i, n - 2 - i))))
    k = cand[0]
    return split_long(ws[: k + 1]) + split_long(ws[k + 1:])


def build_file(stem, units, aligner, events, topic):
    x = read_wav(AUDIO / stem[:3] / f"{stem}.wav")
    dur = len(x) / 16000
    words = []  # dict: word, start, end, score, spk, unit, tags_after
    # Unit berurutan dari speaker yang sama (jeda < BLOCK_GAP) di-align sekaligus sebagai satu
    # blok, sehingga kata yang salah ditaruh ASR di unit sebelah tetap mendapat waktu yang benar.
    blocks = []
    for u in units:
        check_text(u["text"])
        last = blocks[-1] if blocks else None
        if (last and last[-1]["spk"] == u["spk"] and u["start"] - last[-1]["end"] < BLOCK_GAP
                and u["end"] - last[0]["start"] < MAX_BLOCK):
            last.append(u)
        else:
            blocks.append([u])
    for bi, blk in enumerate(blocks):
        toks, owners = [], []
        for u in blk:
            for t in u["text"].split():
                toks.append(t)
                owners.append(u)
        sp_idx = [i for i, t in enumerate(toks) if t not in EVENT_TAGS]
        if not sp_idx:
            continue
        spoken = [toks[i] for i in sp_idx]
        st, en = blk[0]["start"], blk[-1]["end"]
        prev_end = blocks[bi - 1][-1]["end"] if bi else 0.0
        next_start = blocks[bi + 1][0]["start"] if bi + 1 < len(blocks) else dur
        a = max(0.0, st - 0.4, min(st, (prev_end + st) / 2))
        b = min(dur, en + 0.4, max(en, (en + next_start) / 2))
        need = sum(len(aligner.tokens(w)) for w in spoken) * 2 * FRAME_SEC
        if b - a < need:
            a, b = max(0.0, a - need / 2), min(dur, b + need / 2)
        al = aligner.align(x[int(a * 16000): int(b * 16000)], spoken)
        base = len(words)
        for j, (w, (s, e, sc)) in enumerate(zip(spoken, al)):
            u = owners[sp_idx[j]]
            words.append(dict(word=w, start=round(a + s, 3), end=round(a + e, 3), score=round(sc, 3),
                              spk=u["spk"], unit=u["id"], topic=u["topic"], tags=[]))
        # tag event dari review ditempel setelah kata yang mendahuluinya
        for i, t in enumerate(toks):
            if t in EVENT_TAGS:
                k = sum(1 for q in sp_idx if q < i) - 1
                if k < 0:
                    words[base].setdefault("pre", []).append(t)
                else:
                    words[base + k]["tags"].append(t)
    # CTC bersifat "peaky": batas kata hanya di frame puncak. Perluas tiap kata ke kiri/kanan
    # selama frame masih bersuara (energi > noise floor + 10 dB), maksimal EXTEND_MAX.
    fr = x[: len(x) // 320 * 320].reshape(-1, 320)
    e = 10 * np.log10((fr.astype(np.float64) ** 2).mean(1) + 1e-12)
    voiced = e > np.percentile(e, 10) + 10
    def is_v(t):
        i = int(t / FRAME_SEC)
        return 0 <= i < len(voiced) and voiced[i]
    words.sort(key=lambda w: w["start"])
    for i, w in enumerate(words):
        lo = words[i - 1]["end"] if i else 0.0
        hi = words[i + 1]["start"] if i + 1 < len(words) else dur
        s0 = w["start"]
        while s0 - FRAME_SEC >= max(lo, w["start"] - EXTEND_MAX) and is_v(s0 - FRAME_SEC):
            s0 -= FRAME_SEC
        e0 = w["end"]
        while e0 + FRAME_SEC <= min(hi, w["end"] + EXTEND_MAX) and is_v(e0):
            e0 += FRAME_SEC
        w["start"], w["end"] = round(s0, 3), round(e0, 3)
    # jamin urutan monoton (tanpa tumpang tindih)
    for p, q in zip(words, words[1:]):
        if q["start"] < p["end"]:
            mid = round((p["end"] + q["start"]) / 2, 3)
            p["end"], q["start"] = mid, mid
    # susun utterance
    utts, cur = [], []
    for w in words:
        if cur and (w["spk"] != cur[-1]["spk"] or w["topic"] != cur[-1]["topic"]
                    or w["start"] - cur[-1]["end"] >= MAX_GAP):
            utts.append(cur)
            cur = []
        cur.append(w)
    if cur:
        utts.append(cur)
    utts = [p for u in utts for p in split_long(u)]

    ev = events.get(stem, [])
    rows, wrows, tiers_u, tiers_s, tiers_w, tiers_e = [], [], [], [], [], []
    for n, u in enumerate(utts, 1):
        uid = f"{stem[:-4]}_{n:04d}"
        if uid in SPK_OVERRIDE:  # koreksi Speaker ID hasil cek speaker embedding
            for w in u:
                w["spk"] = SPK_OVERRIDE[uid]
        st, en = u[0]["start"], u[-1]["end"]
        # event otomatis yang jatuh di dalam / menempel utterance
        auto = [e for e in ev if e["end"] > st - 0.3 and e["start"] < en + 0.3]
        for e in auto:
            # sisipkan tag di kata terdekat sebelum event
            idx = max([i for i, w in enumerate(u) if w["start"] <= e["start"]] or [0])
            tag = f"[{e['label']}]"
            if e["start"] < u[0]["start"]:
                u[0].setdefault("pre", [])
                if tag not in u[0]["pre"]:
                    u[0]["pre"].append(tag)
            elif tag not in u[idx]["tags"]:
                u[idx]["tags"].append(tag)
        toks = []
        for w in u:
            toks += w.get("pre", []) + [w["word"]] + w["tags"]
        text = " ".join(toks)
        evs = []
        for t in re.findall(r"\[(noise|laugh|cough|overlap)\]", text):
            if t not in evs:
                evs.append(t)
        if any(w["word"] == "[unclear]" for w in u):
            evs.append("unclear")
        rows.append(dict(Utterance_ID=uid, File=f"{stem}.wav", Speaker=u[0]["spk"], Start=f"{st:.2f}",
                         End=f"{en:.2f}", Duration=f"{en - st:.2f}", Transcription=text,
                         **{"Language/Dialect": LANG}, Topic=u[0]["topic"] or topic, Events=";".join(evs)))
        tiers_u.append((st, en, text))
        tiers_s.append((st, en, u[0]["spk"]))
        for i, w in enumerate(u, 1):
            wrows.append(dict(Utterance_ID=uid, File=f"{stem}.wav", Speaker=w["spk"], Word_Index=i,
                              Start=f"{w['start']:.3f}", End=f"{w['end']:.3f}", Word=w["word"],
                              Align_Score=f"{w['score']:.3f}"))
            tiers_w.append((w["start"], w["end"], w["word"]))
    for e in ev:
        tiers_e.append((e["start"], e["end"], e["label"]))
    (ANN / "textgrid").mkdir(parents=True, exist_ok=True)
    write_textgrid(ANN / "textgrid" / f"{stem}.TextGrid", dur,
                   [("utterance", tiers_u), ("speaker", tiers_s), ("word", tiers_w), ("event", tiers_e)])
    return rows, wrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speakers", nargs="+")
    ap.add_argument("--reviewed", required=True)
    ap.add_argument("--events")
    ap.add_argument("--speaker-override", help="CSV Utterance_ID,Speaker (koreksi hasil speaker_check)")
    ap.add_argument("--flags", help='JSON {"code_switch_words": [...]} untuk label code-switch')
    args = ap.parse_args()
    rev = Path(args.reviewed)
    if args.speaker_override:
        for r in csv.DictReader(open(args.speaker_override, encoding="utf-8")):
            SPK_OVERRIDE[r["Utterance_ID"]] = r["Speaker"]
    events = json.loads(Path(args.events).read_text(encoding="utf-8")) if args.events else {}
    cs = json.loads(Path(args.flags).read_text(encoding="utf-8")) if args.flags else {}
    cs_words = set(cs.get("code_switch_words", []))
    topics = topic_map()
    aligner = Aligner()
    rows, wrows = [], []
    for spk in args.speakers:
        for p in sorted(rev.glob(f"{spk}_T*.tsv")):
            stem = p.stem
            r, w = build_file(stem, load_reviewed(p), aligner, events, topics.get(f"{stem}.wav", ""))
            # tandai code-switch
            ids = {x["Utterance_ID"] for x in w if x["Word"] in cs_words}
            for x in r:
                if x["Utterance_ID"] in ids:
                    x["Events"] = ";".join([e for e in x["Events"].split(";") if e] + ["code-switch"])
            rows += r
            wrows += w
            print(stem, len(r), "utterances", len(w), "words", flush=True)
    ANN.mkdir(exist_ok=True)
    cols = ["Utterance_ID", "File", "Speaker", "Start", "End", "Duration", "Transcription",
            "Language/Dialect", "Topic", "Events"]
    with open(ANN / "annotation.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    with open(ANN / "annotation_words.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["Utterance_ID", "File", "Speaker", "Word_Index", "Start",
                                            "End", "Word", "Align_Score"])
        wr.writeheader()
        wr.writerows(wrows)


if __name__ == "__main__":
    main()
