"""QC teknis untuk file audio korpus.

Mengukur per file: format (codec/sample rate/bit depth/channel), durasi, peak dan
RMS (dBFS), jumlah sampel clipping, noise floor, perkiraan SNR, rasio bagian
bersuara, dan apakah awal/akhir file kemungkinan terpotong di tengah ucapan.
Juga mencari posisi setiap potongan topik di rekaman penuh (SNN.wav) untuk
mendeteksi potongan yang tumpang tindih atau bagian yang tidak masuk potongan mana pun.

Pemakaian:
    python utils/qc_audio.py S01 S02 S03 S04 S05 S07
Output: documentation/qc/qc_technical.csv dan documentation/qc/segment_offsets.csv
"""
import csv
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio"
OUT = ROOT / "documentation" / "qc"

FRAME = 400   # 25 ms @16 kHz
HOP = 160     # 10 ms
CLIP_LEVEL = 32767 * 0.999


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        info = dict(sr=w.getframerate(), ch=w.getnchannels(), bits=w.getsampwidth() * 8,
                    n=w.getnframes())
        x = np.frombuffer(w.readframes(info["n"]), dtype=np.int16)
    return x, info


def db(v):
    return 20 * np.log10(max(v, 1e-9) / 32768.0)


def frame_rms(x):
    if len(x) < FRAME:
        return np.array([np.sqrt(np.mean(x.astype(np.float64) ** 2))])
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    f = x[idx].astype(np.float64)
    return np.sqrt(np.mean(f ** 2, axis=1))


def clip_runs(x):
    """Jumlah sampel clipping dan jumlah run >= 3 sampel berturut-turut."""
    c = np.abs(x.astype(np.int32)) >= CLIP_LEVEL
    total = int(c.sum())
    if total == 0:
        return 0, 0
    d = np.diff(np.concatenate([[0], c.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    runs = int(np.sum((ends - starts) >= 3))
    return total, runs


def analyse(path):
    x, info = read_wav(path)
    dur = info["n"] / info["sr"]
    rms = frame_rms(x)
    rms_db = np.array([db(v) for v in rms])
    noise = float(np.percentile(rms_db, 10))
    speech = float(np.percentile(rms_db, 95))
    voiced = float(np.mean(rms_db > noise + 12))
    total_clip, runs = clip_runs(x)
    edge = int(0.15 * info["sr"])
    head = db(np.sqrt(np.mean(x[:edge].astype(np.float64) ** 2))) if len(x) > edge else speech
    tail = db(np.sqrt(np.mean(x[-edge:].astype(np.float64) ** 2))) if len(x) > edge else speech
    return {
        "file": path.name,
        "codec": "pcm_s16le" if info["bits"] == 16 else f"pcm_{info['bits']}",
        "sample_rate": info["sr"],
        "bit_depth": info["bits"],
        "channels": info["ch"],
        "duration_sec": round(dur, 2),
        "peak_dbfs": round(db(np.abs(x.astype(np.int32)).max()), 2),
        "rms_dbfs": round(db(np.sqrt(np.mean(x.astype(np.float64) ** 2))), 2),
        "clipped_samples": total_clip,
        "clip_runs_ge3": runs,
        "noise_floor_dbfs": round(noise, 1),
        "speech_p95_dbfs": round(speech, 1),
        "snr_est_db": round(speech - noise, 1),
        "voiced_ratio": round(voiced, 2),
        "dc_offset": round(float(np.mean(x)) / 32768.0, 5),
        "head_dbfs": round(head, 1),
        "tail_dbfs": round(tail, 1),
        "head_cut": head > noise + 15,
        "tail_cut": tail > noise + 15,
    }


def locate(segment, full, sr):
    """Offset (detik) potongan di rekaman penuh, via korelasi silang 4 detik pertama."""
    probe = segment[: sr * 4].astype(np.float64)
    if len(probe) < sr // 2:
        probe = segment.astype(np.float64)
    f = full.astype(np.float64)
    n = 1 << int(np.ceil(np.log2(len(f) + len(probe))))
    corr = np.fft.irfft(np.fft.rfft(f, n) * np.conj(np.fft.rfft(probe, n)), n)[: len(f) - len(probe) + 1]
    # normalisasi energi jendela
    c2 = np.concatenate([[0], np.cumsum(f ** 2)])
    win = np.sqrt(np.maximum(c2[len(probe):] - c2[: -len(probe)], 1e-9))[: len(corr)]
    score = corr / (win * np.sqrt(np.sum(probe ** 2)) + 1e-9)
    k = int(np.argmax(score))
    return k / sr, float(score[k])


def main(speakers):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, offs = [], []
    for spk in speakers:
        files = sorted((AUDIO / spk).glob("*.wav"))
        full_path = AUDIO / spk / f"{spk}.wav"
        full, finfo = read_wav(full_path)
        for p in files:
            r = analyse(p)
            r["speaker"] = spk
            rows.append(r)
            if p == full_path:
                continue
            seg, info = read_wav(p)
            start, score = locate(seg, full, info["sr"])
            offs.append({"file": p.name, "speaker": spk, "start_in_full": round(start, 3),
                         "end_in_full": round(start + len(seg) / info["sr"], 3),
                         "match_score": round(score, 4)})
        print(f"{spk} done", flush=True)

    with open(OUT / "qc_technical.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["speaker"] + [k for k in rows[0] if k != "speaker"])
        w.writeheader()
        w.writerows(rows)

    # cek tumpang tindih / celah antar potongan
    for spk in speakers:
        seg = sorted([o for o in offs if o["speaker"] == spk], key=lambda o: o["start_in_full"])
        for a, b in zip(seg, seg[1:]):
            gap = round(b["start_in_full"] - a["end_in_full"], 3)
            b["gap_from_prev"] = gap
        if seg:
            seg[0]["gap_from_prev"] = seg[0]["start_in_full"]
    with open(OUT / "segment_offsets.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["speaker", "file", "start_in_full", "end_in_full",
                                           "gap_from_prev", "match_score"])
        w.writeheader()
        w.writerows(sorted(offs, key=lambda o: (o["speaker"], o["start_in_full"])))


if __name__ == "__main__":
    main(sys.argv[1:] or ["S01", "S02", "S03", "S04", "S05", "S07"])
