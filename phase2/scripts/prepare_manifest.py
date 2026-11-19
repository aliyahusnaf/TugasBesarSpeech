import argparse
from collections import Counter
from itertools import combinations
import json

from .dataset import (PHASE2, ROOT, CORPUS, FIELDS, collect, source_snapshot,
                      write_csv, write_json, read_csv, segment_pcm, verify_source_snapshot)


def speaker_summary(rows):
    result = {}
    for row in rows:
        s = result.setdefault(row["speaker_id"], {"utterances": 0, "duration_sec": 0,
                            "native": row["native"], "speech_type": row["speech_type"],
                            "gender": row["gender"]})
        if any(s[k] != row[k] for k in ["native", "speech_type", "gender"]):
            raise ValueError(f"Metadata speaker tidak konsisten: {row['speaker_id']}")
        s["utterances"] += 1
        s["duration_sec"] += float(row["duration_sec"])
    return dict(sorted(result.items()))


def choose_split(stats):
    # Cari 7/2/2 yang mendekati porsi durasi dan ujaran, tanpa melihat hasil model.
    ids = sorted(stats)
    if len(ids) != 11:
        raise ValueError("Pemilihan otomatis ini khusus 11 speaker; tinjau ulang jika korpus berubah")
    total_duration = sum(r["duration_sec"] for r in stats.values())
    total_rows = sum(r["utterances"] for r in stats.values())
    options = []
    for test in combinations(ids, 2):
        for val in combinations([s for s in ids if s not in test], 2):
            train = tuple(s for s in ids if s not in test + val)
            parts = {"train": train, "validation": val, "test": test}
            if any({stats[s]["native"] for s in group} != {"Ya", "Tidak"}
                   or {stats[s]["gender"] for s in group} != {"L", "P"}
                   for group in parts.values()):
                continue
            score = 0
            for group in parts.values():
                target = len(group) / len(ids)
                score += (sum(stats[s]["duration_sec"] for s in group) / total_duration - target) ** 2
                score += (sum(stats[s]["utterances"] for s in group) / total_rows - target) ** 2
            options.append((score, train, val, test))
    if not options:
        raise ValueError("Tidak ada split yang memenuhi syarat; perlu tinjau strategi")
    score, train, val, test = min(options)
    return {"train": list(train), "validation": list(val), "test": list(test)}, score


def validate_rows(rows, config, verify_pcm=True):
    import hashlib
    import math
    import wave
    errors, seen, hashes, intervals = [], set(), {}, {}
    assignments = {}
    if set(config["speakers"]) != {"train", "validation", "test"}:
        errors.append("Nama split tidak lengkap")
    for split, speakers in config["speakers"].items():
        for s in speakers:
            if s in assignments:
                errors.append(f"Speaker lintas split: {s}")
            assignments[s] = split
    for r in rows:
        sid = r["sample_id"]
        if sid in seen:
            errors.append(f"ID dobel: {sid}")
        seen.add(sid)
        if assignments.get(r["speaker_id"]) != r["split"]:
            errors.append(f"Split speaker salah: {sid}")
        if not r["transcript"].strip() or r["speaker_id"] == "OTHER" or r["quality"].lower() == "reject":
            errors.append(f"Data terlarang masuk: {sid}")
        p = (ROOT / r["audio_path"]).resolve()
        from pathlib import Path
        if Path(r["audio_path"]).is_absolute() or not p.is_relative_to(CORPUS.resolve()):
            errors.append(f"Path di luar korpus: {sid}")
            continue
        try:
            a, b = float(r["start_sec"]), float(r["end_sec"])
            with wave.open(str(p), "rb") as w:
                if (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getcomptype()) != (16000, 1, 2, "NONE"):
                    raise ValueError("Format audio salah")
                if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= w.getnframes()/16000):
                    raise ValueError("Batas segmen salah")
            if abs(float(r["duration_sec"]) - (round(b*16000)-round(a*16000))/16000) > 1e-6:
                raise ValueError("Durasi segmen salah")
            if verify_pcm and hashlib.sha256(segment_pcm(p, a, b)).hexdigest() != r["pcm_sha256"]:
                raise ValueError("Hash segmen berubah")
        except (OSError, EOFError, wave.Error, ValueError) as exc:
            errors.append(f"{sid}: {exc}")
            continue
        if r["pcm_sha256"] in hashes:
            errors.append(f"PCM dobel: {sid} / {hashes[r['pcm_sha256']]}")
        hashes[r["pcm_sha256"]] = sid
        intervals.setdefault(r["audio_path"], []).append((a, b, sid, r["split"]))
    # Segmen beririsan dalam satu file tidak boleh berada pada split berbeda.
    for group in intervals.values():
        group.sort()
        for i, (a, b, sid, split) in enumerate(group):
            for c, d, other, other_split in group[i+1:]:
                if c >= b:
                    break
                if split != other_split:
                    errors.append(f"Overlap lintas split: {sid} / {other}")
    if {r["speaker_id"] for r in rows} != set(assignments):
        errors.append("Speaker manifest tidak sama dengan konfigurasi")
    return errors


def prepare():
    snapshot = source_snapshot()
    rows, excluded, audio, metadata, annotations, issues = collect()
    stats = speaker_summary(rows)
    config_path = PHASE2 / "configs/split_v1.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        saved = json.loads((PHASE2 / "data/source_snapshot.json").read_text(encoding="utf-8"))
        if config["source_fingerprint"] != saved["fingerprint"]:
            raise ValueError("Konfigurasi split tidak cocok dengan snapshot tersimpan")
        verify_source_snapshot(saved, current=snapshot)
    else:
        speakers, score = choose_split(stats)
        config = {"version": "palembang_internal_v1", "source_commit": snapshot["commit"],
                  "source_fingerprint": snapshot["fingerprint"], "speakers": speakers,
                  "selection": "enumerate all 7/2/2; native+non-native and L+P in every split; minimize squared duration and utterance share deviations from speaker share; lexical tie-break",
                  "objective": score, "seed": None,
                  "status": "internal split fixed"}
        saved = {**snapshot, "commit": config["source_commit"]}
    mapping = {s: split for split, ids in config["speakers"].items() for s in ids}
    for row in rows:
        row["split"] = mapping[row["speaker_id"]]
    errors = validate_rows(rows, config)
    if errors:
        raise ValueError(errors)
    snapshot["commit"] = config["source_commit"]
    if source_snapshot()["fingerprint"] != snapshot["fingerprint"]:
        raise ValueError("Sumber berubah saat proses berjalan")
    write_json(config_path, config)
    write_json(PHASE2 / "data/source_snapshot.json", saved)
    for split in config["speakers"]:
        write_csv(PHASE2 / f"data/manifests/{split}.csv", [r for r in rows if r["split"] == split], FIELDS)
    write_csv(PHASE2 / "data/manifests/excluded.csv", excluded,
              ["sample_id", "file", "speaker_id", "reasons"])
    write_csv(PHASE2 / "data/audio_inventory.csv", audio,
              ["audio_path", "filename", "sha256", "error", "sample_rate", "channels", "bit_depth", "frames", "duration_sec", "pcm_sha256"])
    speaker_rows = [{"speaker_id": s, "split": mapping[s], **v} for s, v in stats.items()]
    write_csv(PHASE2 / "data/speakers.csv", speaker_rows,
              ["speaker_id", "split", "utterances", "duration_sec", "native", "speech_type", "gender"])
    split_stats = {split: {"speakers": ids, "utterances": sum(r["split"] == split for r in rows),
                  "duration_sec": sum(float(r["duration_sec"]) for r in rows if r["split"] == split)}
                   for split, ids in config["speakers"].items()}
    reasons = Counter(reason for r in excluded for reason in r["reasons"].split(";"))
    # Teks identik bisa muncul antar-speaker read speech, bukan audio duplikat.
    texts = {split: {r["transcript"] for r in rows if r["split"] == split} for split in split_stats}
    overlaps = {f"{a}/{b}": len(texts[a] & texts[b]) for a, b in combinations(texts, 2)}
    summary = {"source_audio_files": len(audio), "metadata_files": len(metadata),
               "annotation_rows": len(annotations), "accepted": len(rows), "excluded": len(excluded),
               "exclusion_reasons_nonexclusive": dict(reasons), "splits": split_stats,
               "audio_errors": [r for r in audio if r["error"]], "annotation_warnings": issues,
               "exact_transcript_overlap_between_splits": overlaps,
               "leakage_errors": errors}
    write_json(PHASE2 / "data/dataset_stats.json", summary)
    make_docs(summary, config, stats)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_docs(summary, config, stats):
    labels = {"quality_reject": "rekaman berlabel Reject", "other_speaker": "suara orang lain (OTHER)",
              "unclear_transcript": "transkripsi tidak jelas"}
    reasons = ", ".join(f"{labels.get(k, k)} {v}" for k, v in summary["exclusion_reasons_nonexclusive"].items())
    table = "| Split | Speaker | Ujaran | Durasi (menit) |\n|---|---|---:|---:|\n"
    for split, v in summary["splits"].items():
        table += f"| {split} | {', '.join(v['speakers'])} | {v['utterances']} | {v['duration_sec'] / 60:.2f} |\n"
    overlaps = ", ".join(f"{k} {v}" for k, v in summary["exact_transcript_overlap_between_splits"].items())
    doc = f"""# Persiapan Dataset

Sumber: korpus Fase 1 (`Korpus_Bahasa_Palembang/`), commit `{config['source_commit']}`.
Hash file sumber disimpan di `data/source_snapshot.json`.

## Data yang dipakai

- {summary['annotation_rows']} ujaran teranotasi dari {len(stats)} penutur; {summary['accepted']} dipakai dan {summary['excluded']} dikeluarkan.
- Alasan dikeluarkan (satu ujaran bisa memiliki lebih dari satu alasan): {reasons}. Rincian per ujaran ada di `data/manifests/excluded.csv`.
- Semua audio berformat WAV 16 kHz, mono, 16-bit. Kesalahan format: {len(summary['audio_errors'])}.

## Pembagian data

{table}
Pembagian bersifat speaker-independent: semua ujaran dari satu penutur berada di split yang sama.
Komposisi 7/2/2 dipilih dari semua kombinasi penutur secara deterministik. Setiap split wajib
memuat penutur asli dan bukan asli serta laki-laki dan perempuan, lalu dipilih kombinasi yang
proporsi durasi dan jumlah ujarannya paling mendekati proporsi penutur. Pemilihan tidak memakai
hasil model. Pembagian disimpan di `configs/split_v1.json`.

Tidak ada penutur maupun audio (hash PCM) yang sama antar-split.

## Keterbatasan

- Hanya {len(stats)} penutur, dengan 2 penutur di validation dan 2 di test, sehingga hasil peka terhadap pilihan penutur.
- Penutur asli berbicara spontan dan penutur bukan asli membaca naskah, sehingga pengaruh kedua faktor tidak dapat dipisahkan.
- Beberapa kalimat identik muncul di lebih dari satu split karena naskah yang sama dibaca beberapa penutur ({overlaps}).

## Format manifest

`data/manifests/{{train,validation,test}}.csv` berisi satu baris per ujaran: `sample_id`, `audio_path`,
`start_sec`/`end_sec` (posisi ujaran dalam rekaman), `transcript` (sesuai anotasi), metadata penutur,
dan `pcm_sha256` (hash audio ujaran).
"""
    (PHASE2 / "docs/dataset_preparation.md").write_text(doc, encoding="utf-8")

if __name__ == "__main__":
    argparse.ArgumentParser(description="Audit dan buat manifest internal Palembang").parse_args()
    prepare()
