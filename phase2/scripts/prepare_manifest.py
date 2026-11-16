import argparse
from collections import Counter
from itertools import combinations
import json

from .dataset import (PHASE2, ROOT, CORPUS, FIELDS, collect, source_snapshot,
                      write_csv, write_json, read_csv, segment_pcm)
from .markdown_utils import join_paragraph_lines


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
        if config["source_fingerprint"] != snapshot["fingerprint"]:
            raise ValueError("Sumber berubah. Buat versi dataset/split baru, jangan timpa split_v1")
    else:
        speakers, score = choose_split(stats)
        config = {"version": "palembang_internal_v1", "source_commit": snapshot["commit"],
                  "source_fingerprint": snapshot["fingerprint"], "speakers": speakers,
                  "selection": "enumerate all 7/2/2; native+non-native and L+P in every split; minimize squared duration and utterance share deviations from speaker share; lexical tie-break",
                  "objective": score, "seed": None,
                  "status": "internal split fixed; Owner 5 review and external corpus integration pending"}
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
    write_json(PHASE2 / "data/source_snapshot.json", snapshot)
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
    # Teks identik bisa muncul antar-speaker read speech; bukan audio duplikat.
    texts = {split: {r["transcript"] for r in rows if r["split"] == split} for split in split_stats}
    overlaps = {f"{a}/{b}": len(texts[a] & texts[b]) for a, b in combinations(texts, 2)}
    summary = {"source_audio_files": len(audio), "metadata_files": len(metadata),
               "annotation_rows": len(annotations), "accepted": len(rows), "excluded": len(excluded),
               "exclusion_reasons_nonexclusive": dict(reasons), "splits": split_stats,
               "audio_errors": [r for r in audio if r["error"]], "annotation_warnings": issues,
               "exact_transcript_overlap_between_splits": overlaps,
               "leakage_errors": errors}
    write_json(PHASE2 / "data/dataset_stats.json", summary)
    make_docs(summary, config, audio, stats)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_docs(summary, config, audio, stats):
    table = "| Split | Speaker | Ujaran | Menit segmen |\n|---|---|---:|---:|\n"
    for split, v in summary["splits"].items():
        table += f"| {split} | {', '.join(v['speakers'])} | {v['utterances']} | {v['duration_sec']/60:.2f} |\n"
    source_only = [r["filename"] for r in audio if "_T" not in r["filename"]]
    inventory = f"""# Audit korpus Fase 1

Snapshot commit: `{config['source_commit']}`. Hash sumber tersedia di
`phase2/data/source_snapshot.json`; hash ini mengikat isi file lokal yang diaudit.

- WAV ditemukan: {summary['source_audio_files']}; metadata: {summary['metadata_files']} file.
- Anotasi: {summary['annotation_rows']} segmen; diterima: {summary['accepted']}; dikecualikan: {summary['excluded']}.
- Speaker utama: {len(stats)}; nama asli tidak dimasukkan ke manifest.
- File sumber panjang yang tidak dipakai langsung: {', '.join(source_only)}.
- Semua WAV dibaca penuh untuk cek header, payload, sinyal nol, dan hash.
- Kesalahan format/audio: {len(summary['audio_errors'])}; warning durasi anotasi: {len(summary['annotation_warnings'])}.

Rekaman panjang dan rekaman per topik tidak dihitung sebagai sampel terpisah.
Anotasi menunjuk rekaman per topik. `documentation/qc/segment_offsets.csv`
menyediakan pemetaan ke rekaman panjang; pemeriksaan kecocokan sampel ada
di `phase2/data/source_relations.csv` setelah menjalankan audit tambahan.

## Kebijakan pengecualian

Reject dikeluarkan satu file beserta seluruh segmennya. OTHER dikeluarkan karena
pengguna mengonfirmasi bahwa itu orang lain, bukan S04. Segmen dengan transkripsi
tidak jelas ([unclear]/event unclear), kosong, hanya tag non-ucapan, batas invalid,
atau audio rusak/nol tidak dipakai. Semua alasan per ujaran ada di
`data/manifests/excluded.csv`; alasan dapat lebih dari satu, jangan dijumlahkan
sebagai jumlah segmen. Duplikat PCM satu speaker hanya memakai ID pertama;
jika speaker berbeda, semua anggota grup ditahan karena identitas ambigu.

`[nama]`, code-switch, dan event cough/noise yang masih punya kata dipertahankan
dan ditandai, bukan dihapus otomatis. Orang 2/5 perlu menyepakati penanganan tag
dan ejaan untuk training/scoring; transkripsi asli tidak diubah di tahap ini.

{table}
Durasi dihitung dari jumlah sampel pada batas anotasi, bukan panjang file penuh.
Jumlah durasi segmen dapat menghitung bagian overlap lebih dari sekali.
Audit teknis tidak menjamin kebenaran transkripsi; belum ada listening baru.
QC lama ada di `Korpus_Bahasa_Palembang/documentation/qc/`.

## Catatan sumber panjang vs file per topik

Audit sampel ulang menemukan 41/46 file topik sama persis dengan rentang sumber
panjang. Empat file berbeda satu sampel (S01_T05, S02_T01, S03_T04, S06_T02).
S05_T05 cocok pada bagian awal, tetapi 307748 sampel terakhir (sekitar 19.23 detik)
melewati akhir S05.wav. Jangan mengganti path anotasi dengan rekaman panjang atau
menganggap keduanya identik. File per topik menjadi sumber kanonik manifest karena
itulah yang dirujuk annotation.csv; masalah hubungan sumber ini dicatat untuk
review Orang 2. Rincian terukur ada di source_relations.csv.

Sembilan segmen diterima berdurasi di bawah 250 ms; tidak otomatis ditolak karena
ujaran kata pendek dapat valid. Daftar review juga mencakup tag/event yang masih
punya transkripsi. Tidak ada segmen diterima yang beririsan dengan OTHER.
"""
    card = f"""# Dataset Palembang internal v1

Sumber: https://github.com/aliyahusnaf/TugasBesarSpeech, commit `{config['source_commit']}`.
Manifest memakai data Fase 1 yang lolos kebijakan di bagian audit di dokumen ini.
Versi split: `{config['version']}`. File Fase 1 tetap utuh.

{table}
## Strategi split

Pemisahan berdasarkan speaker, termasuk validation. Semua segmen dari file/speaker
yang sama tetap satu split. Setiap split memuat native dan non-native serta L/P.
Pilih kombinasi 7/2/2 dengan enumerasi deterministik, meminimalkan jumlah kuadrat
selisih porsi durasi dan jumlah ujaran terhadap porsi speaker (7/11, 2/11, 2/11).
Jika skor sama, gunakan urutan ID. Tidak memakai prediksi, WER, atau CER untuk
memilih speaker. Tidak ada seed karena tidak ada random sampling.
Pembagian disimpan di `configs/split_v1.json` dan dipakai ulang, bukan diacak lagi.

## Batasan

Hanya 11 speaker dan dua speaker per held-out set; hasil peka terhadap pilihan
speaker. Native berjenis spontaneous, non-native read: kedua faktor terikat dan
pengaruhnya tidak bisa dipisahkan begitu saja. Kalimat sama dapat muncul pada
read speech antar-speaker; overlap teks dicatat di `data/dataset_stats.json`,
bukan dianggap generalisasi ke kalimat baru. Ini split internal, belum mencakup
korpus kelompok lain. Review Orang 5 masih diperlukan sebelum eksperimen final.

README Fase 1 menyatakan informed consent dan penggunaan akademik non-komersial.
Formulir consent tidak disalin ke Fase 2; belum dilakukan penilaian ulang ruang
lingkup izin distribusi. Konfirmasi dahulu untuk korpus tambahan atau publikasi baru.

Kecocokan metadata speaker tidak membuktikan tidak ada suara latar/orang lain
yang belum dianotasi. Tidak ada verifikasi identitas secara biometrik/listening
baru. Pemisahan speaker disimpulkan dari ID dan anotasi yang tersedia.

## Kontrak data

Satu baris = satu ujaran. `audio_path` relatif root repo TugasBesarSpeech.
`start_sec`/`end_sec` relatif file itu, akhir eksklusif; konversi batas sampel
dengan `round(seconds * sample_rate)`. Jangan beri seluruh file ke model dengan
transkripsi satu segmen. `transcript` persis sumber; normalisasi milik Orang 2.
`pcm_sha256` adalah hash byte PCM rentang segmen. Label native/gender/speech_type
diambil dari metadata, bukan ditebak dari suara. Kolom `text_tags`/`events`
diteruskan agar kebijakan scoring bisa ditetapkan secara terbuka.
"""
    path = PHASE2 / "docs/dataset_preparation.md"
    begin, end = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"
    generated = "\n\n".join(
        "\n".join("#" + line if line.startswith("#") else line for line in section.splitlines())
        for section in [inventory, card])
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing.count(begin) != 1 or existing.count(end) != 1:
            raise ValueError("Penanda bagian otomatis di dokumentasi tidak lengkap")
        prefix, rest = existing.split(begin, 1)
        _, suffix = rest.split(end, 1)
        content = prefix + begin + "\n" + generated + "\n" + end + suffix
    else:
        content = "# Dokumentasi Persiapan Dataset\n\n" + begin + "\n" + generated + "\n" + end + "\n"
    path.write_text(join_paragraph_lines(content), encoding="utf-8")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Audit dan buat manifest internal Palembang").parse_args()
    prepare()
