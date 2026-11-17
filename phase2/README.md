# Fase 2 - ASR Bahasa Palembang

## Baseline Whisper — Orang 3 (M-01, M-02, B-03)

Teknik Whisper multilingual `small` dipilih pengguna untuk implementasi.
Konfirmasi koordinasi kelas/kelompok pembanding tetap tugas Orang 1; tidak
menjadi klaim bahwa persetujuan kelas sudah diterima. Model merupakan
encoder-decoder Transformer; checkpoint memakai konversi CTranslate2
[`Systran/faster-whisper-small`](https://huggingface.co/Systran/faster-whisper-small),
revision `536b0662742c02347bc0e980a01041f333bce120`.
Tokenizer dari snapshot yang sama, tanpa vocabulary baru. JSON mengikuti
konvensi konfigurasi yang sudah dipakai repo.

Dari root repo, gunakan Python 3.12 (lingkungan lokal yang diuji):

```sh
python -m venv phase2/.venv
source phase2/.venv/bin/activate
python -m pip install -r phase2/requirements-model.txt
python -m phase2.scripts.check_model_input
python -m unittest discover -s phase2/tests -v
python -m phase2.src.infer --run-id whisper_small_smoke_001 --limit 3
python -m phase2.src.infer --run-id whisper_small_baseline_001
```

Windows: aktivasi dengan `phase2\.venv\Scripts\Activate.ps1`.
Unduhan pertama memerlukan internet; bobot disimpan di cache Hugging Face di
luar repo. Tidak memerlukan API berbayar. Default CPU, INT8, empat thread;
perangkat/precision aktual dicatat dalam hasil. Jangan menyebut estimasi memori
upstream sebagai pengukuran lokal.

Baseline memakai `language=id` (hint Bahasa Indonesia, bukan label Palembang),
`task=transcribe`, beam 1, temperature 0, tanpa prompt, tanpa VAD tambahan dan
`condition_on_previous_text=false`. Setiap ujaran diproses mandiri. Teks referensi
tidak dimasukkan sebagai prompt. Parameter dan versi backend dicatat agar default
lain bisa ditelusuri. Tidak ada sampling acak sehingga seed dicatat `null`.

### Kontrak adapter dan hasil

```python
from phase2.src.model import WhisperASR, load_config, transcribe

config = load_config()  # phase2/configs/model.json
text = transcribe("input.wav", config)  # seluruh file untuk demo
asr = WhisperASR(config)               # muat sekali untuk batch
# text = asr.transcribe_sample(row)    # satu rentang anotasi, melalui load_sample
```

`WhisperASR` menerima sinyal mono float32 16 kHz lewat `transcribe_signal`;
`transcribe_sample` dan `transcribe` memakai preprocessing Orang 2 tepat sekali.
Jangan berikan seluruh WAV sumber untuk satu baris manifest. Prosesor Whisper
mengubah waveform menjadi fitur log-Mel; tokenizer dan decoder dipakai melalui
backend yang sama oleh Orang 4/6.

Hasil disimpan di `phase2/results/baseline/<run_id>/`:

- `predictions.csv`: `run_id,sample_id,split,reference,prediction`; referensi dan
  prediksi mentah tetap disimpan. Normalisasi/scoring milik Orang 5.
- `config.json`: konfigurasi efektif.
- `run.json`: status, jumlah selesai, revision dan hash model/tokenizer,
  versi paket, perangkat aktual, waktu eksekusi, hash manifest/preprocessing,
  snapshot/split dan commit kode. Kegagalan menyimpan traceback dan prediksi
  yang sudah selesai; status `failed` tidak boleh dilaporkan sebagai baseline lengkap.

ID run tidak dapat ditimpa; gunakan ID baru untuk pengulangan. `--limit` memilih N baris pertama validation untuk
smoke test dan dicatat; hasil subset tidak mewakili semua validation. Runner
ini hanya membaca validation, tidak memilih model dengan test. Scorer WER/CER,
training/fine-tuning, dan demo integrasi akhir tetap milik Orang 4/5/6.

`predictions.csv` tetap tersedia lokal dan diabaikan Git: smoke run menunjukkan
model menghasilkan nama pada ujaran yang referensinya dimasking `[nama]`.
Jangan mempublikasikan nama yang dipulihkan model. Config/log dan contoh tanpa
identitas dapat direview lewat PR; distribusi prediksi lengkap perlu review
hak akses oleh Orang 1/6. File mentah lokal tidak disensor atau ditimpa.

### Variasi terkontrol untuk Orang 4 (T-02 / X-01)

```sh
python -m phase2.src.infer --config phase2/configs/experiments/whisper_small_beam5.json --run-id whisper_small_beam5_001
python -m phase2.src.infer --config phase2/configs/experiments/whisper_small_rms_trim.json --run-id whisper_small_rms_trim_001
```

Beam 5 menguji apakah pencarian lebih luas memperbaiki transkripsi dibanding beam
1. Varian kedua menguji paket preprocessing yang sudah disediakan Orang 2
(RMS + trim) dibanding tanpa transformasi; hasilnya tidak dapat memisahkan efek
RMS dari trimming. Semua faktor model/decoder lain tetap. Orang 5 membandingkan
WER/CER pada ID validation yang sama dengan kebijakan scoring tetap, termasuk
penanganan `[nama]`. Jangan memilih pemenang dari perkiraan atau hanya contoh bagus.

### Provenance dan keterbatasan

Validator manifest lama membandingkan hash byte dan gagal pada checkout LF
ketika snapshot dibuat dengan CRLF. Runner baseline memeriksa seluruh inventori
sumber: hanya perubahan line ending CSV/JSON yang ekuivalen dalam memori
boleh diteruskan, dan seluruh perbedaan dicatat di `run.json`. Perubahan isi
lain menghentikan run. Setiap PCM segmen diperiksa terhadap hash manifest sebelum
inference. Snapshot, manifest dan korpus tidak ditulis ulang. Orang 1 tetap perlu
mereview portabilitas validator lama; pemeriksaan ini tidak menggantikan review
leakage Orang 5.

Referensi semula didraf dengan Whisper large-v3 lalu direview dalam Fase 1.
Proses ini perlu disebut dalam laporan karena dapat memengaruhi independensi
referensi. Hasil waveform yang valid tidak membuktikan kebenaran transkripsi,
ketepatan dialek, atau akurasi pengenalan.

Referensi implementasi: [Whisper resmi](https://github.com/openai/whisper),
[faster-whisper 1.2.1](https://github.com/SYSTRAN/faster-whisper/tree/v1.2.1).
Bobot dan kode yang dipakai mengikuti lisensi sumbernya; checkpoint card
CTranslate2 menyatakan MIT. Ini bukan eksperimen fine-tuning.

Bagian Orang 1: audit korpus, manifest per ujaran, split speaker, dan statistik. Script ini memakai Python 3.9+ dan standard library; tidak perlu model/GPU. Korpus `Korpus_Bahasa_Palembang/` hanya dibaca, tidak dipindah atau diubah.

Jalankan dari root repo `TugasBesarSpeech`:

```powershell
python -m phase2.scripts.prepare_manifest
python -m phase2.scripts.audit_phase1
python -m phase2.scripts.validate_manifest
python -m unittest discover -s phase2/tests -v
```

`prepare_manifest` membangun audit dan manifest. Pilihan speaker disimpan dalam `phase2/configs/split_v1.json`. Run berikutnya memakai pilihan itu. Jika hash sumber berubah, proses berhenti: tinjau perubahan dan buat versi dataset baru, jangan hapus konfigurasi hanya supaya proses lolos.

## File untuk anggota lain

- `data/manifests/train.csv`, `validation.csv`, `test.csv`: data yang dipakai.
- `data/manifests/excluded.csv`: ujaran yang tidak dipakai beserta alasannya.
- `data/speakers.csv`, `data/dataset_stats.json`: komposisi dan durasi split.
- `data/audio_inventory.csv`, `data/source_snapshot.json`: audit format/hash.
- `data/source_relations.csv`, `data/audit_details.json`: hubungan file panjang dan potongan, serta hal yang perlu ditinjau Orang 2.
- [Dokumentasi persiapan dataset](docs/dataset_preparation.md): audit, dataset card, keputusan, catatan koordinasi, dan bahan laporan Orang 1.
- `data/validation_report.json`: hasil pemeriksaan manifest terakhir.

## Membaca satu ujaran

**Satu baris menunjuk segmen, bukan seluruh file WAV.** Path relatif root repo. Contoh pembacaan array untuk pipeline Orang 2-4 (NumPy milik pipeline model):

```python
import csv
from pathlib import Path
import wave
import numpy as np

root = Path.cwd()  # Root repo TugasBesarSpeech.
with (root / "phase2/data/manifests/train.csv").open(encoding="utf-8", newline="") as f:
    row = next(csv.DictReader(f))

with wave.open(str(root / row["audio_path"]), "rb") as audio:
    sample_rate = audio.getframerate()
    start = round(float(row["start_sec"]) * sample_rate)
    end = round(float(row["end_sec"]) * sample_rate)
    audio.setpos(start)
    signal = np.frombuffer(audio.readframes(end - start), dtype="<i2").astype(np.float32) / 32768
text = row["transcript"]  # Teks asli; normalisasi dilakukan terpisah oleh Orang 2.
```

Orang 2 boleh membuat WAV per ujaran di lokasi baru, tetapi harus mempertahankan `sample_id`, sumber, split, dan transkripsi asli. Jangan trimming rekaman panjang sebelum memakai batas anotasi karena offset bisa bergeser. Orang 3/4 melatih di train dan memilih konfigurasi di validation. Test disimpan untuk evaluasi final Orang 5. Preprocessing dan adapter Whisper tersedia; metrik evaluasi tetap pekerjaan Orang 5.
