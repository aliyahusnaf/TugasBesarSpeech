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
python -m phase2.src.infer --run-id local_whisper_small_smoke_001 --limit 3
python -m phase2.src.infer --run-id local_whisper_small_baseline_001
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
ini hanya membuat prediksi validation, tidak memilih model dengan test. Scorer WER/CER,
training/fine-tuning, dan demo integrasi akhir tetap milik Orang 4/5/6.

`predictions.csv` tetap tersedia lokal dan diabaikan Git: smoke run menunjukkan
model menghasilkan nama pada ujaran yang referensinya dimasking `[nama]`.
Jangan mempublikasikan nama yang dipulihkan model. Config/log dan contoh tanpa
identitas dapat direview lewat PR; distribusi prediksi lengkap perlu review
hak akses oleh Orang 1/6. File mentah lokal tidak disensor atau ditimpa.

Bukti aktual: [ringkasan tujuh run](results/baseline/README.md), termasuk baseline
235/235 ujaran dan reload ketiga konfigurasi. Bahan laporan Orang 3:
[metode dan baseline](report/model_baseline.md).

### Variasi terkontrol untuk Orang 4 (T-02 / X-01)

```sh
python -m phase2.src.infer --config phase2/configs/experiments/whisper_small_beam5.json --run-id local_whisper_small_beam5_001
python -m phase2.src.infer --config phase2/configs/experiments/whisper_small_rms_trim.json --run-id local_whisper_small_rms_trim_001
```

Beam 5 menguji apakah pencarian lebih luas memperbaiki transkripsi dibanding beam
1. Varian kedua menguji paket preprocessing yang sudah disediakan Orang 2
(RMS + trim) dibanding tanpa transformasi; hasilnya tidak dapat memisahkan efek
RMS dari trimming. Semua faktor model/decoder lain tetap. Orang 5 membandingkan
WER/CER pada ID validation yang sama dengan kebijakan scoring tetap, termasuk
penanganan `[nama]`. Jangan memilih pemenang dari perkiraan atau hanya contoh bagus.

### Provenance dan keterbatasan

Validator manifest, audit, dan generator menerima perbedaan LF/CRLF pada CSV/JSON, sesuai kebijakan runner baseline. Perubahan isi teks, inventori file, dan byte audio tetap ditolak. Snapshot serta split lama tidak diubah. Perbedaan line ending dicatat oleh runner di `run.json`; PCM segmen tetap diperiksa sebelum inference. Review leakage Orang 5 tetap diperlukan.

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

## Training dan eksperimen — Orang 4

Pekerjaan Orang 4 (T-01, X-02, X-03, F-01, F-06) adalah fine-tuning Whisper small
pada data train, eksperimen terkontrol, dan pemilihan model final di validation.
Test tidak dibaca oleh skrip mana pun di bagian ini.

- Bahan laporan: [training dan eksperimen](report/training_experiments.md)
- Log semua run, termasuk yang dibatalkan: [experiment log](docs/experiment_log.md)
- Serah terima ke Orang 5: [handover](docs/handover_owner5.md)
- Ringkasan angka: [results/experiments/README.md](results/experiments/README.md),
  `results/summary.csv`, `results/training_summary.csv`,
  `results/experiments/comparison.json`

**Model final:** checkpoint `ft_lr1e-5_001` (epoch 4), config inference
`phase2/configs/experiments/finetuned_ft_lr1e-5_001_beam5.json`. WER validation 26,18%, CER 9,22%
(metrik seleksi Orang 4).

`predictions.csv` dan bobot model tidak masuk Git: prediksi dapat memuat nama
yang diucapkan pada ujaran `[nama]`, dan bobot berukuran ±466 MB per run. Semua
hasil di bawah dapat dibuat ulang dengan perintah berikut, dijalankan dari root repo.

### 1. Lingkungan

```sh
python -m venv phase2/.venv
source phase2/.venv/bin/activate          # Windows: phase2\.venv\Scripts\Activate.ps1
# Tanpa GPU NVIDIA, pasang PyTorch CPU dulu:
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r phase2/requirements-train.txt
python -m unittest discover -s phase2/tests -v
```

`requirements-train.txt` sudah memuat `requirements-model.txt` (inference Orang 3).
Untuk inference saja, `requirements-model.txt` cukup ditambah `jiwer` jika ingin
menjalankan skrip ringkasan.

### 2. Unduh model fine-tuned (tanpa training ulang)

Unduh folder dari Google Drive: *(link Google Drive menyusul)*

Letakkan isinya sehingga struktur menjadi:

```
phase2/models/ft_lr1e-5_001/ct2/
    config.json  model.bin  preprocessor_config.json  tokenizer.json  vocabulary.json
```

Model lain (`ft_lr5e-6_001`, `ft_lr3e-5_001`, `ft_lr1e-5_freeze_encoder_001`)
mengikuti pola yang sama. `WhisperASR` memeriksa hash SHA-256 `model.bin`
terhadap `model_sha256` di config dan menolak file yang berbeda.

```python
from phase2.src.model import WhisperASR, load_config

asr = WhisperASR(load_config("phase2/configs/experiments/finetuned_ft_lr1e-5_001_beam5.json"))
text = asr.transcribe("input.wav")      # seluruh file
# text = asr.transcribe_sample(row)     # satu baris manifest
```

### 3. Inference dan perbandingan di validation

Setiap perintah membuat `phase2/results/experiments/<run_id>/` berisi
`predictions.csv` (lokal), `config.json` dan `run.json`. ID run tidak dapat
ditimpa; hapus folder lama atau pakai ID baru. Setiap run memakan ±10–30 menit di CPU.

```sh
OUT=phase2/results/experiments
# Model pretrained (Orang 3) dengan variasi decoding/preprocessing
python -m phase2.src.infer --run-id pretrained_beam1_001 --output-root $OUT
python -m phase2.src.infer --config phase2/configs/experiments/whisper_small_beam5.json --run-id pretrained_beam5_001 --output-root $OUT
python -m phase2.src.infer --config phase2/configs/experiments/whisper_small_rms_trim.json --run-id pretrained_rms_trim_001 --output-root $OUT
# Model fine-tuned (perlu langkah 2)
python -m phase2.src.infer --config phase2/configs/experiments/finetuned_ft_lr1e-5_001.json --run-id finetuned_lr1e-5_001 --output-root $OUT
python -m phase2.src.infer --config phase2/configs/experiments/finetuned_ft_lr5e-6_001.json --run-id finetuned_lr5e-6_001 --output-root $OUT
python -m phase2.src.infer --config phase2/configs/experiments/finetuned_ft_lr3e-5_001.json --run-id finetuned_lr3e-5_001 --output-root $OUT
python -m phase2.src.infer --config phase2/configs/experiments/finetuned_ft_lr1e-5_freeze_encoder_001.json --run-id finetuned_lr1e-5_freeze_encoder_001 --output-root $OUT
python -m phase2.src.infer --config phase2/configs/experiments/finetuned_ft_lr1e-5_001_beam5.json --run-id finetuned_lr1e-5_beam5_001 --output-root $OUT

# Tabel WER/CER, uji bootstrap, dan grafik kurva training
python -m phase2.scripts.summarize_experiments
python -m phase2.scripts.compare_runs --reference finetuned_lr1e-5_001
python -m phase2.scripts.plot_training_curves
```

Windows PowerShell: ganti `OUT=...` dengan `$OUT = "phase2/results/experiments"`.

### 4. Training ulang (opsional, perlu GPU)

Smoke test jalur training di CPU (whisper-tiny, 8 ujaran, ±2 menit):

```sh
python -m phase2.src.train --config phase2/configs/train/smoke_whisper_tiny.json --run-id smoke_tiny_001 --output-root /tmp/owner4_smoke
```

Satu run penuh di mesin dengan GPU CUDA (±15 menit di T4):

```sh
python -m phase2.src.train --config phase2/configs/train/whisper_small_ft_lr1e-5.json --run-id ft_lr1e-5_002
```

Hasilnya `phase2/results/training/<run_id>/` (`history.csv`, `run.json`, prediksi
per epoch) dan `model/ct2/` (checkpoint terpilih, format inference).

**Lewat Kaggle (dipakai untuk hasil di laporan).** Perlu Kaggle CLI dengan token
API di `~/.kaggle/` dan akun Kaggle yang sudah verifikasi nomor HP (agar kernel
boleh memakai internet). Kernel mengambil commit dasar dari GitHub dan commit
lokal dari git bundle, jadi semua perubahan harus sudah di-commit, dan branch
dasar (`origin/phase2`) sudah ada di GitHub.

```sh
python -m phase2.scripts.kaggle_job prepare --username <USERNAME_KAGGLE>
cd phase2/.kaggle_build/dataset
kaggle datasets create -p .                       # berikutnya: kaggle datasets version -p . -m "update"
cd ../kernel
kaggle kernels push -p .
kaggle kernels status <USERNAME_KAGGLE>/palembang-asr-whisper-finetune
cd ../../..
kaggle kernels output <USERNAME_KAGGLE>/palembang-asr-whisper-finetune -p <FOLDER_UNDUHAN>
python -m phase2.scripts.kaggle_job collect --download <FOLDER_UNDUHAN>
```

Di Windows, jalankan `kaggle ... -p .` dari dalam folder tujuan; CLI Kaggle gagal
dengan path relatif. `collect` menyalin log ke `results/training/<run_id>/`, bobot
ke `models/<run_id>/ct2/`, lalu membuat config inference
`configs/experiments/finetuned_<run_id>.json`. Run ID yang sudah ada tidak ditimpa.
Job keempat run memakan ±50 menit kuota GPU Kaggle.
