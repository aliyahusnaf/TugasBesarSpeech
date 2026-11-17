# Log eksperimen — Orang 4 (training & eksperimen)

Catatan kronologis semua run Orang 4, termasuk run yang dibatalkan. Angka WER/CER
di sini adalah **validation** dengan aturan seleksi
`phase2.src.selection_metrics/v1` (normalisasi Orang 2, WER/CER level korpus,
jiwer 4). Test set tidak dibaca oleh skrip mana pun di bawah ini. Angka resmi dan
evaluasi test milik Orang 5.

Semua waktu dalam UTC, tanggal 2026-11-17.

## Lingkungan

| Peran | Mesin | Software utama |
|---|---|---|
| Inference/decoding | Laptop Windows 11, i7-13620H, RAM 16 GB, tanpa GPU NVIDIA; CPU 4 thread, INT8 (`int8_float32`) | Python 3.11, faster-whisper 1.2.1, CTranslate2 4.9.0, tokenizers 0.22.2, huggingface_hub 0.36.2 |
| Fine-tuning | Kaggle Notebook, 1× Tesla T4 (16 GB), fp16 autocast | Python 3.13.16, PyTorch 2.11.0+cu130, Transformers 4.57.1, jiwer 4.0.0, CTranslate2 4.9.0 |

Versi paket lengkap setiap run ada di `run.json`. `pip freeze` lingkungan Kaggle
ada di `results/training/kaggle_job_v1/pip_freeze.txt`.

## Run

### 1. Smoke test pipeline training (lokal, CPU) — bukan hasil

- ID `smoke_tiny_001`, config `configs/train/smoke_whisper_tiny.json`. Model
  `openai/whisper-tiny`, 8 ujaran train, 4 ujaran validation, 1 epoch.
- Output di folder sementara di luar repo. Run ini hanya menguji jalur
  training → seleksi epoch → konversi CTranslate2 → dimuat `WhisperASR`.
- Status `complete` (134,8 detik). WER 4 ujaran tidak bermakna, jadi tidak
  dilaporkan.

### 2. Decoding pretrained di CPU (`results/experiments/`)

| run_id | Config | Mulai | Status |
|---|---|---|---|
| `pretrained_beam1_001` (percobaan 1) | `configs/model.json` | ±12:00 | **dibatalkan** di 55/235 |
| `pretrained_beam1_001` | `configs/model.json` | 12:15 | complete 235/235 |
| `pretrained_beam5_001` | `configs/experiments/whisper_small_beam5.json` | 12:46 | complete 235/235 |
| `pretrained_rms_trim_001` | `configs/experiments/whisper_small_rms_trim.json` | 13:14 | complete 235/235 |

- **Pembatalan.** Percobaan pertama `pretrained_beam1_001` dihentikan manual
  di 55/235. Penyebabnya: pemasangan `transformers==4.57.1` untuk training
  di venv yang sama mengganti `tokenizers` (0.23.3 → 0.22.2) dan
  `huggingface_hub` (2.4.0 → 0.36.2) saat run berjalan. Agar satu run tidak
  memakai dua lingkungan, folder parsial (status `running`) dihapus dan run
  diulang dari awal dengan ID yang sama. Semua run lengkap di tabel ini
  memakai lingkungan setelah perubahan tersebut.
- `matplotlib` dipasang ketika `pretrained_beam1_001` berjalan. Paket inference
  tidak berubah (diperiksa), tetapi daftar paket di `run.json` beam 5 dan
  rms_trim memuat matplotlib beserta dependensinya.
- Config ketiga run sama byte-per-byte dengan config Orang 3. Run beam 1 adalah
  ulangan lokal baseline `whisper_small_baseline_001` Orang 3. Prediksi Orang 3
  tidak ada di repo (diabaikan Git), jadi kesamaan prediksinya tidak diperiksa.
- `pretrained_beam1_001` tercatat dengan commit `1d7e4bc`, yang lain `9178323`.
  Commit di antaranya hanya menambah `train.py` dan `selection_metrics.py`;
  kode inference identik.

### 3. Fine-tuning di Kaggle T4 (`results/training/`)

Job `configs/train/kaggle_job_v1.json`, kode `9178323`, dijalankan berurutan dalam
satu sesi kernel (13:32–14:22, total ±50 menit). Setiap run memakai base
`openai/whisper-small` revision `973afd24965f72e36ca33b3055d56a652f456b4d`, seed 42,
755 ujaran train (S01, S02, S04, S07, S08, S09, S11; 58 menit audio) dan 235
ujaran validation (S03, S06).

| run_id | Faktor yang diubah | Epoch dijalankan | Epoch terpilih | WER val (HF greedy) | Loss val | Waktu |
|---|---|---|---|---|---|---|
| `ft_lr1e-5_001` | acuan (lr 1e-5) | 7 (early stop) | 4 | 26,87% | 0,614 | 15,2 mnt |
| `ft_lr5e-6_001` | lr 5e-6 | 7 (early stop) | 4 | 27,32% | 0,603 | 14,8 mnt |
| `ft_lr3e-5_001` | lr 3e-5 | 5 (early stop) | 2 | 27,44% | 0,627 | 10,8 mnt |
| `ft_lr1e-5_freeze_encoder_001` | encoder dibekukan | 7 (early stop) | 4 | 28,61% | 0,621 | 8,8 mnt |

- Epoch 0 (pretrained, pipeline HF yang sama): WER 48,20%, CER 18,41%, loss 1,607.
- Semua run berhenti karena early stopping (patience 3); tidak ada yang
  mencapai batas 8 epoch.
- Satu epoch training ±90 detik (full fine-tune) atau ±39 detik (encoder
  beku). Evaluasi validation ±28 detik per epoch.
- **Kendala.** Upload dataset Kaggle pertama gagal karena CLI Kaggle di Windows
  salah menangani path relatif. Upload kedua (dari dalam folder build) berhasil.
  Tidak ada kode atau hasil yang berubah.
- Bobot CTranslate2 float16 tiap run (±466 MB) disimpan di luar Git:
  `phase2/models/<run_id>/ct2/`. Hash `model.bin` dicatat di `run.json` dan
  config inference.

### 4. Inference CPU model fine-tuned (`results/experiments/`)

Setiap checkpoint terpilih didecode lewat adapter `WhisperASR` Orang 3 dengan
pengaturan yang sama seperti baseline (CPU, INT8, beam 1, `language=id`),
sehingga sebanding langsung dengan run pretrained.

| run_id | Config | Mulai | Status |
|---|---|---|---|
| `finetuned_lr1e-5_001` | `configs/experiments/finetuned_ft_lr1e-5_001.json` | 14:33 | complete |
| `finetuned_lr5e-6_001` | `configs/experiments/finetuned_ft_lr5e-6_001.json` | 14:41 | complete |
| `finetuned_lr3e-5_001` | `configs/experiments/finetuned_ft_lr3e-5_001.json` | 14:58 | complete |
| `finetuned_lr1e-5_freeze_encoder_001` | `configs/experiments/finetuned_ft_lr1e-5_freeze_encoder_001.json` | 15:22 | complete |
| `finetuned_lr1e-5_beam5_001` | `configs/experiments/finetuned_ft_lr1e-5_001_beam5.json` | 15:50 | complete |

- Beam 5 hanya diuji pada checkpoint terbaik, setelah checkpoint dipilih dengan
  beam 1. Ini faktor decoding, bukan faktor training.
- Commit kode berbeda antar-run (`3ae76cd`, `b36dedb`, `42b5385`) karena commit
  log dan skrip dibuat di sela run. Commit tersebut tidak mengubah kode
  inference.
- `elapsed_sec` di CPU dipengaruhi beban lain di laptop yang sama, jadi
  **bukan** benchmark kecepatan. Selisih waktu antar-run tidak boleh ditafsirkan
  sebagai perbedaan kecepatan model.

## Hasil (validation, 235 ujaran)

Sumber: `results/summary.csv` (dibuat `summarize_experiments`) dan
`results/experiments/comparison.json` (dibuat `compare_runs`).

| run_id | WER | CER | S | D | I | Δ WER vs acuan [CI 95%] |
|---|---|---|---|---|---|---|
| `pretrained_beam1_001` | 47,35% | 17,49% | 919 | 182 | 69 | +20,4 [+17,9; +22,9] |
| `pretrained_rms_trim_001` | 46,50% | 16,90% | 911 | 166 | 72 | +19,5 [+17,1; +22,0] |
| `pretrained_beam5_001` | 44,15% | 15,42% | 848 | 172 | 71 | +17,2 [+14,7; +19,6] |
| `finetuned_lr1e-5_freeze_encoder_001` | 28,53% | 10,23% | 523 | 70 | 112 | +1,5 [+0,1; +2,9] |
| `finetuned_lr3e-5_001` | 28,13% | 10,09% | 530 | 66 | 99 | +1,1 [−0,3; +2,5] |
| `finetuned_lr5e-6_001` | 27,52% | 9,76% | 508 | 73 | 99 | +0,5 [−0,6; +1,7] |
| `finetuned_lr1e-5_001` (acuan) | 26,99% | 9,44% | 495 | 64 | 108 | — |
| **`finetuned_lr1e-5_beam5_001`** (final) | **26,18%** | **9,22%** | 471 | 63 | 113 | −0,8 [−1,6; −0,04] |

Δ dalam poin persen. CI berasal dari paired bootstrap ujaran (10.000 sampel
ulang, seed 42). Jumlah referensi: 2.471 kata, 14.276 karakter (termasuk spasi).

## Keputusan

- **Model final: checkpoint `ft_lr1e-5_001` (epoch 4) dengan decoding beam 5.**
  Alasannya ada di [bagian laporan](../report/training_experiments.md#pemilihan-model-final).
- Pemilihan memakai validation saja. Karena validation juga dipakai untuk
  memilih epoch dan konfigurasi, WER validation model terpilih cenderung
  optimistis. Estimasi yang tidak bias adalah WER test dari Orang 5.
