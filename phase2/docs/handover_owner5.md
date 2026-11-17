# Handover Orang 4 → Orang 5 (evaluasi)

Dokumen ini berisi semua yang perlu Orang 5 ketahui untuk mengevaluasi model final
di test dan menyusun tabel perbandingan validation. Detail metode ada di
[laporan training](../report/training_experiments.md). Riwayat setiap run ada di
[experiment log](experiment_log.md).

## Status test set

Test set (S05, S10; 230 ujaran) **belum pernah dibaca** oleh training, seleksi
checkpoint, maupun pemilihan konfigurasi. `train.load_rows("test", ...)` menolak
split test, dan setiap `run.json` training mencatat `"test_set_used": false`.
Seluruh pemilihan memakai validation (S03, S06). Evaluasi test pertama adalah
milik Orang 5 (F-02).

## Model final (F-01)

| Item | Nilai |
|---|---|
| Checkpoint | `ft_lr1e-5_001`, epoch 4 (dari 7 epoch, early stopping) |
| Base | `openai/whisper-small` revision `973afd24965f72e36ca33b3055d56a652f456b4d` |
| Config inference | `phase2/configs/experiments/finetuned_ft_lr1e-5_001_beam5.json` |
| Bobot | `phase2/models/ft_lr1e-5_001/ct2/` (CTranslate2 float16, diunduh dari Google Drive; lihat README bagian Orang 4 langkah 2) |
| SHA-256 `model.bin` | `768791f7a4498a7431a5debc7a50a61a8ef43aa68573c74c081295095b43373c` |
| Decoding | CPU INT8, beam 5, `language=id`, temperature 0, tanpa prompt/VAD |
| Preprocessing | `preprocess_v1` (sama dengan saat training) |
| WER/CER validation | 26,18% / 9,22% (metrik seleksi Orang 4, lihat di bawah) |

Model dimuat lewat adapter Orang 3 tanpa perubahan kode:

```python
from phase2.src.model import WhisperASR, load_config

asr = WhisperASR(load_config("phase2/configs/experiments/finetuned_ft_lr1e-5_001_beam5.json"))
```

## Menjalankan model final di test

`phase2/src/infer.py` sengaja hanya mendecode validation. Untuk test, Orang 5 perlu
runner evaluasi sendiri (atau menambah opsi split ke `infer.py`) dengan pola yang
sama:

```python
from phase2.scripts.dataset import PHASE2, read_csv

rows = read_csv(PHASE2 / "data/manifests/test.csv")
predictions = [asr.transcribe_sample(row) for row in rows]
```

`transcribe_sample` membaca rentang anotasi dan preprocessing Orang 2 dari
`preprocess_config` di config, jadi pipeline-nya sama persis dengan validation.
Perkiraan waktu di laptop CPU: ±10–30 menit untuk 230 ujaran.

Sebaiknya test juga dijalankan untuk **baseline pretrained**
(`phase2/configs/model.json`) dengan runner yang sama, agar laporan memiliki
selisih pretrained vs fine-tuned di test.

## Prediksi validation yang sudah ada

Semua prediksi ada di `phase2/results/experiments/<run_id>/predictions.csv`. File
ini lokal dan diabaikan Git, jadi di mesin lain harus dibuat ulang dengan perintah
di README bagian Orang 4 langkah 3. `run.json` dan `config.json` setiap run ada di
Git.

| run_id | Isi |
|---|---|
| `pretrained_beam1_001` | Whisper small pretrained, beam 1 (sama dengan baseline Orang 3) |
| `pretrained_beam5_001` | pretrained, beam 5 |
| `pretrained_rms_trim_001` | pretrained, beam 1, preprocessing RMS + trim |
| `finetuned_lr1e-5_001` | fine-tuned lr 1e-5, beam 1 |
| `finetuned_lr5e-6_001` | fine-tuned lr 5e-6, beam 1 |
| `finetuned_lr3e-5_001` | fine-tuned lr 3e-5, beam 1 |
| `finetuned_lr1e-5_freeze_encoder_001` | fine-tuned lr 1e-5 encoder beku, beam 1 |
| `finetuned_lr1e-5_beam5_001` | fine-tuned lr 1e-5, beam 5 |

Format kolom sama dengan baseline Orang 3:
`run_id, sample_id, split, reference, prediction`. Referensi dan prediksi mentah
(belum dinormalisasi) disimpan apa adanya.

Prediksi per epoch saat training (HF greedy) ada di
`phase2/results/training/<run_id>/epochs/epoch_XX/predictions.csv`, juga lokal saja.
Folder ini hanya ada di mesin yang menjalankan `kaggle_job collect`.

## Metrik seleksi Orang 4 (bukan scorer resmi)

`phase2/src/selection_metrics.py`:

- referensi dan prediksi dinormalisasi dengan `normalize_text(keep_hyphen=True)`
  dari Orang 2 (huruf kecil, tanpa tanda baca dan tag `[...]`, tanda hubung di
  dalam kata dipertahankan);
- WER/CER level korpus: total edit dibagi total kata/karakter referensi (jiwer 4).
  CER menghitung spasi;
- referensi yang kosong setelah normalisasi ditolak (tidak ada di validation).

Orang 5 bebas menetapkan kebijakan resmi. Jika berbeda, angka di laporan Orang 4
perlu ditulis sebagai "metrik seleksi" dan tabel resmi memakai scorer Orang 5.
Hal yang perlu diputuskan:

1. **Ujaran `[nama]`** (4 di validation). Tag hilang saat normalisasi, sehingga
   nama yang diucapkan dan ditranskripsi model dihitung sebagai insertion
   (10 insertion pada model final). Alternatif: abaikan ujaran ini atau
   masking nama di prediksi. Nama yang muncul di prediksi **tidak boleh
   dipublikasikan**.
2. **Tanda hubung** (`keep_hyphen`). Kata ulang seperti `apo-apo` dihitung satu
   kata. Dengan `keep_hyphen=False` jumlah kata berubah.
3. **Angka dan ejaan varian** tidak dinormalisasi (mis. angka vs kata bilangan).

## Ketidakpastian

`phase2/results/experiments/comparison.json` berisi paired bootstrap (10.000 sampel
ulang ujaran) setiap run terhadap checkpoint terpilih dengan beam 1
(`finetuned_lr1e-5_001`), beserta WER per speaker. Ringkasnya:

- fine-tuned vs pretrained: selisih ±20 poin WER, interval jauh dari nol;
- antar-learning rate: interval melewati nol, jadi **tidak berbeda nyata**;
- encoder beku: 1,5 poin lebih buruk, interval [+0,1; +2,9];
- beam 5 (model final): 0,8 poin lebih baik, interval [−1,6; −0,04].

Bootstrap mengambil sampel ulang ujaran, bukan speaker (validation hanya dua
speaker), sehingga ketidakpastian sebenarnya lebih besar. WER validation model
final juga sedikit optimistis karena validation dipakai untuk memilih.

## Bahan analisis error (F-04)

- Model pretrained menghasilkan prediksi kosong pada 7 ujaran validation; model
  fine-tuned tidak pernah.
- Fine-tuning (keduanya beam 1) memangkas substitusi (919 → 495) dan deletion
  (182 → 64), tetapi insertion naik (69 → 108). Model final (beam 5): S 471, D 63, I 113. Insertion tersebar 1–4 kata per ujaran, tanpa
  pengulangan tak berujung.
- Speaker S03 lebih sulit dari S06 untuk semua model (model final: 30,7% vs 21,5%).

Angka di atas berasal dari validation. Analisis error resmi sebaiknya memakai
prediksi test.
