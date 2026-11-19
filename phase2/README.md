# Fase 2 - ASR Bahasa Palembang

Sistem pengenal ucapan bahasa Palembang dengan Whisper `small` yang di-fine-tune
pada korpus Fase 1 (`Korpus_Bahasa_Palembang/`). Semua perintah dijalankan dari
root repo.

## Quickstart

Cara tercepat untuk mencoba model final pada file audio sendiri:

```sh
python -m venv phase2/.venv
phase2\.venv\Scripts\Activate.ps1                 # Linux/macOS: source phase2/.venv/bin/activate
python -m pip install -r phase2/requirements-demo.txt
python -m phase2.scripts.download_model
python -m phase2.demo rekaman.wav
```

Atau dengan UI web:

```sh
python -m phase2.demo --ui
```

lalu buka http://127.0.0.1:7860.

## Pipeline

```
file audio (wav/mp3/m4a/...)
  -> load_audio     src/audio.py       WAV 16 kHz mono 16-bit dibaca langsung, format lain dikonversi ffmpeg
  -> preprocess     src/preprocess.py  sesuai configs/preprocess_v1.json (sama dengan saat training dan evaluasi)
  -> Whisper        src/model.py       checkpoint fine-tuned ft_lr1e-5_001, beam 5, CPU INT8
  -> transkripsi    src/pipeline.py    teks mentah dan teks ternormalisasi (src/text_normalization.py)
```

Semua tahap dirangkai di `src/pipeline.py`:

```python
from phase2.src.pipeline import TranscriptionPipeline

pipeline = TranscriptionPipeline()
result = pipeline.transcribe("rekaman.wav")
print(result.text)
```

`TranscriptionPipeline(config_path)` menerima config model apa pun di `configs/`,
misalnya `phase2/configs/model.json` untuk Whisper small pretrained.

## Struktur repository

```
Korpus_Bahasa_Palembang/        korpus Fase 1 (audio, anotasi, metadata), hanya dibaca
phase2/
  demo.py                       demo CLI dan UI Gradio
  configs/
    split_v1.json               pembagian speaker train/validation/test
    preprocess_v1.json          preprocessing utama
    preprocess_rms_trim_v1.json varian RMS + trim untuk eksperimen
    model.json                  Whisper small pretrained
    experiments/                config inference tiap eksperimen (termasuk model final)
    train/                      config fine-tuning dan job Kaggle
  data/
    manifests/                  manifest train/validation/test dan ujaran yang dikeluarkan
    processed/                  hasil preprocessing per versi
  docs/dataset_preparation.md   audit data, kebijakan pengecualian, strategi split
  kaggle/run_kaggle.py          entry point training di Kaggle
  models/                       checkpoint fine-tuned (tidak masuk Git, lihat bagian Model)
  report/figures/               gambar untuk laporan
  results/                      hasil eksperimen validation, training, dan test
  scripts/
    prepare_manifest.py         membuat manifest dari korpus Fase 1
    audit_phase1.py             audit korpus Fase 1
    validate_manifest.py        cek konsistensi manifest dan split
    download_model.py           unduh checkpoint fine-tuned dari Google Drive
    kaggle_job.py               siapkan dan kumpulkan job training Kaggle
    run_test.py                 evaluasi akhir pada split test
    summarize_experiments.py    ringkasan eksperimen validation
    compare_runs.py             uji signifikansi antar-eksperimen (paired bootstrap)
    plot_training_curves.py     kurva training
  src/
    audio.py                    baca/tulis WAV, konversi ffmpeg, metrik audio
    preprocess.py               QC dan preprocessing audio/teks
    text_normalization.py       normalisasi teks untuk training dan scoring
    model.py                    loader Whisper (faster-whisper)
    pipeline.py                 pipeline audio -> transkripsi
    infer.py                    prediksi validation per config
    train.py                    fine-tuning Whisper
    evaluate.py                 WER/CER
    selection_metrics.py        metrik pemilihan checkpoint
  requirements.txt              numpy
  requirements-model.txt        inference (faster-whisper, ctranslate2)
  requirements-demo.txt         inference + evaluasi + demo + unduh model
  requirements-train.txt        training (torch, transformers, jiwer, matplotlib)
```

## 1. Setup

Python 3.12 (sudah dicoba juga di 3.13). Untuk audio selain WAV 16 kHz mono,
`ffmpeg` harus ada di PATH.

```sh
python -m venv phase2/.venv
source phase2/.venv/bin/activate          # Windows: phase2\.venv\Scripts\Activate.ps1
```

Pilih requirements sesuai kebutuhan:

| Kebutuhan | Perintah |
|---|---|
| Demo, inference, evaluasi | `python -m pip install -r phase2/requirements-demo.txt` |
| Training | `python -m pip install -r phase2/requirements-train.txt` |

Tanpa GPU NVIDIA, pasang PyTorch CPU dulu sebelum requirements training:
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## 2. Model

Model final: `ft_lr1e-5_001` (epoch 4), inference dengan
`configs/experiments/finetuned_ft_lr1e-5_001_beam5.json`. Bobot (±466 MB per run)
tidak masuk Git dan disimpan di
[Google Drive](https://drive.google.com/drive/folders/10jwfv_vFGCfXh0vtOfake7wC8Flz0Sqq).

```sh
python -m phase2.scripts.download_model
```

Script ini hanya mengunduh run yang dipakai config, menaruhnya di
`phase2/models/ft_lr1e-5_001/ct2/`, lalu mencocokkan SHA-256 `model.bin` dengan
`model_sha256` di config. Untuk run eksperimen lain, beri config-nya:

```sh
python -m phase2.scripts.download_model --config phase2/configs/experiments/finetuned_ft_lr3e-5_001.json
```

Kalau ingin mengunduh manual, simpan isi folder `<run>/ct2/` dari Drive ke
`phase2/models/<run>/ct2/` (`config.json`, `model.bin`, `preprocessor_config.json`,
`tokenizer.json`, `vocabulary.json`). Model pretrained (`configs/model.json`)
diunduh otomatis dari Hugging Face saat pertama dipakai.

## 3. Demo

CLI, bisa beberapa file sekaligus:

```sh
python -m phase2.demo rekaman1.wav rekaman2.mp3
python -m phase2.demo rekaman.wav --normalized
python -m phase2.demo rekaman.wav --config phase2/configs/model.json
```

UI web (upload file atau rekam dari mikrofon, bisa memilih model fine-tuned atau
pretrained):

```sh
python -m phase2.demo --ui
python -m phase2.demo --ui --port 8000 --share
```

## 4. Data

```sh
python -m phase2.scripts.prepare_manifest
python -m phase2.scripts.audit_phase1
python -m phase2.scripts.validate_manifest
```

Korpus hanya dibaca. Pembagian speaker disimpan di `configs/split_v1.json`
(speaker-independent):

| Split | Speaker | Ujaran |
|---|---|---|
| train | S01, S02, S04, S07, S08, S09, S11 | 762 |
| validation | S03, S06 | 235 |
| test | S05, S10 | 230 |

- `data/manifests/{train,validation,test}.csv`: satu baris per ujaran (path audio,
  batas waktu, transkripsi, speaker).
- `data/manifests/excluded.csv`: ujaran yang tidak dipakai beserta alasannya.
- `data/speakers.csv`, `data/dataset_stats.json`: komposisi dan durasi per split.
- [docs/dataset_preparation.md](docs/dataset_preparation.md): audit, kebijakan
  pengecualian, dan strategi split.

## 5. Preprocessing

```sh
python -m phase2.src.preprocess --config phase2/configs/preprocess_v1.json
```

Audio dipotong sesuai batas anotasi dan diperiksa formatnya (WAV 16 kHz mono 16-bit).
Teks dinormalisasi untuk training dan scoring (`src/text_normalization.py`): huruf
kecil, tanda baca dan tag `[...]` dibuang, tanda hubung di dalam kata dipertahankan,
ejaan Palembang tidak diubah. `configs/preprocess_rms_trim_v1.json` adalah varian
dengan normalisasi RMS dan trimming untuk eksperimen.

## 6. Inference

Whisper multilingual `small` (encoder-decoder Transformer) melalui
[faster-whisper 1.2.1](https://github.com/SYSTRAN/faster-whisper/tree/v1.2.1).
Model pretrained: [`Systran/faster-whisper-small`](https://huggingface.co/Systran/faster-whisper-small)
revision `536b0662742c02347bc0e980a01041f333bce120`. Decoding: `language=id`,
`task=transcribe`, temperature 0, tanpa prompt dan VAD, CPU INT8.

Satu file audio: lihat bagian Demo atau Pipeline. Prediksi seluruh split validation
untuk satu config:

```sh
python -m phase2.src.infer --config <config> --run-id <run_id> --output-root phase2/results/experiments
```

## 7. Training / fine-tuning

Fine-tuning dengan HF Transformers + PyTorch pada data train; checkpoint terbaik
dipilih dengan WER validation lalu dikonversi ke CTranslate2.

```sh
python -m phase2.src.train --config phase2/configs/train/whisper_small_ft_lr1e-5.json --run-id <run_id>
```

Hasil yang dilaporkan dilatih di Kaggle (GPU T4, ±15 menit per run). Perlu Kaggle
CLI, token API di `~/.kaggle/`, dan semua perubahan sudah di-commit:

```sh
python -m phase2.scripts.kaggle_job prepare --username <USERNAME_KAGGLE>
cd phase2/.kaggle_build/dataset && kaggle datasets create -p .
cd ../kernel && kaggle kernels push -p .
cd ../../..
kaggle kernels output <USERNAME_KAGGLE>/palembang-asr-whisper-finetune -p <FOLDER_UNDUHAN>
python -m phase2.scripts.kaggle_job collect --download <FOLDER_UNDUHAN>
```

Eksperimen (config di `configs/train/` dan `configs/experiments/`):

| Variasi | Nilai |
|---|---|
| Pretrained vs fine-tuned | Whisper small tanpa dan dengan fine-tuning |
| Learning rate | 5e-6, 1e-5, 3e-5 |
| Encoder | dilatih vs dibekukan |
| Beam size | 1 vs 5 |
| Preprocessing | tanpa transformasi vs RMS + trim |

Ringkasan validation dan kurva training:

```sh
python -m phase2.scripts.summarize_experiments
python -m phase2.scripts.plot_training_curves
python -m phase2.scripts.compare_runs --reference finetuned_lr1e-5_001
```

Hasilnya di `results/summary.csv`, `results/training_summary.csv`,
`results/experiments/README.md`, dan `results/experiments/comparison.json`
(selisih WER tiap run terhadap acuan beserta interval kepercayaan 95%).

## 8. Evaluasi

WER/CER level korpus (total edit dibagi total kata/karakter referensi) dihitung
`src/evaluate.py`. Ujaran yang referensinya memuat `[nama]` tidak masuk angka utama
karena nama penutur disensor di transkripsi.

```sh
python -m phase2.scripts.run_test --config phase2/configs/experiments/finetuned_ft_lr1e-5_001_beam5.json --run-id finetuned_final_test_001
python -m phase2.scripts.run_test --config phase2/configs/model.json --run-id pretrained_test_001
```

`run_test` memastikan speaker dan audio test tidak ada di train/validation, lalu
menulis `results/test/<run_id>/predictions.csv`, `run.json`, dan
`evaluation/metrics.json` (WER/CER total dan per speaker).

Hasil pada split test (S05 dan S10, 228 ujaran):

| Model | WER | CER |
|---|---|---|
| Whisper small pretrained | 39,03% | 14,68% |
| Whisper small fine-tuned (`ft_lr1e-5_001`, beam 5) | 23,43% | 7,20% |

Model fine-tuned per speaker: S05 WER 20,05% / CER 5,29%, S10 WER 26,37% / CER 8,63%.
Detail per run ada di `results/test/<run_id>/run.json` dan `evaluation/`.

`predictions.csv` tidak masuk Git karena model dapat menuliskan nama penutur.

Referensi: [Whisper](https://github.com/openai/whisper),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[HF Transformers](https://github.com/huggingface/transformers),
[jiwer](https://github.com/jitsi/jiwer),
[Gradio](https://github.com/gradio-app/gradio).
