# Fase 2 - ASR Bahasa Palembang

Sistem pengenal ucapan bahasa Palembang dengan Whisper `small` yang di-fine-tune
pada korpus Fase 1 (`Korpus_Bahasa_Palembang/`). Semua perintah dijalankan dari
root repo.

## 1. Setup

Python 3.12.

```sh
python -m venv phase2/.venv
source phase2/.venv/bin/activate          # Windows: phase2\.venv\Scripts\Activate.ps1
python -m pip install -r phase2/requirements-train.txt
```

Tanpa GPU NVIDIA, pasang PyTorch CPU dulu:
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
Untuk inference dan evaluasi saja cukup
`python -m pip install -r phase2/requirements-model.txt jiwer==4.0.0`.

## 2. Data

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

## 3. Preprocessing

```sh
python -m phase2.src.preprocess --config phase2/configs/preprocess_v1.json
```

Audio dipotong sesuai batas anotasi dan diperiksa formatnya (WAV 16 kHz mono 16-bit).
Teks dinormalisasi untuk training dan scoring (`src/text_normalization.py`): huruf
kecil, tanda baca dan tag `[...]` dibuang, tanda hubung di dalam kata dipertahankan,
ejaan Palembang tidak diubah. `configs/preprocess_rms_trim_v1.json` adalah varian
dengan normalisasi RMS dan trimming untuk eksperimen.

## 4. Model dan inference

Whisper multilingual `small` (encoder-decoder Transformer) melalui
[faster-whisper 1.2.1](https://github.com/SYSTRAN/faster-whisper/tree/v1.2.1).
Model pretrained: [`Systran/faster-whisper-small`](https://huggingface.co/Systran/faster-whisper-small)
revision `536b0662742c02347bc0e980a01041f333bce120`. Decoding: `language=id`,
`task=transcribe`, temperature 0, tanpa prompt dan VAD, CPU INT8.

```python
from phase2.src.model import WhisperASR, load_config

asr = WhisperASR(load_config("phase2/configs/experiments/finetuned_ft_lr1e-5_001_beam5.json"))
text = asr.transcribe("input.wav")
```

Prediksi validation untuk satu config:

```sh
python -m phase2.src.infer --config <config> --run-id <run_id> --output-root phase2/results/experiments
```

## 5. Training / fine-tuning

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
```

Hasilnya di `results/summary.csv`, `results/training_summary.csv`, dan
`results/experiments/README.md`.

## 6. Evaluasi

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

## 7. Model

Model final: `ft_lr1e-5_001` (epoch 4), inference dengan
`configs/experiments/finetuned_ft_lr1e-5_001_beam5.json`. Bobot (±466 MB per run)
tidak masuk Git; unduh dari
[Google Drive](https://drive.google.com/drive/folders/10jwfv_vFGCfXh0vtOfake7wC8Flz0Sqq)
ke:

```
phase2/models/ft_lr1e-5_001/ct2/
    config.json  model.bin  preprocessor_config.json  tokenizer.json  vocabulary.json
```

`WhisperASR` mencocokkan hash SHA-256 `model.bin` dengan `model_sha256` di config.
`predictions.csv` juga tidak masuk Git karena model dapat menuliskan nama penutur.

Referensi: [Whisper](https://github.com/openai/whisper),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[HF Transformers](https://github.com/huggingface/transformers),
[jiwer](https://github.com/jitsi/jiwer).
