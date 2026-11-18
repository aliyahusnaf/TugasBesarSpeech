# Ringkasan eksperimen (validation)

Dihasilkan oleh `python -m phase2.scripts.summarize_experiments`. WER/CER level korpus
dengan aturan `phase2.src.selection_metrics/v1: normalize_text(keep_hyphen=True); corpus-level jiwer 4 word/char edits`. Bukan angka test; scorer resmi milik Orang 5.

## Inference CPU (faster-whisper INT8, adapter Orang 3)

| run_id | kind | beam_size | preprocess_version | utterances | wer | cer | substitutions | deletions | insertions | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrained_beam1_001 | pretrained | 1 | preprocess_v1 | 235 | 0.4735 | 0.1749 | 919 | 182 | 69 | 1835.4 |
| pretrained_beam5_001 | pretrained | 5 | preprocess_v1 | 235 | 0.4415 | 0.1542 | 848 | 172 | 71 | 1721.1 |
| pretrained_rms_trim_001 | pretrained | 1 | preprocess_rms_trim_v1 | 235 | 0.465 | 0.169 | 911 | 166 | 72 | 883.0 |
| finetuned_lr1e-5_001 | finetuned | 1 | preprocess_v1 | 235 | 0.2699 | 0.0944 | 495 | 64 | 108 | 473.3 |
| finetuned_lr1e-5_beam5_001 | finetuned | 5 | preprocess_v1 | 235 | 0.2618 | 0.0922 | 471 | 63 | 113 | 1749.0 |
| finetuned_lr1e-5_freeze_encoder_001 | finetuned | 1 | preprocess_v1 | 235 | 0.2853 | 0.1023 | 523 | 70 | 112 | 1608.3 |
| finetuned_lr3e-5_001 | finetuned | 1 | preprocess_v1 | 235 | 0.2813 | 0.1009 | 530 | 66 | 99 | 1482.9 |
| finetuned_lr5e-6_001 | finetuned | 1 | preprocess_v1 | 235 | 0.2752 | 0.0976 | 508 | 73 | 99 | 1009.0 |

## Fine-tuning (GPU, WER greedy pipeline HF saat training)

| run_id | learning_rate | freeze_encoder | epochs_run | best_epoch | epoch0_val_wer_hf_greedy | best_val_wer_hf_greedy | best_val_loss | elapsed_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ft_lr1e-5_001 | 1e-05 | False | 7 | 4 | 0.481991 | 0.2687 | 0.6142 | 913.4 |
| ft_lr1e-5_freeze_encoder_001 | 1e-05 | True | 7 | 4 | 0.481991 | 0.2861 | 0.6206 | 530.34 |
| ft_lr3e-5_001 | 3e-05 | False | 5 | 2 | 0.481991 | 0.2744 | 0.6267 | 648.11 |
| ft_lr5e-6_001 | 5e-06 | False | 7 | 4 | 0.481991 | 0.2732 | 0.6026 | 890.06 |
