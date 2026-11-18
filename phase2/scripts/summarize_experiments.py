"""Ringkasan eksperimen di validation (bukan test)."""
import argparse
import json
from pathlib import Path

from phase2.scripts.dataset import PHASE2, digest, read_csv, write_csv
from phase2.src import selection_metrics


def markdown_table(columns, rows):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)

RUN_ROOTS = [PHASE2 / "results/baseline", PHASE2 / "results/experiments"]
SUMMARY_FIELDS = ["run_id", "kind", "model", "model_revision_or_sha256", "training_run", "preprocess_version",
                  "beam_size", "split", "utterances", "wer", "cer", "substitutions", "deletions", "insertions",
                  "ref_words", "ref_chars", "elapsed_sec", "code_commit", "config_sha256",
                  "predictions_sha256", "scoring_policy"]
TRAINING_FIELDS = ["run_id", "learning_rate", "freeze_encoder", "effective_batch_size", "epochs_run",
                   "best_epoch", "best_val_wer_hf_greedy", "best_val_cer_hf_greedy", "best_val_loss",
                   "epoch0_val_wer_hf_greedy", "train_utterances", "trainable_parameters", "gpu",
                   "elapsed_sec", "code_commit", "ct2_model_sha256"]


def inference_rows(min_rows):
    rows = []
    for root in RUN_ROOTS:
        for run_file in sorted(root.glob("*/run.json")):
            run = json.loads(run_file.read_text(encoding="utf-8"))
            predictions = run_file.parent / "predictions.csv"
            if run.get("status") != "complete" or not predictions.exists():
                continue
            preds = read_csv(predictions)
            if len(preds) < min_rows:  # smoke/reload run: bukan perbandingan korpus penuh
                continue
            scores = selection_metrics.corpus_scores([p["reference"] for p in preds],
                                                     [p["prediction"] for p in preds])
            config = run["config"]
            local = "model_path" in config
            rows.append({
                "run_id": run["run_id"], "kind": "finetuned" if local else "pretrained",
                "model": config.get("base_model_id") if local else config["model_id"],
                "model_revision_or_sha256": config["model_sha256"] if local else config["revision"],
                "training_run": config.get("training_run", ""),
                "preprocess_version": run.get("preprocess_version"),
                "beam_size": config["decoding"]["beam_size"], "split": run["split"],
                "utterances": scores["utterances"], "wer": round(scores["wer"], 4),
                "cer": round(scores["cer"], 4),
                **{k: scores[k] for k in ("substitutions", "deletions", "insertions")},
                "ref_words": scores["ref_words"], "ref_chars": scores["ref_chars"],
                "elapsed_sec": round(run["elapsed_sec"], 1), "code_commit": run["code_commit"][:7],
                "config_sha256": run["config_sha256"][:12], "predictions_sha256": digest(predictions)[:12],
                "scoring_policy": selection_metrics.SCORING_POLICY.split(":")[0]})
    return sorted(rows, key=lambda r: (r["kind"] != "pretrained", r["run_id"]))


def training_rows():
    rows = []
    for run_file in sorted((PHASE2 / "results/training").glob("*/run.json")):
        run = json.loads(run_file.read_text(encoding="utf-8"))
        history = run.get("history") or []
        if not history:
            continue
        best = run.get("best") or {}
        config = run["config"]
        rows.append({
            "run_id": run["run_id"], "learning_rate": config["learning_rate"],
            "freeze_encoder": config["freeze_encoder"], "effective_batch_size": run.get("effective_batch_size"),
            "epochs_run": max(h["epoch"] for h in history), "best_epoch": best.get("epoch"),
            "best_val_wer_hf_greedy": round(best["val_wer"], 4) if best.get("epoch") else "",
            "best_val_cer_hf_greedy": round(best["val_cer"], 4) if best.get("epoch") else "",
            "best_val_loss": round(best["val_loss"], 4) if best.get("epoch") else "",
            "epoch0_val_wer_hf_greedy": history[0]["val_wer"],
            "train_utterances": run.get("train_utterances"), "trainable_parameters": run.get("trainable_parameters"),
            "gpu": run.get("gpu"), "elapsed_sec": run.get("elapsed_sec"),
            "code_commit": (run.get("code_commit") or "")[:7], "ct2_model_sha256": (run.get("ct2_model_sha256") or "")[:12]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-rows", type=int, default=235, help="Abaikan run subset (smoke)")
    args = parser.parse_args()
    inference, training = inference_rows(args.min_rows), training_rows()
    write_csv(PHASE2 / "results/summary.csv", inference, SUMMARY_FIELDS)
    write_csv(PHASE2 / "results/training_summary.csv", training, TRAINING_FIELDS)
    columns = ["run_id", "kind", "beam_size", "preprocess_version", "utterances", "wer", "cer",
               "substitutions", "deletions", "insertions", "elapsed_sec"]
    text = ["# Ringkasan eksperimen (validation)", "",
            "Dihasilkan oleh `python -m phase2.scripts.summarize_experiments`. WER/CER level korpus",
            f"dengan aturan `{selection_metrics.SCORING_POLICY}`. Bukan angka test.", "",
            "## Inference CPU (faster-whisper INT8, adapter `model.py`)", "",
            markdown_table(columns, [[r[c] for c in columns] for r in inference]), ""]
    if training:
        tcols = ["run_id", "learning_rate", "freeze_encoder", "epochs_run", "best_epoch",
                 "epoch0_val_wer_hf_greedy", "best_val_wer_hf_greedy", "best_val_loss", "elapsed_sec"]
        text += ["## Fine-tuning (GPU, WER greedy pipeline HF saat training)", "",
                 markdown_table(tcols, [[r[c] for c in tcols] for r in training]), ""]
    (PHASE2 / "results/experiments").mkdir(parents=True, exist_ok=True)
    (PHASE2 / "results/experiments/README.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
