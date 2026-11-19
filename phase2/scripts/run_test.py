import argparse
import csv
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import time

from phase2.scripts.dataset import PHASE2, ROOT, digest, read_csv, write_json
from phase2.scripts.validate_manifest import validate
from phase2.src import evaluate, model
from phase2.src.infer import PREDICTION_FIELDS

MANIFESTS = PHASE2 / "data/manifests"


def check_no_leakage(test_rows):
    seen = read_csv(MANIFESTS / "train.csv") + read_csv(MANIFESTS / "validation.csv")
    if {r["speaker_id"] for r in test_rows} & {r["speaker_id"] for r in seen}:
        raise ValueError("Speaker test juga ada di train/validation")
    if {r["pcm_sha256"] for r in test_rows} & {r["pcm_sha256"] for r in seen}:
        raise ValueError("Audio test juga ada di train/validation")


def run(config_path, run_id):
    manifest = MANIFESTS / "test.csv"
    validate()
    rows = read_csv(manifest)
    check_no_leakage(rows)
    config = model.load_config(config_path)
    asr = model.WhisperASR(config)
    out = PHASE2 / "results/test" / run_id
    out.mkdir(parents=True)
    started = time.perf_counter()
    predictions = out / "predictions.csv"
    with predictions.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({"run_id": run_id, "sample_id": row["sample_id"], "split": "test",
                             "reference": row["transcript"], "prediction": asr.transcribe_sample(row)})
    result = evaluate.run(predictions)
    write_json(out / "run.json", {
        "run_id": run_id, "config": Path(config_path).as_posix(), "config_sha256": digest(config_path),
        "manifest_sha256": digest(manifest), "predictions_sha256": digest(predictions),
        "preprocess_config_sha256": asr.preprocess_config["_sha256"],
        "model_artifacts": asr.artifact_hashes,
        "utterances": len(rows), "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("faster-whisper", "ctranslate2", "tokenizers", "numpy", "jiwer")},
        "actual_device": asr.backend.model.device, "actual_compute_type": asr.backend.model.compute_type,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                                   cwd=ROOT, text=True).strip()),
        "elapsed_sec": round(time.perf_counter() - started, 1)})
    return result


def main():
    parser = argparse.ArgumentParser(description="Jalankan model di test set lalu hitung WER/CER")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run(args.config, args.run_id)
    print(f"WER {result['primary']['wer']:.4f}  CER {result['primary']['cer']:.4f}")


if __name__ == "__main__":
    main()
