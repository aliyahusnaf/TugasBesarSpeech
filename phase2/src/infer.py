# Buat prediksi validation untuk satu config model.
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import re
import subprocess
import time
import traceback

from phase2.scripts.dataset import PHASE2, ROOT, digest, read_csv, source_snapshot, write_json
from phase2.src import audio, model

PREDICTION_FIELDS = ["run_id", "sample_id", "split", "reference", "prediction"]


def provenance():
    # Pastikan file sumber sama dengan snapshot, beda line ending CRLF/LF diperbolehkan.
    saved = json.loads((PHASE2 / "data/source_snapshot.json").read_text(encoding="utf-8"))
    current = source_snapshot()
    if set(saved["files"]) != set(current["files"]):
        raise ValueError("Source inventory changed; request a new snapshot")
    newline_only = []
    for path, expected in saved["files"].items():
        if current["files"][path] == expected:
            continue
        payload = (ROOT / path).read_bytes()
        if Path(path).suffix not in {".csv", ".json"}:
            raise ValueError(f"Source changed: {path}")
        lf = payload.replace(b"\r\n", b"\n")
        if expected not in {hashlib.sha256(lf).hexdigest(),
                            hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}:
            raise ValueError(f"Source content changed: {path}")
        newline_only.append(path)
    return {"saved_commit": saved["commit"], "saved_fingerprint": saved["fingerprint"],
            "local_fingerprint": current["fingerprint"], "line_endings_only": newline_only}


def run(run_id, config_path=model.DEFAULT_CONFIG, limit=None, output_root=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("run_id must contain only letters, digits, underscores or hyphens")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    config = model.load_config(config_path)
    out = Path(output_root or PHASE2 / "results/experiments") / run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metadata = {"run_id": run_id, "status": "running", "config": config,
                "config_sha256": digest(config_path), "split": "validation", "limit": limit,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(),
                "machine": platform.machine(), "seed": None,
                "packages": {d.metadata["Name"]: d.version for d in distributions()},
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()),
                "completed_utterances": 0}
    write_json(out / "config.json", config)
    write_json(out / "run.json", metadata)
    try:
        metadata["source_snapshot"] = provenance()
        split_path = PHASE2 / "configs/split_v1.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        manifest = PHASE2 / "data/manifests/validation.csv"
        rows = read_csv(manifest)
        if (not rows or len({r["sample_id"] for r in rows}) != len(rows)
                or any(r["split"] != "validation" or r["speaker_id"] not in split["speakers"]["validation"] for r in rows)):
            raise ValueError("Invalid validation manifest IDs, split or speakers")
        metadata.update(manifest_sha256=digest(manifest), split_version=split["version"],
                        split_config_sha256=digest(split_path), total_utterances=len(rows))
        if limit is not None:
            rows = rows[:limit]
        asr = model.WhisperASR(config)
        metadata.update(model_artifacts=asr.artifact_hashes,
                        preprocess_config_sha256=asr.preprocess_config["_sha256"],
                        preprocess_version=asr.preprocess_config["version"],
                        actual_device=asr.backend.model.device,
                        actual_compute_type=asr.backend.model.compute_type)
        with (out / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            for row in rows:
                pcm = audio.read_segment(ROOT / row["audio_path"], row["start_sec"], row["end_sec"])
                if hashlib.sha256(pcm.tobytes()).hexdigest() != row["pcm_sha256"]:
                    raise ValueError(f"Source PCM changed: {row['sample_id']}")
                prediction = asr.transcribe_sample(row)
                writer.writerow({"run_id": run_id, "sample_id": row["sample_id"],
                                 "split": row["split"], "reference": row["transcript"],
                                 "prediction": prediction})
                stream.flush()
                metadata["completed_utterances"] += 1
                print(f"{run_id}: {metadata['completed_utterances']}/{len(rows)} {row['sample_id']}", flush=True)
        metadata["status"] = "complete"
    except BaseException:
        metadata["status"] = "failed"
        metadata["error"] = traceback.format_exc()
        raise
    finally:
        metadata["elapsed_sec"] = time.perf_counter() - started
        write_json(out / "run.json", metadata)
    return out


def main():
    parser = argparse.ArgumentParser(description="Buat prediksi validation untuk satu config model.")
    parser.add_argument("--config", type=Path, default=model.DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, help="Hanya N ujaran pertama validation")
    parser.add_argument("--output-root", type=Path, help="Default phase2/results/experiments")
    args = parser.parse_args()
    run(args.run_id, args.config, args.limit, args.output_root)


if __name__ == "__main__":
    main()
