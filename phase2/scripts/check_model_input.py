"""Owner 3: check the existing validation waveform handoff without loading a model."""
import hashlib
import json

import numpy as np

from phase2.scripts.dataset import PHASE2, digest, read_csv
from phase2.src import audio
from phase2.src.preprocess import load_config, load_sample, output_dir, prepare_text


def check(config=None):
    config = config or load_config()
    source = PHASE2 / "data/manifests/validation.csv"
    processed = output_dir(config) / "validation.csv"
    rows, prepared = read_csv(source), read_csv(processed)
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Validation manifest must be nonempty with unique sample IDs")
    by_id = {r["sample_id"]: r for r in prepared}
    if len(by_id) != len(prepared) or set(by_id) != {r["sample_id"] for r in rows}:
        raise ValueError("Source and processed validation sample IDs differ")
    lengths = []
    for row in rows:
        sample_id = row["sample_id"]
        target = by_id[sample_id]
        if row["split"] != "validation" or target["split"] != "validation":
            raise ValueError(f"{sample_id}: expected validation split")
        if target["transcript"] != row["transcript"] or target["text"] != prepare_text(row["transcript"], config):
            raise ValueError(f"{sample_id}: raw/normalized text mismatch")
        signal = load_sample(row, config)
        if (signal.dtype != np.float32 or signal.ndim != 1 or not signal.size
                or not np.isfinite(signal).all() or np.max(np.abs(signal)) > 1):
            raise ValueError(f"{sample_id}: expected finite nonempty mono float32 in [-1, 1]")
        processed_signal = load_sample(target, config)
        if not np.array_equal(signal, processed_signal):
            raise ValueError(f"{sample_id}: source/processed waveform mismatch")
        if hashlib.sha256(audio.to_pcm(signal).tobytes()).hexdigest() != target["pcm_sha256"]:
            raise ValueError(f"{sample_id}: processed PCM hash mismatch")
        if len(signal) != round(float(target["duration_sec"]) * audio.SAMPLE_RATE):
            raise ValueError(f"{sample_id}: processed duration mismatch")
        lengths.append(len(signal))
    return {
        "status": "input_contract_passed",
        "scope": "validation only; no model/tokenizer loaded or predictions generated",
        "source_manifest_sha256": digest(source),
        "processed_manifest_sha256": digest(processed),
        "preprocess_config_sha256": config["_sha256"],
        "preprocess_version": config["version"],
        "utterances": len(rows), "sample_rate": audio.SAMPLE_RATE,
        "dtype": "float32", "channels": 1,
        "min_samples": min(lengths), "max_samples": max(lengths),
        "duration_sec": sum(lengths) / audio.SAMPLE_RATE,
    }


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
