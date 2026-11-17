import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from phase2.scripts.dataset import PHASE2, ROOT, digest, read_csv, write_csv, write_json
from phase2.src import audio
from phase2.src.text_normalization import normalize_text, text_flags

DEFAULT_CONFIG = PHASE2 / "configs/preprocess_v1.json"
SPLITS = ["train", "validation", "test"]
MANIFEST_FIELDS = ["sample_id", "split", "speaker_id", "audio_path", "source_audio_path",
                   "start_sec", "end_sec", "duration_sec", "transcript", "text", "text_flags",
                   "qc_flags", "train_eligible", "exclusion_reason", "pcm_sha256",
                   "preprocess_version"]
QC_FIELDS = ["sample_id", "split", "speaker_id", "duration_sec", "peak_dbfs", "rms_dbfs",
             "dc_offset", "clipped_samples", "longest_clip_run", "active_ratio",
             "leading_silence_sec", "trailing_silence_sec", "dynamic_range_db",
             "chars_per_sec", "qc_flags"]
SOURCE_FIELDS = ["audio_path", "sample_rate", "channels", "bit_depth", "compression", "frames",
                 "standard_format", "action", "used_path"]


def load_config(path=DEFAULT_CONFIG):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_path"] = path.resolve().relative_to(ROOT).as_posix()
    config["_sha256"] = digest(path)
    return config


def output_dir(config):
    return ROOT / config["output_dir"] / config["version"]


def prepare_text(transcript, config):
    return normalize_text(transcript, keep_hyphen=config["text"]["keep_hyphen"])


def process_pcm(pcm, config):
    return audio.apply_variant(pcm, config["audio"], config["qc"])


def load_sample(row, config=None):
    """Sinyal float32 16 kHz untuk satu baris manifest sumber atau hasil preprocessing."""
    config = config or load_config()
    if row.get("preprocess_version", config["version"]) != config["version"]:
        raise ValueError(f"Versi preprocessing baris {row['preprocess_version']} != {config['version']}")
    path = row.get("source_audio_path") or row["audio_path"]
    pcm = audio.read_segment(ROOT / path, row["start_sec"], row["end_sec"])
    return audio.to_float(process_pcm(pcm, config))


def prepare_input(path, config=None):
    """Jalur yang sama untuk audio di luar manifest, misalnya input demo."""
    config = config or load_config()
    return audio.to_float(process_pcm(audio.load_audio_file(path), config))


def qc_flags(metrics, text, qc):
    flags = []
    if metrics["active_ratio"] == 0:
        flags.append("no_active_audio")
    elif metrics["active_ratio"] < qc["min_active_ratio"]:
        flags.append("mostly_silent")
    if metrics["longest_clip_run"] >= qc["min_clip_run"]:
        flags.append("clipping")
    if metrics["rms_dbfs"] < qc["low_level_rms_dbfs"]:
        flags.append("low_level")
    if abs(metrics["dc_offset"]) > qc["max_abs_dc_offset"]:
        flags.append("dc_offset")
    if metrics["leading_silence_sec"] > qc["max_edge_silence_sec"]:
        flags.append("long_leading_silence")
    if metrics["trailing_silence_sec"] > qc["max_edge_silence_sec"]:
        flags.append("long_trailing_silence")
    if metrics["duration_sec"] < qc["min_duration_sec"]:
        flags.append("very_short")
    if metrics["duration_sec"] > qc["max_duration_sec"]:
        flags.append("too_long")
    if not text:
        flags.append("empty_text")
    elif metrics["chars_per_sec"] > qc["max_chars_per_sec"]:
        flags.append("text_rate_high")
    elif metrics["chars_per_sec"] < qc["min_chars_per_sec"] and metrics["speech_span_sec"] >= 1:
        flags.append("text_rate_low")
    return flags


def training_decision(text_flag_list, qc_flag_list, policy):
    reasons = [flag for flag in ("empty_text", "no_active_audio") if flag in qc_flag_list]
    if policy["masked_name"] == "exclude_from_training" and "masked_name" in text_flag_list:
        reasons.append("masked_name")
    return not reasons, ";".join(reasons)


def check_sources(rows, config):
    result, used = [], {}
    for path in sorted({r["audio_path"] for r in rows}):
        info = audio.wav_format(ROOT / path)
        standard = audio.is_standard_format(info)
        target = path
        if not standard:
            converted = output_dir(config) / "converted" / path
            audio.convert_to_standard(ROOT / path, converted)
            target = converted.relative_to(ROOT).as_posix()
        used[path] = target
        result.append({"audio_path": path, **info, "standard_format": standard,
                       "action": "none" if standard else "converted_ffmpeg", "used_path": target})
    return result, used


def fmt(value, digits):
    return f"{value:.{digits}f}" if value != float("-inf") else "-inf"


def build(config, export_audio=False):
    sources = {split: read_csv(PHASE2 / f"data/manifests/{split}.csv") for split in SPLITS}
    rows = []
    for split, part in sources.items():
        if any(r["split"] != split for r in part):
            raise ValueError(f"Label split salah di {split}.csv")
        rows += part
    source_check, used_paths = check_sources(rows, config)
    out = output_dir(config)
    manifests, qc_rows, changes = {s: [] for s in SPLITS}, [], []
    for row in rows:
        source_pcm = audio.read_segment(ROOT / used_paths[row["audio_path"]], row["start_sec"], row["end_sec"])
        converted = used_paths[row["audio_path"]] != row["audio_path"]
        if not converted and hashlib.sha256(source_pcm.tobytes()).hexdigest() != row["pcm_sha256"]:
            raise ValueError(f"PCM sumber berubah sejak manifest dibuat: {row['sample_id']}")
        text = prepare_text(row["transcript"], config)
        metrics = audio.audio_metrics(source_pcm, config["qc"])
        span = max(metrics["duration_sec"] - metrics["leading_silence_sec"] - metrics["trailing_silence_sec"], 0.01)
        metrics["speech_span_sec"] = span
        metrics["chars_per_sec"] = len(text.replace(" ", "")) / span
        tflags = text_flags(row["transcript"], row["events"])
        qflags = qc_flags(metrics, text, config["qc"])
        eligible, reason = training_decision(tflags, qflags, config["policy"])
        processed = process_pcm(source_pcm, config)
        target = out / "audio" / row["split"] / f"{row['sample_id']}.wav"
        if export_audio:
            audio.write_wav(target, processed)
        manifests[row["split"]].append({
            "sample_id": row["sample_id"], "split": row["split"], "speaker_id": row["speaker_id"],
            "audio_path": target.relative_to(ROOT).as_posix(), "source_audio_path": row["audio_path"],
            "start_sec": row["start_sec"], "end_sec": row["end_sec"],
            "duration_sec": fmt(len(processed) / audio.SAMPLE_RATE, 6),
            "transcript": row["transcript"], "text": text, "text_flags": ";".join(tflags),
            "qc_flags": ";".join(qflags), "train_eligible": eligible, "exclusion_reason": reason,
            "pcm_sha256": hashlib.sha256(processed.tobytes()).hexdigest(),
            "preprocess_version": config["version"]})
        qc_rows.append({"sample_id": row["sample_id"], "split": row["split"],
                        "speaker_id": row["speaker_id"], "qc_flags": ";".join(qflags),
                        **{k: fmt(metrics[k], 6 if k.endswith(("_sec", "ratio", "offset")) else 2)
                           for k in ["duration_sec", "peak_dbfs", "rms_dbfs", "dc_offset", "active_ratio",
                                     "leading_silence_sec", "trailing_silence_sec", "dynamic_range_db",
                                     "chars_per_sec"]},
                        "clipped_samples": metrics["clipped_samples"],
                        "longest_clip_run": metrics["longest_clip_run"]})
        if text != row["transcript"]:
            changes.append({"sample_id": row["sample_id"], "transcript": row["transcript"], "text": text})
    return manifests, qc_rows, source_check, changes


def process(config, export_audio=False):
    manifests, qc_rows, source_check, changes = build(config, export_audio)
    out = output_dir(config)
    for split, part in manifests.items():
        write_csv(out / f"{split}.csv", part, MANIFEST_FIELDS)
    write_csv(out / "audio_qc.csv", qc_rows, QC_FIELDS)
    write_csv(out / "source_audio_check.csv", source_check, SOURCE_FIELDS)
    write_csv(out / "text_changes.csv", changes, ["sample_id", "transcript", "text"])
    summary = summarize(config, manifests, source_check, changes)
    write_json(out / "summary.json", summary)
    return summary


def summarize(config, manifests, source_check, changes):
    words = {s: Counter(w for r in part for w in r["text"].split()) for s, part in manifests.items()}
    chars = {s: sorted({c for r in part for c in r["text"]}) for s, part in manifests.items()}
    splits = {}
    for split, part in manifests.items():
        splits[split] = {
            "utterances": len(part),
            "train_eligible": sum(r["train_eligible"] for r in part),
            "duration_sec": round(sum(float(r["duration_sec"]) for r in part), 3),
            "qc_flags": dict(sorted(Counter(f for r in part for f in r["qc_flags"].split(";") if f).items())),
            "text_flags": dict(sorted(Counter(f for r in part for f in r["text_flags"].split(";") if f).items())),
            "exclusion_reasons": dict(sorted(Counter(f for r in part for f in r["exclusion_reason"].split(";") if f).items())),
            "words": sum(words[split].values()),
            "vocabulary": len(words[split]),
            "charset": "".join(chars[split]),
        }
        if split != "train":
            unseen = [w for w in words[split] if w not in words["train"]]
            splits[split]["oov_word_types"] = len(unseen)
            splits[split]["oov_token_rate"] = round(sum(words[split][w] for w in unseen) / max(1, splits[split]["words"]), 4)
            splits[split]["chars_not_in_train"] = "".join(sorted(set(chars[split]) - set(chars["train"])))
    return {
        "version": config["version"], "config": config["_path"], "config_sha256": config["_sha256"],
        "source_manifests": {s: digest(PHASE2 / f"data/manifests/{s}.csv") for s in SPLITS},
        "source_files": len(source_check),
        "source_files_converted": sum(not r["standard_format"] for r in source_check),
        "text_changed_utterances": len(changes),
        "splits": splits,
    }


def main():
    parser = argparse.ArgumentParser(description="QC dan preprocessing audio/teks per ujaran")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--export-audio", action="store_true", help="tulis WAV per ujaran")
    args = parser.parse_args()
    print(json.dumps(process(load_config(args.config), args.export_audio), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
