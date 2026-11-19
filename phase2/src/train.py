# Fine-tuning Whisper (HF Transformers + PyTorch) untuk Bahasa Palembang, test set tidak dibaca.
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import math
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import time
import traceback

import numpy as np

from phase2.scripts.dataset import PHASE2, ROOT, digest, read_csv, write_csv, write_json
from phase2.src import audio, infer, preprocess, selection_metrics

DEFAULT_OUTPUT_ROOT = PHASE2 / "results/training"
HISTORY_FIELDS = ["epoch", "optimizer_steps", "train_loss", "val_loss", "val_wer", "val_cer",
                  "learning_rate", "epoch_train_sec", "epoch_eval_sec", "is_best"]
PREDICTION_FIELDS = infer.PREDICTION_FIELDS
REQUIRED_KEYS = {"technique", "base_model_id", "base_revision", "preprocess_config", "language",
                 "task", "seed", "learning_rate", "num_epochs", "per_device_batch_size",
                 "gradient_accumulation_steps", "warmup_ratio", "weight_decay", "max_grad_norm",
                 "freeze_encoder", "fp16", "early_stopping_patience", "eval_batch_size",
                 "generation_max_new_tokens", "ct2_quantization"}


def load_train_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"Config training kurang kunci: {sorted(missing)}")
    if config["technique"] != "whisper" or not re.fullmatch(r"[0-9a-f]{40}", config["base_revision"]):
        raise ValueError("Fine-tuning Whisper memerlukan base_revision berupa commit SHA")
    if config["task"] != "transcribe" or config["learning_rate"] <= 0 or config["num_epochs"] < 1:
        raise ValueError("Config training tidak valid")
    return config


def load_rows(split, preprocess_config, limit=None):
    # Baris manifest preprocessing, split train hanya `train_eligible`.
    if split == "test":
        raise ValueError("Training/seleksi tidak boleh membaca test set")
    path = ROOT / preprocess_config["output_dir"] / preprocess_config["version"] / f"{split}.csv"
    rows = read_csv(path)
    if any(r["split"] != split or r["preprocess_version"] != preprocess_config["version"] for r in rows):
        raise ValueError(f"{path}: label split/versi preprocessing tidak cocok")
    if split == "train":
        rows = [r for r in rows if r["train_eligible"] == "True"]
    return (rows[:limit] if limit else rows), digest(path)


def load_signal(row, preprocess_config):
    # Sama dengan `preprocess.load_sample`, ditambah verifikasi hash PCM hasil preprocessing.
    pcm = audio.read_segment(ROOT / row["source_audio_path"], row["start_sec"], row["end_sec"])
    processed = preprocess.process_pcm(pcm, preprocess_config)
    if hashlib.sha256(processed.tobytes()).hexdigest() != row["pcm_sha256"]:
        raise ValueError(f"PCM berubah sejak manifest preprocessing dibuat: {row['sample_id']}")
    return audio.to_float(processed)


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_examples(rows, preprocess_config, feature_extractor, tokenizer, with_labels):
    examples = []
    for row in rows:
        signal = load_signal(row, preprocess_config)
        features = feature_extractor(signal, sampling_rate=audio.SAMPLE_RATE,
                                     return_tensors="np").input_features[0].astype(np.float32)
        example = {"sample_id": row["sample_id"], "features": features, "labels": None}
        if with_labels:
            labels = tokenizer(row["text"]).input_ids
            if len(labels) > 448:
                raise ValueError(f"Label terlalu panjang untuk decoder Whisper: {row['sample_id']}")
            example["labels"] = labels
        examples.append(example)
    return examples


def collate(batch, decoder_start_token_id):
    import torch
    features = torch.from_numpy(np.stack([b["features"] for b in batch]))
    if batch[0]["labels"] is None:
        return features, None
    width = max(len(b["labels"]) for b in batch)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, :len(b["labels"])] = torch.tensor(b["labels"])
    # Model menambahkan decoder_start_token sendiri saat menggeser label.
    if (labels[:, 0] == decoder_start_token_id).all():
        labels = labels[:, 1:]
    return features, labels


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def evaluate(model, val_examples, loss_examples, val_rows, tokenizer, config, device, use_amp):
    import torch
    model.eval()
    start_token = model.config.decoder_start_token_id
    losses, predictions = [], []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        for batch in batches(loss_examples, config["eval_batch_size"]):
            features, labels = collate(batch, start_token)
            out = model(input_features=features.to(device), labels=labels.to(device))
            losses.append((out.loss.item(), len(batch)))
        for batch in batches(val_examples, config["eval_batch_size"]):
            features, _ = collate(batch, start_token)
            generated = model.generate(input_features=features.to(device), language=config["language"],
                                       task=config["task"], num_beams=1, do_sample=False,
                                       max_new_tokens=config["generation_max_new_tokens"])
            predictions += [t.strip() for t in tokenizer.batch_decode(generated, skip_special_tokens=True)]
    val_loss = sum(l * n for l, n in losses) / sum(n for _, n in losses)
    scores = selection_metrics.corpus_scores([r["transcript"] for r in val_rows], predictions)
    model.train()
    return val_loss, scores, predictions


def write_predictions(path, run_id, rows, predictions):
    write_csv(path, [{"run_id": run_id, "sample_id": r["sample_id"], "split": r["split"],
                      "reference": r["transcript"], "prediction": p}
                     for r, p in zip(rows, predictions)], PREDICTION_FIELDS)


def convert_to_ct2(hf_dir, ct2_dir, quantization):
    from ctranslate2.converters import TransformersConverter
    converter = TransformersConverter(str(hf_dir), copy_files=["tokenizer.json", "preprocessor_config.json"])
    converter.convert(str(ct2_dir), quantization=quantization, force=False)
    return digest(Path(ct2_dir) / "model.bin")


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=ROOT, text=True).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def run(run_id, config_path, output_root=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id hanya boleh huruf, angka, titik, underscore atau tanda hubung")
    config = load_train_config(config_path)
    out = Path(output_root or DEFAULT_OUTPUT_ROOT) / run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    code_commit, code_dirty = git_state()
    metadata = {"run_id": run_id, "status": "running", "config": config,
                "config_path": Path(config_path).as_posix(), "config_sha256": digest(config_path),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {d.metadata["Name"]: d.version for d in distributions()},
                "code_commit": code_commit, "code_dirty": code_dirty, "seed": config["seed"],
                "selection_metric": selection_metrics.SCORING_POLICY, "test_set_used": False}
    write_json(out / "config.json", config)
    write_json(out / "run.json", metadata)
    history = []
    try:
        import torch
        from transformers import (WhisperFeatureExtractor, WhisperForConditionalGeneration,
                                  WhisperTokenizerFast, get_linear_schedule_with_warmup)

        metadata["source_snapshot"] = infer.provenance()
        pre_cfg = preprocess.load_config(ROOT / config["preprocess_config"])
        split = json.loads((PHASE2 / "configs/split_v1.json").read_text(encoding="utf-8"))
        train_rows, train_sha = load_rows("train", pre_cfg, config.get("limit_train"))
        val_rows, val_sha = load_rows("validation", pre_cfg, config.get("limit_validation"))
        for rows, name in ((train_rows, "train"), (val_rows, "validation")):
            if any(r["speaker_id"] not in split["speakers"][name] for r in rows):
                raise ValueError(f"Speaker {name} tidak sesuai split_v1")
        loss_rows = [r for r in val_rows if r["train_eligible"] == "True"]
        metadata.update(split_version=split["version"], preprocess_version=pre_cfg["version"],
                        preprocess_config_sha256=pre_cfg["_sha256"],
                        train_manifest_sha256=train_sha, validation_manifest_sha256=val_sha,
                        train_utterances=len(train_rows), validation_utterances=len(val_rows),
                        validation_loss_utterances=len(loss_rows),
                        train_audio_sec=round(sum(float(r["duration_sec"]) for r in train_rows), 3))

        set_seed(config["seed"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = bool(config["fp16"]) and device.type == "cuda"
        source = {"pretrained_model_name_or_path": config["base_model_id"], "revision": config["base_revision"]}
        feature_extractor = WhisperFeatureExtractor.from_pretrained(**source)
        tokenizer = WhisperTokenizerFast.from_pretrained(**source)
        tokenizer.set_prefix_tokens(language=config["language"], task=config["task"], predict_timestamps=False)
        model = WhisperForConditionalGeneration.from_pretrained(**source).to(device)
        if config["freeze_encoder"]:
            model.model.encoder.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metadata.update(device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                        amp_fp16=use_amp, torch=torch.__version__,
                        parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=trainable)

        t0 = time.perf_counter()
        train_examples = encode_examples(train_rows, pre_cfg, feature_extractor, tokenizer, True)
        val_examples = encode_examples(val_rows, pre_cfg, feature_extractor, tokenizer, False)
        by_id = {e["sample_id"]: e for e in encode_examples(loss_rows, pre_cfg, feature_extractor, tokenizer, True)}
        loss_examples = [by_id[r["sample_id"]] for r in loss_rows]
        metadata["feature_extraction_sec"] = round(time.perf_counter() - t0, 2)
        write_json(out / "run.json", metadata)

        micro = config["per_device_batch_size"]
        accum = config["gradient_accumulation_steps"]
        steps_per_epoch = math.ceil(math.ceil(len(train_examples) / micro) / accum)
        total_steps = steps_per_epoch * config["num_epochs"]
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = get_linear_schedule_with_warmup(optimizer, round(config["warmup_ratio"] * total_steps), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        metadata.update(effective_batch_size=micro * accum, steps_per_epoch=steps_per_epoch,
                        planned_optimizer_steps=total_steps)

        best = {"epoch": None, "val_wer": math.inf, "val_loss": math.inf}
        best_dir = out / "model/hf_best"

        def record(epoch, steps, train_loss, train_sec):
            t_eval = time.perf_counter()
            val_loss, scores, predictions = evaluate(model, val_examples, loss_examples, val_rows,
                                                     tokenizer, config, device, use_amp)
            write_predictions(out / f"epochs/epoch_{epoch:02d}/predictions.csv", run_id, val_rows, predictions)
            improved = epoch > 0 and (scores["wer"], val_loss) < (best["val_wer"], best["val_loss"])
            if improved:
                best.update(epoch=epoch, val_wer=scores["wer"], val_cer=scores["cer"], val_loss=val_loss)
                model.save_pretrained(best_dir, safe_serialization=True)
                tokenizer.save_pretrained(best_dir)
                feature_extractor.save_pretrained(best_dir)
            row = {"epoch": epoch, "optimizer_steps": steps, "train_loss": train_loss,
                   "val_loss": round(val_loss, 6), "val_wer": round(scores["wer"], 6),
                   "val_cer": round(scores["cer"], 6), "learning_rate": scheduler.get_last_lr()[0],
                   "epoch_train_sec": round(train_sec, 2),
                   "epoch_eval_sec": round(time.perf_counter() - t_eval, 2), "is_best": improved}
            history.append(row)
            write_csv(out / "history.csv", history, HISTORY_FIELDS)
            metadata.update(history=history, best=best)
            write_json(out / "run.json", metadata)
            print(f"{run_id} epoch {epoch}: train_loss={train_loss} val_loss={val_loss:.4f} "
                  f"WER={scores['wer']:.4f} CER={scores['cer']:.4f}{' *best' if improved else ''}", flush=True)

        # Epoch 0 = model pretrained pada pipeline HF yang sama (acuan sebelum fine-tuning).
        record(0, 0, "", 0.0)
        generator = torch.Generator().manual_seed(config["seed"])
        steps, stale = 0, 0
        model.train()
        for epoch in range(1, config["num_epochs"] + 1):
            t_train = time.perf_counter()
            order = torch.randperm(len(train_examples), generator=generator).tolist()
            micro_batches = list(batches([train_examples[i] for i in order], micro))
            running, seen = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for index, batch in enumerate(micro_batches, start=1):
                features, labels = collate(batch, model.config.decoder_start_token_id)
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    loss = model(input_features=features.to(device), labels=labels.to(device)).loss
                scaler.scale(loss / accum).backward()
                running += loss.item() * len(batch)
                seen += len(batch)
                if index % accum == 0 or index == len(micro_batches):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    steps += 1
            if not math.isfinite(running):
                raise FloatingPointError(f"Loss training tidak finite pada epoch {epoch}")
            record(epoch, steps, round(running / seen, 6), time.perf_counter() - t_train)
            stale = 0 if history[-1]["is_best"] else stale + 1
            if stale >= config["early_stopping_patience"]:
                metadata["early_stopped_after_epoch"] = epoch
                break

        if best["epoch"] is None:
            raise RuntimeError("Tidak ada epoch fine-tuning yang dievaluasi")
        t_convert = time.perf_counter()
        metadata["ct2_model_sha256"] = convert_to_ct2(best_dir, out / "model/ct2", config["ct2_quantization"])
        metadata["ct2_conversion_sec"] = round(time.perf_counter() - t_convert, 2)
        if not config.get("keep_hf_checkpoint", False):
            # Bobot HF fp32 (~1 GB) tidak diperlukan pipeline, config/tokenizer dipertahankan.
            for weights in best_dir.glob("*.safetensors"):
                weights.unlink()
        metadata["status"] = "complete"
    except BaseException:
        metadata["status"] = "failed"
        metadata["error"] = traceback.format_exc()
        raise
    finally:
        metadata["elapsed_sec"] = round(time.perf_counter() - started, 2)
        write_json(out / "run.json", metadata)
    return out


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Whisper (HF Transformers + PyTorch) untuk Bahasa Palembang, test set tidak dibaca.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, help="Default phase2/results/training")
    args = parser.parse_args()
    run(args.run_id, args.config, args.output_root)


if __name__ == "__main__":
    main()
