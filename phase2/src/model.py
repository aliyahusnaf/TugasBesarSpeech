"""Shared Whisper loader and decoder for Owners 3/4/6."""
import json
import re
from pathlib import Path

import numpy as np

from phase2.scripts.dataset import PHASE2, ROOT, digest
from phase2.src import preprocess

DEFAULT_CONFIG = PHASE2 / "configs/model.json"


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["technique"] != "whisper" or not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Whisper requires an immutable checkpoint commit SHA")
    if config["decoding"]["task"] != "transcribe" or config["decoding"]["beam_size"] < 1:
        raise ValueError("Expected transcription task and positive beam size")
    return config


class WhisperASR:
    def __init__(self, config):
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model

        self.config = config
        self.preprocess_config = preprocess.load_config(ROOT / config["preprocess_config"])
        path = Path(download_model(config["model_id"], revision=config["revision"]))
        required = ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"]
        if any(not (path / name).is_file() for name in required):
            raise ValueError("Checkpoint must include matching weights, config and tokenizer")
        self.artifact_hashes = {name: digest(path / name) for name in required}
        self.backend = WhisperModel(str(path), device=config["device"],
                                   compute_type=config["compute_type"],
                                   cpu_threads=config["cpu_threads"], local_files_only=True)
        if self.backend.feature_extractor.sampling_rate != 16000:
            raise ValueError("Processor must accept the shared 16 kHz waveform")

    def transcribe_signal(self, signal):
        if (not isinstance(signal, np.ndarray) or signal.dtype != np.float32
                or signal.ndim != 1 or not signal.size or not np.isfinite(signal).all()
                or np.max(np.abs(signal)) > 1):
            raise ValueError("Expected nonempty finite mono float32 waveform in [-1, 1] at 16 kHz")
        segments, _ = self.backend.transcribe(signal, **self.config["decoding"])
        # Consume the lazy generator here so decoding errors reach the run log.
        return "".join(segment.text for segment in segments).strip()

    def transcribe_sample(self, row):
        return self.transcribe_signal(preprocess.load_sample(row, self.preprocess_config))

    def transcribe(self, audio_path):
        return self.transcribe_signal(preprocess.prepare_input(audio_path, self.preprocess_config))


def transcribe(audio_path, config):
    """Whole-file convenience API; reuse WhisperASR for manifest batches."""
    return WhisperASR(config).transcribe(audio_path)
