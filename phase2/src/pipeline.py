import time
from dataclasses import dataclass
from pathlib import Path

from phase2.scripts.dataset import PHASE2
from phase2.src import audio, model, preprocess
from phase2.src.text_normalization import normalize_text

FINAL_CONFIG = PHASE2 / "configs/experiments/finetuned_ft_lr1e-5_001_beam5.json"
PRETRAINED_CONFIG = PHASE2 / "configs/model.json"


@dataclass
class Transcription:
    audio_path: str
    duration_sec: float
    text: str
    normalized_text: str
    elapsed_sec: float


class TranscriptionPipeline:
    def __init__(self, config_path=FINAL_CONFIG):
        self.config_path = Path(config_path)
        self.config = model.load_config(self.config_path)
        self.asr = model.WhisperASR(self.config)
        self.preprocess_config = self.asr.preprocess_config

    def load_audio(self, audio_path):
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"{audio_path}: file tidak ditemukan")
        pcm = audio.load_audio_file(audio_path)
        if len(pcm) == 0:
            raise ValueError(f"{audio_path}: audio kosong")
        return pcm

    def preprocess(self, pcm):
        processed = preprocess.process_pcm(pcm, self.preprocess_config)
        return audio.to_float(processed)

    def transcribe(self, audio_path):
        started = time.perf_counter()
        signal = self.preprocess(self.load_audio(audio_path))
        text = self.asr.transcribe_signal(signal)
        keep_hyphen = self.preprocess_config["text"]["keep_hyphen"]
        return Transcription(
            audio_path=str(audio_path),
            duration_sec=len(signal) / audio.SAMPLE_RATE,
            text=text,
            normalized_text=normalize_text(text, keep_hyphen=keep_hyphen),
            elapsed_sec=time.perf_counter() - started,
        )
