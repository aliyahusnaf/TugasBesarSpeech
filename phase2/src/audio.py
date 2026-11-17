import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
FULL_SCALE = 32768.0


def wav_format(path):
    with wave.open(str(path), "rb") as stream:
        return {"sample_rate": stream.getframerate(), "channels": stream.getnchannels(),
                "bit_depth": stream.getsampwidth() * 8, "compression": stream.getcomptype(),
                "frames": stream.getnframes()}


def is_standard_format(fmt):
    return (fmt["sample_rate"], fmt["channels"], fmt["bit_depth"], fmt["compression"]) == \
        (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH * 8, "NONE")


def _require_standard(stream, path):
    fmt = (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype())
    if fmt != (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH, "NONE"):
        raise ValueError(f"{path}: bukan WAV PCM 16-bit mono 16 kHz {fmt}")


def read_wav(path):
    with wave.open(str(path), "rb") as stream:
        _require_standard(stream, path)
        data = stream.readframes(stream.getnframes())
    return np.frombuffer(data, dtype="<i2").copy()


def read_segment(path, start_sec, end_sec):
    """Potongan [start, end) dengan batas sampel round(detik * 16000)."""
    first, last = round(float(start_sec) * SAMPLE_RATE), round(float(end_sec) * SAMPLE_RATE)
    with wave.open(str(path), "rb") as stream:
        _require_standard(stream, path)
        if not 0 <= first < last <= stream.getnframes():
            raise ValueError(f"{path}: batas segmen di luar file ({start_sec}, {end_sec})")
        stream.setpos(first)
        data = stream.readframes(last - first)
    pcm = np.frombuffer(data, dtype="<i2").copy()
    if len(pcm) != last - first:
        raise ValueError(f"{path}: payload segmen terpotong")
    return pcm


def write_wav(path, pcm):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with wave.open(str(tmp), "wb") as stream:
        stream.setnchannels(CHANNELS)
        stream.setsampwidth(SAMPLE_WIDTH)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
    tmp.replace(path)


def find_ffmpeg():
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg tidak ditemukan di PATH; dibutuhkan untuk audio non-standar")
    return path


def _ffmpeg_args(src):
    return [find_ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(src),
            "-map", "0:a:0", "-vn", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le"]


def decode_with_ffmpeg(src):
    result = subprocess.run(_ffmpeg_args(src) + ["-f", "s16le", "-"], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"{src}: ffmpeg gagal: {result.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(result.stdout, dtype="<i2").copy()


def convert_to_standard(src, dst):
    """Salinan baru WAV 16 kHz mono 16-bit; file sumber tidak disentuh."""
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("Output konversi tidak boleh menimpa sumber")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.wav")
    result = subprocess.run(_ffmpeg_args(src) + ["-y", str(tmp)], capture_output=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{src}: ffmpeg gagal: {result.stderr.decode(errors='replace').strip()}")
    tmp.replace(dst)
    return wav_format(dst)


def load_audio_file(path):
    """PCM int16 16 kHz mono dari file apa pun; WAV standar dibaca tanpa ffmpeg."""
    try:
        if is_standard_format(wav_format(path)):
            return read_wav(path)
    except (wave.Error, EOFError):
        pass
    return decode_with_ffmpeg(path)


def to_float(pcm):
    return np.asarray(pcm, dtype=np.float32) / FULL_SCALE


def to_pcm(signal):
    return np.clip(np.round(np.asarray(signal, dtype=np.float64) * FULL_SCALE),
                   -FULL_SCALE, FULL_SCALE - 1).astype("<i2")


def db(value):
    return float(20 * np.log10(value)) if value > 0 else float("-inf")


def frame_levels(signal, frame_sec, hop_sec):
    frame, hop = round(frame_sec * SAMPLE_RATE), round(hop_sec * SAMPLE_RATE)
    if len(signal) < frame:
        return np.array([db(np.sqrt(np.mean(np.square(signal))))]) if len(signal) else np.array([])
    count = 1 + (len(signal) - frame) // hop
    index = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    rms = np.sqrt(np.mean(np.square(signal[index], dtype=np.float64), axis=1))
    with np.errstate(divide="ignore"):
        return 20 * np.log10(rms)


def active_bounds(levels, threshold_db, hop_sec, frame_sec, length):
    """Rentang sampel dari frame aktif pertama sampai terakhir, atau None."""
    active = np.flatnonzero(levels >= threshold_db)
    if not len(active):
        return None
    hop, frame = round(hop_sec * SAMPLE_RATE), round(frame_sec * SAMPLE_RATE)
    return int(active[0] * hop), int(min(length, active[-1] * hop + frame))


def longest_clip_run(pcm):
    clipped = np.concatenate(([False], (pcm >= 32767) | (pcm <= -32768), [False]))
    edges = np.flatnonzero(np.diff(clipped.astype(np.int8)))
    return int((edges[1::2] - edges[::2]).max()) if len(edges) else 0


def audio_metrics(pcm, qc):
    signal = to_float(pcm)
    levels = frame_levels(signal, qc["frame_sec"], qc["hop_sec"])
    bounds = active_bounds(levels, qc["silence_threshold_dbfs"], qc["hop_sec"], qc["frame_sec"], len(pcm))
    finite = levels[np.isfinite(levels)]
    duration = len(pcm) / SAMPLE_RATE
    return {
        "duration_sec": duration,
        "peak_dbfs": db(np.max(np.abs(signal))) if len(signal) else float("-inf"),
        "rms_dbfs": db(np.sqrt(np.mean(np.square(signal, dtype=np.float64)))) if len(signal) else float("-inf"),
        "dc_offset": float(np.mean(signal)) if len(signal) else 0.0,
        "clipped_samples": int(np.count_nonzero((pcm >= 32767) | (pcm <= -32768))),
        "longest_clip_run": longest_clip_run(pcm),
        "active_ratio": float(np.mean(levels >= qc["silence_threshold_dbfs"])) if len(levels) else 0.0,
        "leading_silence_sec": bounds[0] / SAMPLE_RATE if bounds else duration,
        "trailing_silence_sec": (len(pcm) - bounds[1]) / SAMPLE_RATE if bounds else duration,
        "dynamic_range_db": float(np.percentile(finite, 95) - np.percentile(finite, 10)) if len(finite) else 0.0,
    }


def apply_variant(pcm, audio_cfg, qc):
    """Variasi preprocessing deterministik; konfigurasi default mengembalikan PCM asli."""
    pcm = np.asarray(pcm, dtype="<i2")
    if audio_cfg.get("trim_silence"):
        levels = frame_levels(to_float(pcm), qc["frame_sec"], qc["hop_sec"])
        bounds = active_bounds(levels, qc["silence_threshold_dbfs"], qc["hop_sec"], qc["frame_sec"], len(pcm))
        if bounds:
            pad = round(audio_cfg["trim_pad_sec"] * SAMPLE_RATE)
            pcm = pcm[max(0, bounds[0] - pad):min(len(pcm), bounds[1] + pad)]
    mode = audio_cfg.get("amplitude", "none")
    if mode == "none":
        return pcm.copy()
    signal = to_float(pcm).astype(np.float64)
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak == 0:
        return pcm.copy()
    limit = 10 ** (audio_cfg["peak_limit_dbfs"] / 20)
    if mode == "peak":
        gain = limit / peak
    elif mode == "rms":
        rms = float(np.sqrt(np.mean(np.square(signal))))
        gain = min(10 ** (audio_cfg["target_rms_dbfs"] / 20) / rms, limit / peak)
    else:
        raise ValueError(f"Mode amplitudo tidak dikenal: {mode}")
    return to_pcm(signal * gain)
