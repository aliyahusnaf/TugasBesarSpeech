"""Tahap 1 anotasi: transkripsi otomatis awal (draft) dengan faster-whisper large-v3.

Hasilnya hanya draf: kata dan timestamp kasar per kata disimpan sebagai JSON,
lalu dikoreksi manual / dicocokkan dengan naskah dan di-align ulang oleh
utils/build_annotation.py.

Pemakaian:
    python utils/asr_whisper.py --out <dir_json> S01 S02 ...
"""
import argparse
import json
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio"

# Prompt berisi kosakata khas Palembang + filler agar Whisper menulis
# sesuai ejaan Palembang (apo, dak, nian, ...) dan tidak membuang disfluency.
PROMPT = ("ee jadi aku ni wong palembang, em, kito biaso ngomong cak ini bae. "
          "apo, dak, idak, nian, galo, kau, nak, la, jugo, samo, caro, mano, "
          "cak mano, ngapo, dewek, budak, ado, kalu, biaso, makanan pempek, tekwan, "
          "model, laksan, celimpungan. ee aku suko nian.")


def transcribe_sequential(model, wav, prompt):
    """Mode biasa (pembanding / second opinion untuk review)."""
    return model.transcribe(
        str(wav), language="id", beam_size=5, word_timestamps=True,
        initial_prompt=prompt, condition_on_previous_text=False, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=200),
        temperature=[0.0, 0.2, 0.4, 0.6])


def transcribe(model, wav, prompt, sequential=False):
    if sequential:
        segs, info = transcribe_sequential(model.model, wav, prompt)
        return pack(segs, info)
    # Mode batched: audio dipecah per potongan VAD dan prompt dipakai di SETIAP
    # potongan (di mode biasa prompt hanya berlaku untuk 30 detik pertama).
    segs, info = model.transcribe(
        str(wav), language="id", beam_size=5, word_timestamps=True,
        initial_prompt=prompt, batch_size=4, chunk_length=20,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
    )
    return pack(segs, info)


def pack(segs, info):
    out = []
    for s in segs:
        out.append({
            "start": s.start, "end": s.end, "text": s.text,
            "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
            "compression_ratio": s.compression_ratio,
            "words": [{"start": w.start, "end": w.end, "word": w.word, "prob": w.probability}
                      for w in (s.words or [])],
        })
    return {"duration": info.duration, "segments": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speakers", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--prompt-file", help="naskah (read speech) untuk prompt awal")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompt = PROMPT
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")[:600]
    model = BatchedInferencePipeline(
        WhisperModel("large-v3", device="cuda", compute_type="int8_float16"))
    for spk in args.speakers:
        for wav in sorted((AUDIO / spk).glob(f"{spk}_T*.wav")):
            dst = out / f"{wav.stem}.json"
            if dst.exists():
                continue
            res = transcribe(model, wav, prompt, args.sequential)
            dst.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(wav.name, len(res["segments"]), "segments", flush=True)


if __name__ == "__main__":
    main()
