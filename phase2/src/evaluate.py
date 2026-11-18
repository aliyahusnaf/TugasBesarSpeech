from pathlib import Path

import jiwer

from phase2.scripts.dataset import PHASE2, read_csv, write_csv, write_json
from phase2.src.text_normalization import MASKED_NAME_TAG, normalize_text

COUNTS = ("ref_words", "word_errors", "substitutions", "deletions", "insertions", "ref_chars", "char_errors")


def score(reference, prediction):
    ref = normalize_text(reference, keep_hyphen=True)
    hyp = normalize_text(prediction or "", keep_hyphen=True)
    if not ref:
        raise ValueError("Referensi kosong setelah normalisasi")
    w, c = jiwer.process_words(ref, hyp), jiwer.process_characters(ref, hyp)
    counts = {"ref_words": w.hits + w.substitutions + w.deletions,
              "word_errors": w.substitutions + w.deletions + w.insertions,
              "substitutions": w.substitutions, "deletions": w.deletions, "insertions": w.insertions,
              "ref_chars": c.hits + c.substitutions + c.deletions,
              "char_errors": c.substitutions + c.deletions + c.insertions}
    return {**counts, "wer": counts["word_errors"] / counts["ref_words"],
            "cer": counts["char_errors"] / counts["ref_chars"]}


def total(rows):
    sums = {k: sum(r[k] for r in rows) for k in COUNTS}
    return {"utterances": len(rows), "wer": sums["word_errors"] / sums["ref_words"],
            "cer": sums["char_errors"] / sums["ref_chars"], **sums}


def evaluate(predictions, manifest):
    references = {r["sample_id"]: r for r in manifest}
    if sorted(p["sample_id"] for p in predictions) != sorted(references):
        raise ValueError("Ujaran di prediksi tidak sama dengan manifest")
    if any(p["reference"] != references[p["sample_id"]]["transcript"] for p in predictions):
        raise ValueError("Referensi di prediksi berbeda dari manifest")
    rows = [{"sample_id": p["sample_id"], "speaker_id": references[p["sample_id"]]["speaker_id"],
             # Tag [nama] dibuang dari referensi, jadi nama yang ditranskripsi model terhitung insertion
             "masked_name": MASKED_NAME_TAG in p["reference"].lower(),
             **score(p["reference"], p["prediction"])} for p in predictions]
    primary = [r for r in rows if not r["masked_name"]]
    speakers = sorted({r["speaker_id"] for r in primary})
    result = {"primary": total(primary), "all": total(rows),
              "by_speaker": {s: total([r for r in primary if r["speaker_id"] == s]) for s in speakers}}
    return result, rows


def run(predictions_path, output_dir=None):
    predictions = read_csv(predictions_path)
    manifest = read_csv(PHASE2 / f"data/manifests/{predictions[0]['split']}.csv")
    result, rows = evaluate(predictions, manifest)
    out = Path(output_dir or Path(predictions_path).parent / "evaluation")
    write_json(out / "metrics.json", result)
    write_csv(out / "utterance_scores.csv", rows, list(rows[0]))
    return result
