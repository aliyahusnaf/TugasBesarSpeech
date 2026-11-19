# WER/CER untuk pemilihan model di validation, angka resmi dan test dihitung `evaluate.py`.
import jiwer

from phase2.src.text_normalization import normalize_text

SCORING_POLICY = "phase2.src.selection_metrics/v1: normalize_text(keep_hyphen=True); corpus-level jiwer 4 word/char edits"


def normalize(text):
    return normalize_text(text or "", keep_hyphen=True)


def _counts(output):
    edits = output.substitutions + output.deletions + output.insertions
    reference = output.substitutions + output.deletions + output.hits
    return {"edits": edits, "substitutions": output.substitutions, "deletions": output.deletions,
            "insertions": output.insertions, "reference": reference}


def utterance_scores(reference, prediction):
    ref, hyp = normalize(reference), normalize(prediction)
    if not ref:
        raise ValueError("Referensi kosong setelah normalisasi; tidak dapat diskor")
    words = _counts(jiwer.process_words(ref, hyp))
    chars = _counts(jiwer.process_characters(ref, hyp))
    return {"reference_norm": ref, "prediction_norm": hyp,
            "wer": words["edits"] / words["reference"], "cer": chars["edits"] / chars["reference"],
            "word_edits": words["edits"], "ref_words": words["reference"],
            "char_edits": chars["edits"], "ref_chars": chars["reference"],
            "substitutions": words["substitutions"], "deletions": words["deletions"],
            "insertions": words["insertions"]}


def corpus_scores(references, predictions):
    if len(references) != len(predictions) or not references:
        raise ValueError("Jumlah referensi dan prediksi harus sama dan tidak kosong")
    rows = [utterance_scores(r, p) for r, p in zip(references, predictions)]
    total = {k: sum(r[k] for r in rows) for k in
             ("word_edits", "ref_words", "char_edits", "ref_chars", "substitutions", "deletions", "insertions")}
    return {"utterances": len(rows), "wer": total["word_edits"] / total["ref_words"],
            "cer": total["char_edits"] / total["ref_chars"], **total}
