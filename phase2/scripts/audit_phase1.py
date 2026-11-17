import hashlib
import json
from collections import Counter
import wave

from .dataset import CORPUS, PHASE2, ROOT, read_csv, write_csv, write_json, verify_source_snapshot


def pcm(path):
    with wave.open(str(path), "rb") as stream:
        return stream.readframes(stream.getnframes())


def audit_relations():
    rows, cache = [], {}
    for entry in read_csv(CORPUS / "documentation/qc/segment_offsets.csv"):
        speaker = entry["speaker"]
        full = CORPUS / "audio" / speaker / f"{speaker}.wav"
        topic = CORPUS / "audio" / speaker / entry["file"]
        if speaker not in cache:
            cache[speaker] = pcm(full)
        original, chunk = cache[speaker], pcm(topic)
        expected = round(float(entry["start_in_full"]) * 16000)
        match = None
        # Offset lama dibulatkan ke milidetik: cek selisih sampai 16 sampel.
        for offset in sorted(range(max(0, expected-16), expected+17), key=lambda n: abs(n-expected)):
            if original[offset*2:offset*2 + len(chunk)] == chunk:
                match = offset
                break
        compared, different, missing = len(chunk)//2, 0, 0
        if match is None:
            probe = original.find(chunk[64:192], max(0, expected-16)*2+64, (expected+17)*2+192)
            aligned = probe-64 if probe >= 64 else expected*2
            part = original[aligned:aligned+len(chunk)]
            compared = min(len(part), len(chunk))//2
            different = sum(a != b for a, b in zip(memoryview(part[:compared*2]).cast('h'),
                                                   memoryview(chunk[:compared*2]).cast('h')))
            missing = len(chunk)//2-compared
        rows.append({"speaker_id": speaker, "topic_file": entry["file"],
                     "full_file": full.relative_to(ROOT).as_posix(),
                     "qc_start_sec": entry["start_in_full"],
                     "matched_start_sample": "" if match is None else match,
                     "exact_pcm_match": match is not None, "compared_samples": compared,
                     "different_samples": different, "samples_beyond_full_recording": missing})
    write_csv(PHASE2 / "data/source_relations.csv", rows,
              ["speaker_id", "topic_file", "full_file", "qc_start_sec", "matched_start_sample", "exact_pcm_match",
               "compared_samples", "different_samples", "samples_beyond_full_recording"])
    manifests = [r for split in ["train", "validation", "test"]
                 for r in read_csv(PHASE2 / f"data/manifests/{split}.csv")]
    flags = []
    for r in manifests:
        duration = float(r["duration_sec"])
        reasons = []
        if duration < .25:
            reasons.append("shorter_than_250ms")
        if duration > 30:
            reasons.append("longer_than_30s")
        if r["text_tags"] or r["events"]:
            reasons.append("text_or_event_tags")
        if reasons:
            flags.append({"sample_id": r["sample_id"], "flags": ";".join(reasons)})
    write_csv(PHASE2 / "data/review_flags.csv", flags, ["sample_id", "flags"])
    annotations = read_csv(CORPUS / "annotation/annotation.csv")
    other = [r for r in annotations if r["Speaker"] == "OTHER"]
    overlaps = []
    for row in manifests:
        for o in other:
            if row["audio_path"].endswith("/" + o["File"]) and max(float(row["start_sec"]), float(o["Start"])) < min(float(row["end_sec"]), float(o["End"])):
                overlaps.append([row["sample_id"], o["Utterance_ID"]])
    summary = {"topic_files_checked": len(rows),
               "exact_source_topic_matches": sum(r["exact_pcm_match"] for r in rows),
               "unmatched_files": [r["topic_file"] for r in rows if not r["exact_pcm_match"]],
               "accepted_segments_overlapping_OTHER": overlaps,
               "review_flags": len(flags),
               "shorter_than_250ms": sum(float(r["duration_sec"]) < .25 for r in manifests),
               "longer_than_30s": sum(float(r["duration_sec"]) > 30 for r in manifests),
               "tag_counts_accepted": dict(Counter(tag for r in manifests for tag in r["text_tags"].split(";") if tag))}
    metadata = read_csv(CORPUS / "metadata/metadata.csv")
    audio_paths = {p.name: p for p in (CORPUS / "audio").rglob("*.wav")}
    annotated_files = {r["File"] for r in annotations}
    summary["metadata_missing_audio"] = [r["File"] for r in metadata if r["File"] not in audio_paths]
    summary["annotation_missing_metadata"] = sorted(annotated_files - {r["File"] for r in metadata})
    summary["metadata_without_annotation"] = sorted({r["File"] for r in metadata} - annotated_files)
    summary["duration_metadata_warnings"] = []
    for r in metadata:
        with wave.open(str(audio_paths[r["File"]]), "rb") as w:
            duration = w.getnframes()/w.getframerate()
        if abs(duration-float(r["Duration"])) > .02:
            summary["duration_metadata_warnings"].append(r["File"])
    write_json(PHASE2 / "data/audit_details.json", summary)
    frozen = json.loads((PHASE2 / "data/source_snapshot.json").read_text(encoding="utf-8"))
    verify_source_snapshot(frozen)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    audit_relations()
