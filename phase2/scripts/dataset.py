# Persiapan manifest saja; audio dan transkripsi Fase 1 tidak ditulis ulang.

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import wave

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "Korpus_Bahasa_Palembang"
PHASE2 = ROOT / "phase2"
FIELDS = ["sample_id", "audio_path", "start_sec", "end_sec", "duration_sec",
          "transcript", "speaker_id", "source_group", "split", "quality",
          "native", "speech_type", "gender", "topic", "events", "text_tags", "pcm_sha256"]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_snapshot():
    # Metadata dan audio yang benar-benar dibaca ikut di-hash, bukan hanya commit.
    paths = sorted((CORPUS / "audio").rglob("*.wav"))
    paths += [CORPUS / "metadata/metadata.csv", CORPUS / "annotation/annotation.csv"]
    paths += sorted((CORPUS / "documentation/qc").glob("*"))
    files = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths if p.is_file()}
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True).strip()
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"repository": "https://github.com/aliyahusnaf/TugasBesarSpeech",
            "commit": commit, "fingerprint": fingerprint, "files": files}


def verify_source_snapshot(saved, current=None, root=None):
    # Hash lama tetap dipakai. CSV/JSON boleh beda CRLF/LF saja, audio harus persis.
    root = ROOT if root is None else Path(root)
    current = source_snapshot() if current is None else current
    frozen_hash = hashlib.sha256(json.dumps(saved["files"], sort_keys=True).encode()).hexdigest()
    if frozen_hash != saved["fingerprint"]:
        raise ValueError("Saved snapshot fingerprint does not match its inventory")
    if set(saved["files"]) != set(current["files"]):
        raise ValueError("Source inventory changed; request a new snapshot")
    newline_only = []
    for path, expected in saved["files"].items():
        if current["files"][path] == expected:
            continue
        if Path(path).suffix.lower() not in {".csv", ".json"}:
            raise ValueError(f"Source changed: {path}")
        payload = (root / path).read_bytes()
        lf = payload.replace(b"\r\n", b"\n")
        variants = {hashlib.sha256(lf).hexdigest(),
                    hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}
        if expected not in variants:
            raise ValueError(f"Source content changed: {path}")
        newline_only.append(path)
    return newline_only


def inspect_wav(path):
    info = {"audio_path": path.relative_to(ROOT).as_posix(), "filename": path.name,
            "sha256": digest(path), "error": ""}
    try:
        with wave.open(str(path), "rb") as stream:
            rate, channels, width, frames = (stream.getframerate(), stream.getnchannels(),
                                             stream.getsampwidth(), stream.getnframes())
            pcm = stream.readframes(frames)
            info.update(sample_rate=rate, channels=channels, bit_depth=width * 8,
                        frames=frames, duration_sec=frames / rate,
                        pcm_sha256=hashlib.sha256(pcm).hexdigest())
            if (rate, channels, width, stream.getcomptype()) != (16000, 1, 2, "NONE"):
                raise ValueError("format_not_pcm16_mono_16khz")
            if not frames or len(pcm) != frames * channels * width:
                raise ValueError("empty_or_truncated_audio")
            if not any(pcm):
                raise ValueError("zero_signal")
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        info["error"] = str(exc)
    return info


def segment_pcm(path, start, end):
    with wave.open(str(path), "rb") as stream:
        rate = stream.getframerate()
        first, last = round(start * rate), round(end * rate)
        stream.setpos(first)
        pcm = stream.readframes(last - first)
        if len(pcm) != (last - first) * stream.getnchannels() * stream.getsampwidth():
            raise ValueError("segment_payload_truncated")
    return pcm


def collect():
    metadata = read_csv(CORPUS / "metadata/metadata.csv")
    annotations = read_csv(CORPUS / "annotation/annotation.csv")
    meta = {}
    for row in metadata:
        if row["File"] in meta:
            raise ValueError(f"Metadata dobel: {row['File']}")
        meta[row["File"]] = row
    audio = [inspect_wav(p) for p in sorted((CORPUS / "audio").rglob("*.wav"))]
    by_name = {}
    for info in audio:
        if info["filename"] in by_name:
            raise ValueError(f"Nama audio ambigu: {info['filename']}")
        by_name[info["filename"]] = info
    candidates, excluded, issues = [], [], []
    ids = [r["Utterance_ID"] for r in annotations]
    if len(set(ids)) != len(ids):
        raise ValueError("Utterance_ID dobel; perlu dibetulkan sebelum membuat manifest")
    for row in annotations:
        reasons = []
        m, info = meta.get(row["File"]), by_name.get(row["File"])
        if m is None:
            reasons.append("missing_metadata")
        elif m["Quality"].strip().lower() == "reject":
            reasons.append("quality_reject")
        if row["Speaker"] == "OTHER":
            reasons.append("other_speaker")
        elif m and row["Speaker"] != m["Speaker"]:
            reasons.append("speaker_mismatch")
        if info is None or info["error"]:
            reasons.append("missing_or_invalid_audio")
        start, end = 0, 0
        try:
            start, end = float(row["Start"]), float(row["End"])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                raise ValueError()
            if info and (end > info.get("duration_sec", 0) or round(start * 16000) >= round(end * 16000)):
                raise ValueError()
        except ValueError:
            reasons.append("invalid_segment_bounds")
        text = row["Transcription"]
        tags = re.findall(r"\[[^\]]+\]", text)
        if not text.strip():
            reasons.append("empty_transcript")
        elif "[unclear]" in text.lower() or "unclear" in row["Events"].lower():
            reasons.append("unclear_transcript")
        elif not re.search(r"\w", re.sub(r"\[[^\]]+\]", "", text)):
            reasons.append("no_lexical_transcript")
        if reasons:
            excluded.append({"sample_id": row["Utterance_ID"], "file": row["File"],
                             "speaker_id": row["Speaker"], "reasons": ";".join(reasons)})
            continue
        pcm = segment_pcm(ROOT / info["audio_path"], start, end)
        if not any(pcm):
            excluded.append({"sample_id": row["Utterance_ID"], "file": row["File"],
                             "speaker_id": row["Speaker"], "reasons": "zero_signal_segment"})
            continue
        candidates.append(dict(zip(FIELDS, [row["Utterance_ID"], info["audio_path"],
            row["Start"], row["End"], f"{(round(end*16000)-round(start*16000))/16000:.6f}",
            text, row["Speaker"], "phase1_palembang", "", m["Quality"], m["Native"],
            m["Speech_Type"], m["Gender"], row["Topic"], row["Events"],
            ";".join(tags), hashlib.sha256(pcm).hexdigest()])))
        if abs(float(row["Duration"]) - (end - start)) > .02:
            issues.append({"sample_id": row["Utterance_ID"], "issue": "annotation_duration_mismatch"})
    groups = {}
    for row in candidates:
        groups.setdefault(row["pcm_sha256"], []).append(row)
    accepted = []
    for group in groups.values():
        group.sort(key=lambda r: r["sample_id"])
        ambiguous = len({r["speaker_id"] for r in group}) > 1
        for i, row in enumerate(group):
            if ambiguous or i:
                excluded.append({"sample_id": row["sample_id"], "file": Path(row["audio_path"]).name,
                                 "speaker_id": row["speaker_id"],
                                 "reasons": "duplicate_pcm_identity_conflict" if ambiguous else "duplicate_segment_pcm"})
            else:
                accepted.append(row)
    return sorted(accepted, key=lambda r: r["sample_id"]), excluded, audio, metadata, annotations, issues
