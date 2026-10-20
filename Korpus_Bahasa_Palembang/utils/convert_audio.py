import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw"
AUDIO_DIR = ROOT / "audio"
REPORT_PATH = ROOT / "conversion_report.csv"

SOURCE_EXTS = {".m4a", ".mp4", ".wav", ".aac"}
TARGET_CODEC = "pcm_s16le"
TARGET_SR = 16000
TARGET_CH = 1
DURATION_TOLERANCE_SEC = 1.0

# Hening (detik) yang ditambahkan di akhir output per penutur
PAD_END_SEC = {"S08": 8.0}

REPORT_COLUMNS = [
    "source_file", "output_file", "source_format", "source_sample_rate",
    "source_channels", "output_sample_rate", "output_channels", "output_codec",
    "duration_sec", "status", "note",
]


def find_tool(name):
    path = shutil.which(name)
    if path:
        return path
    winget_link = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Links" / f"{name}.exe"
    if winget_link.exists():
        return str(winget_link)
    return None


FFMPEG = find_tool("ffmpeg")
FFPROBE = find_tool("ffprobe")


def speaker_id(src):
    """raw/SNN/x.ext -> SNN ; raw/SNN.ext atau raw/NN-nama.ext -> SNN."""
    parent = src.parent.name
    if re.fullmatch(r"[Ss]\d{2}", parent):
        return parent.upper()
    m = re.match(r"[Ss]?(\d+)", src.stem)
    if m:
        return f"S{int(m.group(1)):02d}"
    return None


def collect_sources():
    files = [p for p in RAW_DIR.rglob("*") if p.is_file() and p.suffix.lower() in SOURCE_EXTS]
    return sorted(files, key=lambda p: (speaker_id(p) or "~", p.name.lower()))


def probe(path):
    """Kembalikan info stream audio pertama + durasi, atau raise RuntimeError."""
    cmd = [FFPROBE, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or "ffprobe gagal")
    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        raise RuntimeError("tidak ada stream audio")
    a = audio[0]
    fmt = data.get("format", {})
    duration = a.get("duration") or fmt.get("duration")
    return {
        "container": fmt.get("format_name", ""),
        "codec": a.get("codec_name", ""),
        "sample_rate": int(a["sample_rate"]) if a.get("sample_rate") else None,
        "channels": a.get("channels"),
        "duration": float(duration) if duration else None,
        "has_video": any(s.get("codec_type") == "video" and
                         not s.get("disposition", {}).get("attached_pic") for s in streams),
        "n_audio": len(audio),
    }


def decoded_duration(path):
    """Durasi hasil decode penuh (akurat untuk ADTS .aac yang durasinya hanya diestimasi ffprobe)."""
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-i", str(path), "-vn", "-map", "0:a:0", "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", res.stderr)
    if not times:
        return None
    h, m, s = times[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def convert(src, dst, pad_end=0.0):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.wav")
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
           "-i", str(src),
           "-map", "0:a:0", "-vn", "-sn", "-dn"]
    if pad_end:
        cmd += ["-af", f"apad=pad_dur={pad_end}"]
    cmd += ["-ac", str(TARGET_CH), "-ar", str(TARGET_SR), "-c:a", TARGET_CODEC, str(tmp)]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "ffmpeg gagal")
    os.replace(tmp, dst)


def rel(p):
    return p.relative_to(ROOT).as_posix()


def fmt_dur(sec):
    if sec is None:
        return "-"
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:05.2f}"


def list_sources(sources):
    print(f"Ditemukan {len(sources)} file di {rel(RAW_DIR)}/\n")
    header = f"{'Speaker':<8}{'File sumber':<26}{'Container':<22}{'Codec':<11}{'SR (Hz)':>8}{'Ch':>4}{'Durasi':>11}  Catatan"
    print(header)
    print("-" * len(header))
    for src in sources:
        spk = speaker_id(src) or "?"
        try:
            info = probe(src)
            notes = []
            if info["has_video"]:
                notes.append("ada video (dibuang)")
            if info["n_audio"] > 1:
                notes.append(f"{info['n_audio']} stream audio, pakai yg pertama")
            print(f"{spk:<8}{src.name:<26}{info['container'][:21]:<22}{info['codec']:<11}"
                  f"{info['sample_rate'] or '-':>8}{info['channels'] or '-':>4}{fmt_dur(info['duration']):>11}  "
                  f"{'; '.join(notes)}")
        except Exception as e:
            print(f"{spk:<8}{src.name:<26}PROBE GAGAL: {e}")


def run(sources):
    rows = []
    used_outputs = {}
    for i, src in enumerate(sources, 1):
        spk = speaker_id(src)
        row = dict.fromkeys(REPORT_COLUMNS, "")
        row["source_file"] = rel(src)
        notes = []
        print(f"[{i}/{len(sources)}] {rel(src)}", end=" ... ", flush=True)
        try:
            if spk is None:
                raise RuntimeError("tidak bisa menentukan speaker ID dari nama file/folder")
            dst = AUDIO_DIR / spk / f"{spk}.wav"
            if dst in used_outputs:
                raise RuntimeError(f"output {rel(dst)} sudah dipakai oleh {used_outputs[dst]}")
            if RAW_DIR in dst.resolve().parents:
                raise RuntimeError("output tidak boleh berada di dalam raw/")
            used_outputs[dst] = rel(src)
            row["output_file"] = rel(dst)

            src_info = probe(src)
            row["source_format"] = f"{src.suffix.lower().lstrip('.')} ({src_info['container']}/{src_info['codec']})"
            row["source_sample_rate"] = src_info["sample_rate"] or ""
            row["source_channels"] = src_info["channels"] or ""
            if src_info["has_video"]:
                notes.append("video dibuang")
            if src_info["channels"] and src_info["channels"] > 1:
                notes.append(f"downmix {src_info['channels']}ch->mono")

            pad = PAD_END_SEC.get(spk, 0.0)
            if pad:
                notes.append(f"ditambah {pad:g}s hening di akhir")

            convert(src, dst, pad)

            out = probe(dst)
            row["output_sample_rate"] = out["sample_rate"]
            row["output_channels"] = out["channels"]
            row["output_codec"] = out["codec"]
            row["duration_sec"] = f"{out['duration']:.3f}" if out["duration"] is not None else ""

            status = "OK"
            bad = []
            if out["codec"] != TARGET_CODEC:
                bad.append(f"codec={out['codec']}")
            if out["sample_rate"] != TARGET_SR:
                bad.append(f"sr={out['sample_rate']}")
            if out["channels"] != TARGET_CH:
                bad.append(f"ch={out['channels']}")
            if bad:
                status = "FAILED"
                notes.append("format output salah: " + ", ".join(bad))

            src_dur = src_info["duration"]
            if src_dur is None or out["duration"] is None:
                notes.append("durasi sumber/output tidak terbaca")
            elif abs(src_dur + pad - out["duration"]) > DURATION_TOLERANCE_SEC:
                # Header durasi bisa meleset (mis. ADTS .aac diestimasi dari bitrate); cek via decode penuh
                dec = decoded_duration(src)
                if dec is not None and abs(dec + pad - out["duration"]) <= DURATION_TOLERANCE_SEC:
                    notes.append(f"durasi header sumber {src_dur:.2f}s meleset; decode penuh {dec:.2f}s cocok")
                else:
                    if status == "OK":
                        status = "DURATION_MISMATCH"
                    notes.append(f"durasi sumber {src_dur:.2f}s vs output {out['duration']:.2f}s"
                                 + (f" (decode sumber {dec:.2f}s)" if dec is not None else ""))
            row["status"] = status
            print(status)
        except Exception as e:
            row["status"] = "FAILED"
            notes.append(str(e).replace("\n", " "))
            print("FAILED")
        row["note"] = "; ".join(notes)
        rows.append(row)

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nLaporan: {rel(REPORT_PATH)}")
    summarize(rows)
    return rows


def summarize(rows):
    per_spk = defaultdict(lambda: [0, 0.0])
    for r in rows:
        if r["status"] in ("OK", "DURATION_MISMATCH") and r["output_file"]:
            spk = Path(r["output_file"]).parent.name
            per_spk[spk][0] += 1
            per_spk[spk][1] += float(r["duration_sec"] or 0)

    print("\nRINGKASAN PER PENUTUR")
    print(f"{'Speaker':<9}{'Jumlah file':>12}{'Durasi (menit)':>16}")
    total_n, total_d = 0, 0.0
    for spk in sorted(per_spk):
        n, d = per_spk[spk]
        total_n += n
        total_d += d
        print(f"{spk:<9}{n:>12}{d / 60:>16.2f}")
    print(f"{'TOTAL':<9}{total_n:>12}{total_d / 60:>16.2f}")

    problems = [r for r in rows if r["status"] != "OK"]
    print("\nFILE BERMASALAH:" + (" tidak ada" if not problems else ""))
    for r in problems:
        print(f"  [{r['status']}] {r['source_file']}: {r['note']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="hanya tampilkan daftar file sumber + info ffprobe")
    args = ap.parse_args()

    missing = [n for n, p in (("ffmpeg", FFMPEG), ("ffprobe", FFPROBE)) if not p]
    if missing:
        sys.exit(f"Tidak ditemukan: {', '.join(missing)}. Install dulu: winget install --id Gyan.FFmpeg -e")
    if not RAW_DIR.is_dir():
        sys.exit(f"Folder sumber tidak ada: {RAW_DIR}")

    sources = collect_sources()
    if not sources:
        sys.exit(f"Tidak ada file {sorted(SOURCE_EXTS)} di {RAW_DIR}")

    if args.list:
        list_sources(sources)
    else:
        run(sources)


if __name__ == "__main__":
    main()
