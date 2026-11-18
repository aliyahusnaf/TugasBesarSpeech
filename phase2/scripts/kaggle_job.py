"""Menyiapkan (prepare) dan mengumpulkan (collect) job fine-tuning Kaggle."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from phase2.scripts.dataset import PHASE2, ROOT, digest, write_json

REPO_URL = "https://github.com/aliyahusnaf/TugasBesarSpeech.git"
BUILD = PHASE2 / ".kaggle_build"
DATASET_SLUG = "palembang-asr-owner4-code"
KERNEL_SLUG = "palembang-asr-whisper-finetune"
DEFAULT_JOB = PHASE2 / "configs/train/kaggle_job_v1.json"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def prepare(job_path, username, remote_base="origin/phase2"):
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Worktree punya perubahan yang belum di-commit; commit dulu agar run dapat ditelusuri.")
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    tracked = set(git("ls-files").splitlines())
    needed = [r["config"] for r in job["runs"]] + ["phase2/src/train.py", "phase2/kaggle/run_kaggle.py",
                                                   "phase2/requirements-train.txt"]
    missing = [p for p in needed if p not in tracked]
    if missing:
        raise SystemExit(f"File belum di-commit: {missing}")
    branch = git("symbolic-ref", "--short", "HEAD")
    code_commit = git("rev-parse", "HEAD")
    base = git("merge-base", "HEAD", remote_base)
    if BUILD.exists():
        shutil.rmtree(BUILD)
    dataset, kernel = BUILD / "dataset", BUILD / "kernel"
    dataset.mkdir(parents=True)
    kernel.mkdir(parents=True)
    bundle = dataset / "owner4.bundle"
    git("bundle", "create", str(bundle), f"{base}..{branch}")
    write_json(dataset / "job.json", {
        "repo_url": REPO_URL, "base_commit": base, "code_commit": code_commit,
        "bundle_file": bundle.name, "bundle_ref": f"refs/heads/{branch}",
        "bundle_sha256": digest(bundle), "job_config": Path(job_path).resolve().relative_to(ROOT).as_posix(),
        "runs": job["runs"], "created_utc": datetime.now(timezone.utc).isoformat()})
    write_json(dataset / "dataset-metadata.json", {
        "title": DATASET_SLUG, "id": f"{username}/{DATASET_SLUG}", "licenses": [{"name": "other"}]})
    shutil.copy2(PHASE2 / "kaggle/run_kaggle.py", kernel / "run_kaggle.py")
    write_json(kernel / "kernel-metadata.json", {
        "id": f"{username}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "run_kaggle.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": [f"{username}/{DATASET_SLUG}"],
        "competition_sources": [], "kernel_sources": []})
    print(f"Build siap di {BUILD.relative_to(ROOT)} untuk commit {code_commit[:7]} (base {base[:7]}).")


def find_run_dir(download, run_id):
    hits = [p.parent for p in Path(download).rglob("run.json") if p.parent.name == run_id]
    if len(hits) != 1:
        raise SystemExit(f"Output run {run_id} tidak ditemukan tepat satu kali di {download}")
    return hits[0]


def collect(download, job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    base_model_config = json.loads((PHASE2 / "configs/model.json").read_text(encoding="utf-8"))
    job_out = PHASE2 / "results/training" / Path(job_path).stem
    job_out.mkdir(parents=True, exist_ok=True)
    for name in ("kaggle_job_status.json", "pip_freeze.txt"):
        found = list(Path(download).rglob(name))
        if found:
            shutil.copy2(found[0], job_out / name)
    for item in job["runs"]:
        run_id = item["run_id"]
        src = find_run_dir(download, run_id)
        run = json.loads((src / "run.json").read_text(encoding="utf-8"))
        dst = PHASE2 / "results/training" / run_id
        if dst.exists():
            raise SystemExit(f"{dst} sudah ada; run ID tidak boleh ditimpa")
        dst.mkdir(parents=True)
        for name in ("config.json", "run.json", "history.csv"):
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        if (src / "epochs").exists():
            shutil.copytree(src / "epochs", dst / "epochs")
        if run["status"] != "complete":
            print(f"{run_id}: status {run['status']} — log disalin, model tidak dikumpulkan")
            continue
        model_dir = PHASE2 / "models" / run_id / "ct2"
        shutil.copytree(src / "model/ct2", model_dir)
        if digest(model_dir / "model.bin") != run["ct2_model_sha256"]:
            raise SystemExit(f"{run_id}: hash model.bin tidak cocok dengan run.json")
        config = {k: v for k, v in base_model_config.items() if k != "revision"}
        config.update(model_id=f"local:{run_id}", model_path=model_dir.relative_to(ROOT).as_posix(),
                      base_model_id=run["config"]["base_model_id"],
                      base_revision=run["config"]["base_revision"],
                      model_sha256=run["ct2_model_sha256"], training_run=run_id)
        write_json(PHASE2 / f"configs/experiments/finetuned_{run_id}.json", config)
        print(f"{run_id}: best epoch {run['best']['epoch']} val WER {run['best']['val_wer']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--username", required=True, help="Username Kaggle (tidak disimpan di repo)")
    p.add_argument("--job", type=Path, default=DEFAULT_JOB)
    c = sub.add_parser("collect")
    c.add_argument("--download", type=Path, required=True, help="Folder hasil `kaggle kernels output`")
    c.add_argument("--job", type=Path, default=DEFAULT_JOB)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.job, args.username)
    else:
        collect(args.download, args.job)


if __name__ == "__main__":
    main()
