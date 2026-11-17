"""Entry point kernel Kaggle (GPU T4) untuk fine-tuning Orang 4.

Jangan jalankan di laptop. File ini disalin oleh `python -m phase2.scripts.kaggle_job prepare`
ke folder kernel. Kernel:
1. mengambil commit dasar repo publik dari GitHub (bukan cabang yang bisa bergeser),
2. menerapkan git bundle berisi commit lokal yang belum di-push, lalu checkout commit
   kode yang persis dicatat di job.json,
3. menginstal requirements-train.txt dan menjalankan setiap run secara berurutan;
   satu run gagal tidak menghentikan run berikutnya, statusnya dicatat.
Output: /kaggle/working/outputs/<run_id>/ (unduh dengan `kaggle kernels output`).
"""
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK = Path("/kaggle/working/outputs")
REPO = Path("/tmp/repo")


def sh(*cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def main():
    job_file = Path(sorted(glob.glob("/kaggle/input/**/job.json", recursive=True))[0])
    job = json.loads(job_file.read_text())
    bundle = job_file.parent / job["bundle_file"]
    WORK.mkdir(parents=True, exist_ok=True)

    sh("git", "init", "-q", REPO)
    sh("git", "remote", "add", "origin", job["repo_url"], cwd=REPO)
    sh("git", "fetch", "-q", "--depth", "1", "origin", job["base_commit"], cwd=REPO)
    sh("git", "checkout", "-q", "--detach", "FETCH_HEAD", cwd=REPO)
    sh("git", "fetch", "-q", bundle, job["bundle_ref"], cwd=REPO)
    sh("git", "checkout", "-q", "--detach", job["code_commit"], cwd=REPO)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head != job["code_commit"]:
        raise RuntimeError(f"HEAD {head} != {job['code_commit']}")

    sh(sys.executable, "-m", "pip", "install", "-q", "-r", "phase2/requirements-train.txt", cwd=REPO)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (WORK / "pip_freeze.txt").write_text(freeze)
    subprocess.run(["nvidia-smi"], check=False)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    status = {"job": job, "code_commit": head, "runs": []}
    for item in job["runs"]:
        started = time.time()
        code = subprocess.run([sys.executable, "-m", "phase2.src.train", "--config", item["config"],
                               "--run-id", item["run_id"], "--output-root", str(WORK)],
                              cwd=REPO, env=env).returncode
        status["runs"].append({**item, "returncode": code, "wall_sec": round(time.time() - started, 1)})
        (WORK / "kaggle_job_status.json").write_text(json.dumps(status, indent=2))
    if any(r["returncode"] for r in status["runs"]):
        print("Ada run yang gagal; lihat run.json masing-masing.", flush=True)


if __name__ == "__main__":
    main()
