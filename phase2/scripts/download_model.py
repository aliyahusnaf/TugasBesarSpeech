import argparse
from pathlib import PurePath

from phase2.scripts.dataset import PHASE2, ROOT, digest
from phase2.src.model import load_config

DRIVE_FOLDER = "https://drive.google.com/drive/folders/10jwfv_vFGCfXh0vtOfake7wC8Flz0Sqq"
FINAL_CONFIG = PHASE2 / "configs/experiments/finetuned_ft_lr1e-5_001_beam5.json"


def list_drive_files(run_name):
    import gdown

    files = gdown.download_folder(DRIVE_FOLDER, skip_download=True, quiet=True)
    wanted = []
    for item in files:
        parts = PurePath(item.path.replace("\\", "/")).parts
        if parts[:2] == (run_name, "ct2"):
            wanted.append((item.id, parts[-1]))
    if not wanted:
        raise FileNotFoundError(f"Folder {run_name}/ct2 tidak ada di Google Drive")
    return wanted


def download(config_path, force=False):
    import gdown

    config = load_config(config_path)
    if "model_path" not in config:
        print("Config ini memakai model pretrained dari Hugging Face, tidak perlu diunduh.")
        return
    target = ROOT / config["model_path"]
    model_file = target / "model.bin"
    if model_file.is_file() and not force and digest(model_file) == config["model_sha256"]:
        print(f"Model sudah ada di {target}")
        return

    run_name = config["training_run"]
    target.mkdir(parents=True, exist_ok=True)
    for file_id, name in list_drive_files(run_name):
        print(f"Mengunduh {run_name}/ct2/{name}")
        gdown.download(id=file_id, output=str(target / name), quiet=False)

    if digest(model_file) != config["model_sha256"]:
        raise ValueError(f"Hash {model_file} tidak sama dengan model_sha256 di config")
    print(f"Model tersimpan di {target}")


def main():
    parser = argparse.ArgumentParser(description="Unduh checkpoint fine-tuned dari Google Drive")
    parser.add_argument("--config", default=str(FINAL_CONFIG))
    parser.add_argument("--force", action="store_true", help="unduh ulang walaupun model sudah ada")
    args = parser.parse_args()
    download(args.config, args.force)


if __name__ == "__main__":
    main()
