"""Gambar kurva training (WER dan loss validation per epoch) untuk laporan Orang 4.

Dua panel terpisah (bukan dual axis): WER validation dan loss validation per epoch
untuk setiap run fine-tuning. Epoch 0 (model pretrained sebelum fine-tuning) tidak
diplot agar skala tidak memampatkan perbedaan antar-run; nilainya ditulis di subjudul.
Angka lengkap ada di history.csv tiap run dan results/training_summary.csv.
"""
import json
from pathlib import Path

from phase2.scripts.dataset import PHASE2, read_csv

# Palet kategorikal tervalidasi (urutan tetap; warna mengikuti run, bukan peringkat).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OUT = PHASE2 / "report/figures/training_curves.png"


def label(run):
    c = run["config"]
    return f"lr {c['learning_rate']:g}" + (" + freeze encoder" if c["freeze_encoder"] else "")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    job = json.loads((PHASE2 / "configs/train/kaggle_job_v1.json").read_text(encoding="utf-8"))
    runs = []
    for item in job["runs"]:  # urutan job = urutan warna tetap
        folder = PHASE2 / "results/training" / item["run_id"]
        if (folder / "history.csv").exists():
            runs.append((json.loads((folder / "run.json").read_text(encoding="utf-8")),
                         read_csv(folder / "history.csv")))
    if not runs:
        raise SystemExit("Belum ada history.csv di phase2/results/training/")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    panels = [("val_wer", "WER validation (%, greedy, ternormalisasi)", 100, "{:.1f}%"),
              ("val_loss", "Loss validation (cross-entropy)", 1, "{:.2f}")]
    epoch0 = runs[0][1][0]  # epoch 0 identik untuk semua run (model dasar sama)
    for ax, (key, title, scale, fmt) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for color, (run, history) in zip(SERIES, runs):
            kept = [h for h in history if int(h["epoch"]) > 0]
            epochs = [int(h["epoch"]) for h in kept]
            values = [float(h[key]) * scale for h in kept]
            ax.plot(epochs, values, color=color, linewidth=2, solid_capstyle="round",
                    solid_joinstyle="round", label=label(run))
            best = run.get("best", {}).get("epoch")
            if best is not None:
                i = epochs.index(best)
                ax.plot(epochs[i], values[i], "o", markersize=8, color=color,
                        markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.set_title(title, loc="left", color=TEXT, fontsize=11, pad=18)
        ax.text(0, 1.02, "Epoch 0 (pretrained, sebelum fine-tuning): "
                + fmt.format(float(epoch0[key]) * scale), transform=ax.transAxes,
                color=MUTED, fontsize=9)
        ax.set_xlabel("Epoch fine-tuning")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(runs), frameon=False,
               bbox_to_anchor=(0.5, 1.02), labelcolor=TEXT)
    fig.text(0.01, -0.02, "Titik = epoch terpilih (WER validation terendah). Data lengkap: "
             "results/training/<run_id>/history.csv.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(OUT.relative_to(PHASE2.parent))


if __name__ == "__main__":
    main()
