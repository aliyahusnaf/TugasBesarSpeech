"""Bandingkan run validation terhadap satu run acuan dengan paired bootstrap."""
import argparse
import json
import random

from phase2.scripts.dataset import PHASE2, read_csv, write_json
from phase2.src import selection_metrics

RUNS = PHASE2 / "results/experiments"


def edits(run_id):
    rows = read_csv(RUNS / run_id / "predictions.csv")
    out = {}
    for r in rows:
        s = selection_metrics.utterance_scores(r["reference"], r["prediction"])
        out[r["sample_id"]] = (s["word_edits"], s["ref_words"], s["char_edits"], s["ref_chars"])
    return out


def wer(table, ids):
    return sum(table[i][0] for i in ids) / sum(table[i][1] for i in ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True, help="run_id acuan di results/experiments")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    complete = sorted(p.parent.name for p in RUNS.glob("*/predictions.csv")
                      if json.loads((p.parent / "run.json").read_text(encoding="utf-8"))["status"] == "complete")
    tables = {run_id: edits(run_id) for run_id in complete}
    ref = tables[args.reference]
    ids = sorted(ref)
    speakers = sorted({i.split("_")[0] for i in ids})
    rng = random.Random(args.seed)
    samples = [[rng.choice(ids) for _ in ids] for _ in range(args.resamples)]
    result = {"reference": args.reference, "utterances": len(ids), "resamples": args.resamples,
              "seed": args.seed, "scoring_policy": selection_metrics.SCORING_POLICY,
              "note": "Paired bootstrap per ujaran; delta = WER run - WER acuan. Validation saja.",
              "runs": {}}
    for run_id, table in tables.items():
        if sorted(table) != ids:
            raise SystemExit(f"{run_id}: sample_id tidak sama dengan acuan")
        deltas = sorted(wer(table, s) - wer(ref, s) for s in samples)
        lo, hi = deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas)) - 1]
        result["runs"][run_id] = {
            "wer": round(wer(table, ids), 4),
            "delta_wer_vs_reference": round(wer(table, ids) - wer(ref, ids), 4),
            "delta_ci95": [round(lo, 4), round(hi, 4)],
            "share_resamples_worse_than_reference": round(sum(d > 0 for d in deltas) / len(deltas), 4),
            "wer_by_speaker": {sp: round(wer(table, [i for i in ids if i.startswith(sp + "_")]), 4)
                               for sp in speakers}}
    write_json(RUNS / "comparison.json", result)
    print(f"{'run_id':40} {'WER':>7} {'delta':>7} {'CI95':>17} " + " ".join(f"{s:>6}" for s in speakers))
    for run_id, r in sorted(result["runs"].items(), key=lambda kv: kv[1]["wer"]):
        ci = f"[{r['delta_ci95'][0]:+.3f},{r['delta_ci95'][1]:+.3f}]"
        print(f"{run_id:40} {r['wer']:7.4f} {r['delta_wer_vs_reference']:+7.4f} {ci:>17} "
              + " ".join(f"{v:6.3f}" for v in r["wer_by_speaker"].values()))


if __name__ == "__main__":
    main()
