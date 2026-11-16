import json

from .dataset import PHASE2, FIELDS, collect, read_csv, source_snapshot, write_json
from .prepare_manifest import validate_rows


def validate():
    config = json.loads((PHASE2 / "configs/split_v1.json").read_text(encoding="utf-8"))
    if source_snapshot()["fingerprint"] != config["source_fingerprint"]:
        raise ValueError("Sumber tidak sama dengan snapshot manifest")
    rows = []
    for split in ["train", "validation", "test"]:
        part = read_csv(PHASE2 / f"data/manifests/{split}.csv")
        if any(r["split"] != split for r in part):
            raise ValueError(f"Label split salah dalam {split}.csv")
        rows.extend(part)
    errors = validate_rows(rows, config)
    expected, excluded, *_ = collect()
    actual = {r["sample_id"]: r for r in rows}
    if set(actual) != {r["sample_id"] for r in expected}:
        errors.append("Ada ujaran hilang/ditambah dari kebijakan data")
    for r in expected:
        if r["sample_id"] in actual and any(actual[r["sample_id"]][k] != r[k] for k in FIELDS if k != "split"):
            errors.append(f"Isi manifest tidak sama dengan sumber: {r['sample_id']}")
    saved_excluded = read_csv(PHASE2 / "data/manifests/excluded.csv")
    if sorted(saved_excluded, key=lambda r: r["sample_id"]) != sorted(excluded, key=lambda r: r["sample_id"]):
        errors.append("Daftar pengecualian tidak sesuai sumber")
    if errors:
        raise ValueError("\n".join(errors))
    return {"valid": True, "accepted": len(rows), "excluded": len(excluded),
            "speaker_overlap": False, "pcm_duplicate": False}


if __name__ == "__main__":
    result = validate()
    write_json(PHASE2 / "data/validation_report.json", result)
    print(json.dumps(result, indent=2))
