#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path

import pandas as pd


LABELS = [
    "no_dr",
    "mild_npdr",
    "moderate_npdr",
    "severe_npdr",
    "proliferative_dr",
    "ungradable",
]

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def read_split(root: Path, official_split: str) -> list[dict]:
    txt_path = root / "DR_grading" / f"{official_split}.txt"
    image_dir = root / "DR_grading" / official_split
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Malformed line {txt_path}:{line_no}: {line}")
            image_name, label_raw = parts
            label = int(label_raw)
            if label < 0 or label >= len(LABELS):
                raise ValueError(f"Unexpected label {label} in {txt_path}:{line_no}")
            image_path = image_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image listed in {txt_path}: {image_path}")
            split = SPLIT_MAP[official_split]
            rows.append({
                "id": f"ddr_{split}_{Path(image_name).stem}",
                "image_name": image_name,
                "image_path": str(image_path),
                "split": split,
                "official_split": official_split,
                "label": label,
                "label_name": LABELS[label],
            })
    return rows


def choose_fixed_exemplars(df: pd.DataFrame, seed: int, n: int) -> list[dict]:
    train_df = df[df["split"] == "train"].copy()
    rng = random.Random(seed)
    chosen = []

    # DDR has six labels, so fixed-6 is one deterministic training example per grade.
    for label in range(len(LABELS)):
        candidates = train_df[train_df["label"] == label].sort_values("id").to_dict("records")
        if not candidates:
            raise ValueError(f"No train candidates for label {label} ({LABELS[label]})")
        chosen.append(rng.choice(candidates))

    if len(chosen) > n:
        chosen = chosen[:n]
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/DDR_extracted/DDR-dataset"))
    parser.add_argument("--manifest-csv", type=Path, default=Path("manifests/ddr_official_split.csv"))
    parser.add_argument("--manifest-json", type=Path, default=Path("data/ddr/manifest.json"))
    parser.add_argument("--fixed-exemplars-json", type=Path, default=Path("manifests/ddr_fixed_exemplars_seed3407.json"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--n-fixed", type=int, default=6)
    args = parser.parse_args()

    rows = []
    for official_split in ("train", "valid", "test"):
        rows.extend(read_split(args.data_root, official_split))

    df = pd.DataFrame(rows)
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.manifest_csv, index=False)

    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_json, "w", encoding="utf-8") as f:
        json.dump({"name": "ddr", "labels": LABELS, "samples": rows}, f, indent=2)

    fixed = choose_fixed_exemplars(df, args.seed, args.n_fixed)
    args.fixed_exemplars_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.fixed_exemplars_json, "w", encoding="utf-8") as f:
        json.dump(fixed, f, indent=2)

    print(f"Wrote {len(df)} rows to {args.manifest_csv}")
    print(df.groupby(["split", "label_name"]).size().unstack(fill_value=0))
    print("Fixed exemplars:")
    for row in fixed:
        print(f"  {row['id']} label={row['label']} {row['label_name']}")


if __name__ == "__main__":
    main()
