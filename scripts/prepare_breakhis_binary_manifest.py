#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path

import pandas as pd


BINARY_LABELS = ["benign", "malignant"]


def build_binary_manifest(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, dtype={"id": str, "patient_id": str, "raw_case_id": str})
    required = {"id", "patient_id", "image_path", "split", "binary_label", "label", "label_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    out = df.copy()
    out["subtype_label"] = out["label"].astype(int)
    out["subtype_label_name"] = out["label_name"].astype(str)
    out["label_name"] = out["binary_label"].astype(str)
    out["label"] = out["label_name"].map({"benign": 0, "malignant": 1})
    if out["label"].isna().any():
        bad = sorted(out.loc[out["label"].isna(), "label_name"].astype(str).unique())
        raise ValueError(f"Unexpected binary labels: {bad}")
    out["label"] = out["label"].astype(int)
    return out


def choose_fixed_exemplars(df: pd.DataFrame, seed: int, n_per_class: int) -> list[dict]:
    rng = random.Random(seed)
    train = df[df["split"] == "train"].copy()
    chosen = []
    for label_idx, label_name in enumerate(BINARY_LABELS):
        candidates = train[train["label"] == label_idx].sort_values("id").to_dict("records")
        if len(candidates) < n_per_class:
            raise ValueError(f"Only {len(candidates)} train candidates for {label_name}; need {n_per_class}")
        chosen.extend(rng.sample(candidates, n_per_class))
    return chosen


def summarize(df: pd.DataFrame) -> dict:
    patient_splits = df.groupby("patient_id")["split"].nunique()
    leakage_patients = patient_splits[patient_splits > 1].index.astype(str).tolist()
    return {
        "n_images": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()},
        "patient_split_counts": {
            str(split): int(df[df["split"] == split]["patient_id"].nunique())
            for split in ["train", "val", "test"]
        },
        "label_counts": {
            split: {
                label: int(count)
                for label, count in part["label_name"].value_counts().sort_index().items()
            }
            for split, part in df.groupby("split")
        },
        "patient_leakage_n": int(len(leakage_patients)),
        "patient_leakage_ids": leakage_patients[:20],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=Path("manifests/breakhis_patient_split_seed3407.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("manifests/breakhis_binary_patient_split_seed3407.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("manifests/breakhis_binary_patient_split_seed3407.summary.json"))
    parser.add_argument("--fixed-exemplars-json", type=Path, default=Path("manifests/breakhis_binary_fixed_exemplars_seed3407.json"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--n-per-class", type=int, default=3)
    args = parser.parse_args()

    df = build_binary_manifest(args.input_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    summary = summarize(df)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fixed = choose_fixed_exemplars(df, args.seed, args.n_per_class)
    args.fixed_exemplars_json.parent.mkdir(parents=True, exist_ok=True)
    args.fixed_exemplars_json.write_text(json.dumps(fixed, indent=2), encoding="utf-8")

    print(f"Wrote {len(df)} rows to {args.output_csv}")
    print(json.dumps(summary, indent=2))
    print("Fixed exemplars:")
    for row in fixed:
        print(f"  {row['id']} label={row['label_name']} subtype={row.get('subtype_label_name', '')}")


if __name__ == "__main__":
    main()
