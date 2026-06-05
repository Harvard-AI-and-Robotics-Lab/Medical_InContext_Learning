#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def clean_id(split: str, row_index: int, image_path: str) -> str:
    stem = Path(str(image_path)).stem
    return f"vqa_rad_{split}_{int(row_index):06d}_{stem}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare a VQA-RAD CSV manifest from exported metadata.")
    ap.add_argument("--metadata-csv", type=Path, default=Path("data/VQA_RAD/metadata.csv"))
    ap.add_argument("--data-root", type=Path, default=Path("data/VQA_RAD"))
    ap.add_argument("--output", type=Path, default=Path("manifests/vqa_rad_official_split.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.metadata_csv)
    required = {"split", "row_index", "image_path", "question", "answer"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {args.metadata_csv}: {missing}")

    rows = []
    for row in df.to_dict("records"):
        split = str(row["split"])
        image_rel = str(row["image_path"])
        image_path = args.data_root / image_rel
        if not image_path.exists():
            raise FileNotFoundError(f"Missing VQA-RAD image: {image_path}")
        row_index = int(row["row_index"])
        rows.append(
            {
                "id": clean_id(split, row_index, image_rel),
                "split": split,
                "image_path": str(image_path),
                "source_image_path": str(row.get("source_image_path", "")),
                "row_index": row_index,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "question_type": str(row.get("question_type", "")) if "question_type" in row else "",
            }
        )

    out = pd.DataFrame(rows).sort_values(["split", "row_index", "id"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} rows to {args.output}")
    print(out["split"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
