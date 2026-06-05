#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "flaviagiammarino/vqa-rad"
DEFAULT_OUTPUT_DIR = Path("data/VQA_RAD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the VQA-RAD dataset from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to download the dataset into. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional dataset revision, branch, or commit hash.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Optional Hugging Face token. If omitted, huggingface_hub uses the "
            "cached login token or HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum parallel download workers. Default: 8",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Only download the Hugging Face repo files; do not export images/metadata.",
    )
    return parser.parse_args()


def metadata_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


def split_from_parquet_path(path: Path) -> str:
    return path.name.split("-", 1)[0]


def image_from_value(value: Any) -> tuple[Image.Image, str]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected image column to contain dict values, got {type(value)!r}")
    image_bytes = value.get("bytes")
    if not image_bytes:
        raise ValueError("Image row does not contain embedded bytes")
    image = Image.open(BytesIO(image_bytes))
    source_path = str(value.get("path") or "")
    return image, source_path


def export_images_and_metadata(output_dir: Path) -> None:
    parquet_paths = sorted((output_dir / "data").glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {output_dir / 'data'}")

    rows: list[dict[str, Any]] = []
    jsonl_path = output_dir / "metadata.jsonl"
    csv_path = output_dir / "metadata.csv"
    images_root = output_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for parquet_path in parquet_paths:
            split_name = split_from_parquet_path(parquet_path)
            df = pd.read_parquet(parquet_path)
            if "image" not in df.columns:
                raise ValueError(f"{parquet_path} does not contain an image column")
            split_image_dir = images_root / split_name
            split_image_dir.mkdir(parents=True, exist_ok=True)

            for idx, example in enumerate(df.to_dict("records")):
                image, source_image_path = image_from_value(example["image"])
                image_format = (image.format or "JPEG").lower()
                extension = Path(source_image_path).suffix.lstrip(".").lower()
                if not extension:
                    extension = "jpg" if image_format == "jpeg" else image_format
                if extension == "jpeg":
                    extension = "jpg"
                image_path = split_image_dir / f"{idx:06d}.{extension}"
                image.convert("RGB").save(image_path)

                row = {
                    "split": split_name,
                    "row_index": idx,
                    "image_path": image_path.relative_to(output_dir).as_posix(),
                    "source_image_path": source_image_path,
                }
                for key, value in example.items():
                    if key == "image":
                        continue
                    row[key] = metadata_value(value)

                rows.append(row)
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows")
    print(f"Images: {images_root}")
    print(f"Metadata CSV: {csv_path}")
    print(f"Metadata JSONL: {jsonl_path}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )

    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        token=token,
        max_workers=args.max_workers,
        resume_download=True,
    )

    print(f"Downloaded {args.repo_id} to {local_path}")
    if not args.skip_export:
        export_images_and_metadata(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
