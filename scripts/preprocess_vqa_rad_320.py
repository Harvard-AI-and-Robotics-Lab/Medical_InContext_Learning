#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def output_path(row: dict, output_root: Path) -> Path:
    split = str(row["split"])
    sample_id = str(row["id"])
    return output_root / split / f"{sample_id}.jpg"


def resize_one(args):
    row, data_root, output_root, size, quality, skip_existing = args
    src = Path(str(row["image_path"]))
    if not src.is_absolute():
        src = Path(data_root) / src
    dst = output_path(row, output_root)
    if skip_existing and dst.exists():
        try:
            with Image.open(dst) as im:
                if im.size == (size, size):
                    return str(dst), "skipped"
        except Exception:
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = im.resize((size, size), Image.Resampling.BICUBIC)
        im.save(dst, format="JPEG", quality=quality, optimize=True)
    return str(dst), "resized"


def main() -> int:
    ap = argparse.ArgumentParser(description="Resize VQA-RAD images to an exact square JPEG size without cropping.")
    ap.add_argument("--manifest-csv", type=Path, default=Path("manifests/vqa_rad_official_split.csv"))
    ap.add_argument("--data-root", type=Path, default=Path("."))
    ap.add_argument("--output-root", type=Path, default=Path("data/processed/vqa_rad_320"))
    ap.add_argument("--output-manifest", type=Path, default=Path("manifests/vqa_rad_official_split_320.csv"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest_csv, dtype={"id": str})
    rows = df.to_dict("records")
    tasks = [(row, args.data_root, args.output_root, args.size, args.quality, args.skip_existing) for row in rows]

    resized = skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(resize_one, task) for task in tasks]
        for idx, fut in enumerate(as_completed(futs), start=1):
            _path, status = fut.result()
            if status == "skipped":
                skipped += 1
            else:
                resized += 1
            if idx % 500 == 0 or idx == len(futs):
                print(f"progress {idx}/{len(futs)} resized={resized} skipped={skipped}", flush=True)

    out_df = df.copy()
    out_df["source_image_path"] = out_df["image_path"]
    out_df["image_path"] = [str(output_path(row, args.output_root)) for row in rows]
    out_df["preprocess"] = f"resize_bicubic_{args.size}"
    out_df["preprocess_size"] = args.size

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_manifest, index=False)
    print(f"wrote {len(out_df)} rows to {args.output_manifest}")
    print(out_df["split"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
