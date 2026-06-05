#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def output_path(row: dict, output_root: Path) -> Path:
    split = str(row["split"])
    patient = str(row.get("patient_id", "unknown"))
    study = str(row.get("study_id", "study"))
    view = str(row.get("view", Path(str(row["image_path"])).stem))
    return output_root / split / patient / study / f"{view}.jpg"


def resize_one(args):
    row, output_root, size, quality, skip_existing = args
    src = Path(str(row["image_path"]))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", type=Path, default=Path("manifests/chexpert_official_split.csv"))
    ap.add_argument("--output-root", type=Path, default=Path("data/processed/chexpert_320"))
    ap.add_argument("--output-manifest", type=Path, default=Path("manifests/chexpert_official_split_320.csv"))
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest_csv, dtype={"id": str, "patient_id": str, "study_id": str})
    rows = df.to_dict("records")
    tasks = [(row, args.output_root, args.size, args.quality, args.skip_existing) for row in rows]
    resized = skipped = 0
    paths_by_id = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(resize_one, task) for task in tasks]
        for idx, fut in enumerate(as_completed(futs), start=1):
            path, status = fut.result()
            if status == "skipped":
                skipped += 1
            else:
                resized += 1
            if idx % 5000 == 0 or idx == len(futs):
                print(f"progress {idx}/{len(futs)} resized={resized} skipped={skipped}", flush=True)

    df["image_path"] = [str(output_path(row, args.output_root)) for row in rows]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_manifest, index=False)
    print(f"wrote {len(df)} rows to {args.output_manifest}")
    print(df["split"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
