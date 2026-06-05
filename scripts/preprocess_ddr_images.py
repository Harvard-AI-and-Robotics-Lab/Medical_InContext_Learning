import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def find_content_bbox(image: Image.Image, threshold: int, margin: int):
    arr = np.asarray(image.convert("RGB"))
    mask = arr.max(axis=2) > threshold
    if not mask.any():
        return (0, 0, image.width, image.height)

    ys, xs = np.where(mask)
    left = max(int(xs.min()) - margin, 0)
    upper = max(int(ys.min()) - margin, 0)
    right = min(int(xs.max()) + margin + 1, image.width)
    lower = min(int(ys.max()) + margin + 1, image.height)
    return (left, upper, right, lower)


def resize_and_pad(image: Image.Image, size: int):
    image = image.convert("RGB")
    width, height = image.size
    scale = size / float(max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas.paste(resized, (left, top))
    return canvas, {
        "resized_width": new_width,
        "resized_height": new_height,
        "pad_left": left,
        "pad_top": top,
        "pad_right": size - new_width - left,
        "pad_bottom": size - new_height - top,
    }


def process_one(args):
    row, data_root, output_root, size, threshold, margin, quality, skip_existing = args
    src = Path(row["image_path"])
    if not src.is_absolute():
        src = Path(data_root) / src

    split = str(row["split"])
    sample_id = str(row["id"])
    dst = Path(output_root) / split / f"{sample_id}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and dst.exists():
        with Image.open(dst) as out_im:
            output_size = out_im.size
        return sample_id, str(dst), {
            "skipped_existing": True,
            "output_width": output_size[0],
            "output_height": output_size[1],
        }

    with Image.open(src) as im:
        original_width, original_height = im.size
        bbox = find_content_bbox(im, threshold=threshold, margin=margin)
        cropped = im.crop(bbox)
        crop_width, crop_height = cropped.size
        out_im, pad_info = resize_and_pad(cropped, size=size)
        out_im.save(dst, format="JPEG", quality=quality, optimize=True)

    info = {
        "skipped_existing": False,
        "source_path": str(src),
        "original_width": original_width,
        "original_height": original_height,
        "crop_left": bbox[0],
        "crop_top": bbox[1],
        "crop_right": bbox[2],
        "crop_bottom": bbox[3],
        "crop_width": crop_width,
        "crop_height": crop_height,
        "output_width": size,
        "output_height": size,
        **pad_info,
    }
    return sample_id, str(dst), info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-root", default="data/processed/ddr_crop_pad_1024")
    parser.add_argument("--output-manifest", default="manifests/ddr_official_split_crop_pad_1024.csv")
    parser.add_argument("--summary-json", default="manifests/ddr_official_split_crop_pad_1024.summary.json")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_csv)
    df = pd.read_csv(manifest_path, dtype={"id": str, "image_name": str})
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            row,
            args.data_root,
            args.output_root,
            args.size,
            args.threshold,
            args.margin,
            args.quality,
            args.skip_existing,
        )
        for row in df.to_dict("records")
    ]

    path_by_id = {}
    info_by_id = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, task) for task in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="preprocess_ddr"):
            sample_id, dst, info = fut.result()
            path_by_id[sample_id] = dst
            info_by_id[sample_id] = info

    processed_df = df.copy()
    processed_df["source_image_path"] = processed_df["image_path"]
    processed_df["image_path"] = processed_df["id"].map(path_by_id)
    processed_df["preprocess"] = f"crop_black_threshold{args.threshold}_margin{args.margin}_resize_long_side_pad_{args.size}"
    processed_df["preprocess_size"] = args.size
    processed_df["preprocess_crop_threshold"] = args.threshold
    processed_df["preprocess_crop_margin"] = args.margin

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_manifest, index=False)

    crop_widths = [v.get("crop_width") for v in info_by_id.values() if v.get("crop_width")]
    crop_heights = [v.get("crop_height") for v in info_by_id.values() if v.get("crop_height")]
    summary = {
        "source_manifest": str(manifest_path),
        "output_manifest": str(output_manifest),
        "output_root": str(output_root),
        "n_images": int(len(processed_df)),
        "size": args.size,
        "threshold": args.threshold,
        "margin": args.margin,
        "quality": args.quality,
        "workers": args.workers,
        "split_counts": processed_df["split"].value_counts().sort_index().to_dict(),
        "label_counts": processed_df["label_name"].value_counts().sort_index().to_dict()
        if "label_name" in processed_df.columns
        else {},
        "crop_width_min_median_max": [
            int(np.min(crop_widths)) if crop_widths else None,
            float(np.median(crop_widths)) if crop_widths else None,
            int(np.max(crop_widths)) if crop_widths else None,
        ],
        "crop_height_min_median_max": [
            int(np.min(crop_heights)) if crop_heights else None,
            float(np.median(crop_heights)) if crop_heights else None,
            int(np.max(crop_heights)) if crop_heights else None,
        ],
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
