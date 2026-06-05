#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


def infer_answer_type(answer: str) -> str:
    value = str(answer or "").strip().lower()
    return "CLOSED" if value in {"yes", "no"} else "OPEN"


def save_image(image, path: Path) -> tuple[int, int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    width, height = image.size
    image.save(path, quality=95)
    return width, height, image.mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="flaviagiammarino/vqa-rad")
    parser.add_argument("--output-root", type=Path, default=Path("data/vqa/vqa_rad"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--fixed-k", type=int, default=6)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_root = args.output_root
    images_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.repo_id)
    train_indices = list(range(len(ds["train"])))
    rng.shuffle(train_indices)
    n_val = max(1, int(round(len(train_indices) * args.validation_ratio)))
    val_indices = set(train_indices[:n_val])

    samples = []
    source_stats = {}
    split_counts = Counter()
    answer_type_counts = Counter()

    for hf_split in ("train", "test"):
        widths = []
        heights = []
        answer_types = Counter()
        for row_idx, row in enumerate(ds[hf_split]):
            split = "test"
            if hf_split == "train":
                split = "validation" if row_idx in val_indices else "reference"
            answer_type = infer_answer_type(row["answer"])
            image_rel = Path("images") / split / f"{hf_split}_{row_idx:06d}.jpg"
            width, height, mode = save_image(row["image"], output_root / image_rel)
            sample = {
                "id": f"vqa_rad_{split}_{row_idx:06d}",
                "split": split,
                "image_path": image_rel.as_posix(),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "answer_type": answer_type,
                "metadata": {
                    "source_dataset": "VQA-RAD",
                    "source_repo": args.repo_id,
                    "source_split": hf_split,
                    "source_index": row_idx,
                    "source_width": width,
                    "source_height": height,
                    "source_image_mode": mode,
                },
            }
            samples.append(sample)
            split_counts[split] += 1
            answer_type_counts[f"{split}:{answer_type}"] += 1
            answer_types[answer_type] += 1
            widths.append(width)
            heights.append(height)
        source_stats[hf_split] = {
            "samples": len(ds[hf_split]),
            "answer_type_counts": dict(answer_types),
            "width_min": min(widths) if widths else None,
            "width_max": max(widths) if widths else None,
            "height_min": min(heights) if heights else None,
            "height_max": max(heights) if heights else None,
        }

    manifest = {
        "name": "vqa_rad",
        "task_type": "vqa",
        "source": args.repo_id,
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "n_samples": len(samples),
        "splits": dict(split_counts),
        "answer_type_counts": dict(answer_type_counts),
        "source_stats": source_stats,
        "image_note": "Images are kept at original pixel dimensions from the Hugging Face dataset export.",
        "samples": samples,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    reference = [row for row in samples if row["split"] == "reference"]
    by_type = {}
    for row in reference:
        by_type.setdefault(row["answer_type"], []).append(row)
    fixed = []
    for key in sorted(by_type):
        bucket = list(by_type[key])
        rng.shuffle(bucket)
        fixed.extend(bucket[:1])
    remaining_ids = {row["id"] for row in fixed}
    remaining = [row for row in reference if row["id"] not in remaining_ids]
    rng.shuffle(remaining)
    fixed.extend(remaining[: max(0, args.fixed_k - len(fixed))])
    fixed_path = output_root / f"fixed_exemplars_seed{args.seed}.json"
    fixed_path.write_text(
        json.dumps(
            [
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "answer_type": row["answer_type"],
                }
                for row in fixed[: args.fixed_k]
            ],
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    summary = {
        "manifest": str(manifest_path),
        "fixed_exemplars": str(fixed_path),
        "n_samples": len(samples),
        "splits": dict(split_counts),
        "answer_type_counts": dict(answer_type_counts),
        "source_stats": source_stats,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
