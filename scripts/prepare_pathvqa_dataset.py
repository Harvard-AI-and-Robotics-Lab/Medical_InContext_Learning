#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


SPLIT_TO_INTERNAL = {
    "train": "reference",
    "validation": "validation",
    "test": "test",
}


def normalize_answer(text: str) -> str:
    return str(text or "").strip().lower().strip(". ")


def infer_answer_type(answer: str) -> str:
    return "CLOSED" if normalize_answer(answer) in {"yes", "no"} else "OPEN"


def pick_fixed_exemplars(reference_rows: list[dict], seed: int, k: int) -> list[dict]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for row in reference_rows:
        by_type.setdefault(str(row.get("answer_type", "UNKNOWN")).upper(), []).append(row)
    for rows in by_type.values():
        rng.shuffle(rows)

    picked = []
    for answer_type in ("CLOSED", "OPEN"):
        picked.extend(by_type.get(answer_type, [])[: max(1, k // 2)])
    if len(picked) < k:
        rest = [row for rows in by_type.values() for row in rows if row not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: k - len(picked)])
    return picked[:k]


def save_rgb_jpeg(image, path: Path, quality: int) -> tuple[int, int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = image.size
    mode = image.mode
    if not path.exists():
        image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=True)
    return width, height, mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="flaviagiammarino/path-vqa")
    parser.add_argument("--output-root", type=Path, default=Path("data/vqa/pathvqa"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fixed-k", type=int, default=6)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    samples = []
    split_counts = Counter()
    answer_type_counts = Counter()
    source_modes = Counter()
    widths = []
    heights = []

    for hf_split, internal_split in SPLIT_TO_INTERNAL.items():
        ds = load_dataset(args.repo_id, split=hf_split)
        for row_idx, row in enumerate(ds):
            sample_id = f"pathvqa_{hf_split}_{row_idx:06d}"
            image_rel = Path("images") / hf_split / f"{sample_id}.jpg"
            width, height, mode = save_rgb_jpeg(row["image"], output_root / image_rel, args.jpeg_quality)
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            answer_type = infer_answer_type(answer)
            sample = {
                "id": sample_id,
                "split": internal_split,
                "image_path": str(image_rel),
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "metadata": {
                    "source_dataset": args.repo_id,
                    "source_split": hf_split,
                    "source_index": row_idx,
                    "source_image_mode": mode,
                    "source_width": width,
                    "source_height": height,
                },
            }
            samples.append(sample)
            split_counts[internal_split] += 1
            answer_type_counts[f"{internal_split}:{answer_type}"] += 1
            source_modes[mode] += 1
            widths.append(width)
            heights.append(height)

    manifest = {
        "name": "pathvqa",
        "task_type": "vqa",
        "source": args.repo_id,
        "seed": args.seed,
        "n_samples": len(samples),
        "splits": dict(split_counts),
        "answer_type_counts": dict(answer_type_counts),
        "source_image_modes": dict(source_modes),
        "source_width_min": min(widths) if widths else None,
        "source_width_max": max(widths) if widths else None,
        "source_height_min": min(heights) if heights else None,
        "source_height_max": max(heights) if heights else None,
        "image_note": "Images are saved as RGB JPEG at original pixel dimensions.",
        "samples": samples,
    }
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    fixed = pick_fixed_exemplars([row for row in samples if row["split"] == "reference"], args.seed, args.fixed_k)
    fixed_path = output_root / f"fixed_exemplars_seed{args.seed}.json"
    with fixed_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "answer_type": row["answer_type"],
                }
                for row in fixed
            ],
            f,
            indent=2,
            ensure_ascii=True,
        )

    print(json.dumps({
        "manifest": str(manifest_path),
        "fixed_exemplars": str(fixed_path),
        "n_samples": len(samples),
        "splits": dict(split_counts),
        "answer_type_counts": dict(answer_type_counts),
        "source_image_modes": dict(source_modes),
        "source_width_min": manifest["source_width_min"],
        "source_width_max": manifest["source_width_max"],
        "source_height_min": manifest["source_height_min"],
        "source_height_max": manifest["source_height_max"],
    }, indent=2))


if __name__ == "__main__":
    main()
