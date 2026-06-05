#!/usr/bin/env python3
import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ZENODO_BASE = "https://zenodo.org/api/records/10499039/files"
FILES = [
    "README-VQA-Med-2019-Data.txt",
    "ImageClef-2019-VQA-Med-Training.zip",
    "ImageClef-2019-VQA-Med-Validation.zip",
    "VQAMed2019Test.zip",
]

CATEGORY_FILES = {
    "modality": "C1_Modality",
    "plane": "C2_Plane",
    "organ": "C3_Organ",
    "abnormality": "C4_Abnormality",
}

SPLIT_INFO = {
    "train": {
        "internal_split": "reference",
        "root": "ImageClef-2019-VQA-Med-Training",
        "qa_file": "All_QA_Pairs_train.txt",
        "image_dir": "Train_images",
        "category_suffix": "train",
    },
    "validation": {
        "internal_split": "validation",
        "root": "ImageClef-2019-VQA-Med-Validation",
        "qa_file": "All_QA_Pairs_val.txt",
        "image_dir": "Val_images",
        "category_suffix": "val",
    },
}


def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as response, dst.open("wb") as f:
        shutil.copyfileobj(response, f)


def extract_zip(zip_path: Path, output_dir: Path):
    marker = output_dir / f".{zip_path.stem}.extracted"
    if marker.exists():
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    marker.write_text("ok\n", encoding="utf-8")


def read_pipe_rows(path: Path, expected_fields: int | None = None) -> list[list[str]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split("|")]
            if expected_fields and len(parts) != expected_fields:
                raise ValueError(f"Expected {expected_fields} pipe fields in {path}, got {len(parts)}: {line[:120]}")
            rows.append(parts)
    return rows


def load_category_map(split_root: Path, suffix: str) -> dict[tuple[str, str, str], str]:
    out = {}
    category_root = split_root / "QAPairsByCategory"
    for category, prefix in CATEGORY_FILES.items():
        path = category_root / f"{prefix}_{suffix}.txt"
        if not path.exists():
            continue
        for image_id, question, answer in read_pipe_rows(path, expected_fields=3):
            out[(image_id, question, answer)] = category
    return out


def image_stats(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        image.verify()
    return width, height, mode


def add_split_samples(samples: list[dict], stats: dict, extracted_root: Path, source_split: str):
    info = SPLIT_INFO[source_split]
    split_root = extracted_root / info["root"]
    category_map = load_category_map(split_root, info["category_suffix"])
    qa_rows = read_pipe_rows(split_root / info["qa_file"], expected_fields=3)
    missing_images = []
    category_counts = Counter()
    widths = []
    heights = []
    modes = Counter()

    for row_idx, (image_id, question, answer) in enumerate(qa_rows):
        image_path = split_root / info["image_dir"] / f"{image_id}.jpg"
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue
        category = category_map.get((image_id, question, answer), "unknown")
        width, height, mode = image_stats(image_path)
        rel_image = image_path.relative_to(extracted_root.parent)
        sample_id = f"vqamed2019_{source_split}_{row_idx:05d}_{image_id}"
        samples.append(
            {
                "id": sample_id,
                "split": info["internal_split"],
                "image_path": str(rel_image),
                "question": question,
                "answer": answer,
                "answer_type": category,
                "metadata": {
                    "source_dataset": "ImageCLEF VQA-Med 2019",
                    "source_split": source_split,
                    "source_index": row_idx,
                    "image_id": image_id,
                    "question_category": category,
                    "source_width": width,
                    "source_height": height,
                    "source_image_mode": mode,
                },
            }
        )
        category_counts[category] += 1
        widths.append(width)
        heights.append(height)
        modes[mode] += 1

    stats[source_split] = {
        "qa_rows": len(qa_rows),
        "samples": sum(1 for row in samples if row["metadata"]["source_split"] == source_split),
        "unique_images": len({row[0] for row in qa_rows}),
        "missing_images": len(missing_images),
        "category_counts": dict(category_counts),
        "image_modes": dict(modes),
        "width_min": min(widths) if widths else None,
        "width_max": max(widths) if widths else None,
        "height_min": min(heights) if heights else None,
        "height_max": max(heights) if heights else None,
    }
    return missing_images


def add_test_samples(samples: list[dict], stats: dict, extracted_root: Path):
    split_root = extracted_root / "VQAMed2019Test"
    image_root = split_root / "VQAMed2019_Test_Images"
    nested = image_root / "VQAMed2019_Test_Images"
    if nested.exists():
        image_root = nested
    ref_path = split_root / "VQAMed2019_Test_Questions_w_Ref_Answers.txt"
    rows = read_pipe_rows(ref_path, expected_fields=4)
    missing_images = []
    category_counts = Counter()
    widths = []
    heights = []
    modes = Counter()

    for row_idx, (image_id, category, question, answer) in enumerate(rows):
        image_path = image_root / f"{image_id}.jpg"
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue
        width, height, mode = image_stats(image_path)
        rel_image = image_path.relative_to(extracted_root.parent)
        samples.append(
            {
                "id": f"vqamed2019_test_{row_idx:05d}_{image_id}",
                "split": "test",
                "image_path": str(rel_image),
                "question": question,
                "answer": answer,
                "answer_type": category,
                "metadata": {
                    "source_dataset": "ImageCLEF VQA-Med 2019",
                    "source_split": "test",
                    "source_index": row_idx,
                    "image_id": image_id,
                    "question_category": category,
                    "source_width": width,
                    "source_height": height,
                    "source_image_mode": mode,
                },
            }
        )
        category_counts[category] += 1
        widths.append(width)
        heights.append(height)
        modes[mode] += 1

    stats["test"] = {
        "qa_rows": len(rows),
        "samples": sum(1 for row in samples if row["metadata"]["source_split"] == "test"),
        "unique_images": len({row[0] for row in rows}),
        "missing_images": len(missing_images),
        "category_counts": dict(category_counts),
        "image_modes": dict(modes),
        "width_min": min(widths) if widths else None,
        "width_max": max(widths) if widths else None,
        "height_min": min(heights) if heights else None,
        "height_max": max(heights) if heights else None,
    }
    return missing_images


def pick_fixed_exemplars(reference_rows: list[dict], seed: int, k: int) -> list[dict]:
    rng = random.Random(seed)
    by_category = defaultdict(list)
    for row in reference_rows:
        by_category[str(row.get("answer_type") or "unknown")].append(row)
    for rows in by_category.values():
        rng.shuffle(rows)

    picked = []
    for category in ("modality", "plane", "organ", "abnormality"):
        if by_category.get(category):
            picked.append(by_category[category][0])
    rest = [row for rows in by_category.values() for row in rows if row not in picked]
    rng.shuffle(rest)
    picked.extend(rest[: max(0, k - len(picked))])
    return picked[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/vqa/vqamed2019"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fixed-k", type=int, default=6)
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    raw_dir = output_root / "raw"
    extracted_root = output_root / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_root.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        download(f"{ZENODO_BASE}/{filename}/content", raw_dir / filename)
    for zip_name in FILES:
        if not zip_name.endswith(".zip"):
            continue
        extract_zip(raw_dir / zip_name, extracted_root)
    test_images_zip = extracted_root / "VQAMed2019Test" / "VQAMed2019_Test_Images.zip"
    if test_images_zip.exists():
        extract_zip(test_images_zip, extracted_root / "VQAMed2019Test" / "VQAMed2019_Test_Images")

    samples = []
    stats = {}
    missing = []
    for source_split in ("train", "validation"):
        missing.extend(add_split_samples(samples, stats, extracted_root, source_split))
    missing.extend(add_test_samples(samples, stats, extracted_root))

    if missing and not args.allow_missing_images:
        raise FileNotFoundError(f"Missing {len(missing)} images, first examples: {missing[:10]}")

    split_counts = Counter(row["split"] for row in samples)
    answer_type_counts = Counter(f"{row['split']}:{row['answer_type']}" for row in samples)
    manifest = {
        "name": "vqamed2019",
        "task_type": "vqa",
        "source": "VQA-Med @ ImageCLEF 2019",
        "zenodo_record": "https://zenodo.org/records/10499039",
        "license": "CC-BY-4.0",
        "seed": args.seed,
        "n_samples": len(samples),
        "splits": dict(split_counts),
        "answer_type_counts": dict(answer_type_counts),
        "source_stats": stats,
        "image_note": "Images are kept at original pixel dimensions from the official release.",
        "samples": samples,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    fixed = pick_fixed_exemplars([row for row in samples if row["split"] == "reference"], args.seed, args.fixed_k)
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
                for row in fixed
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
        "source_stats": stats,
        "missing_images": len(missing),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
