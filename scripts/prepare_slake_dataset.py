#!/usr/bin/env python3
import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


SPLIT_TO_INTERNAL = {
    "train": "reference",
    "validation": "validation",
    "test": "test",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "annotations", "samples"):
            if key in data and isinstance(data[key], list):
                return data[key]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported SLAKE JSON structure in {path}")
    return data


def download_file(repo_id: str, filename: str, raw_dir: Path) -> Path:
    src = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename))
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = raw_dir / filename
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return dst


def extract_images(zip_path: Path, output_root: Path) -> None:
    marker = output_root / ".imgs_extracted"
    if marker.exists():
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_root)
    marker.write_text("ok\n", encoding="utf-8")


def find_image(output_root: Path, img_name: str) -> Path | None:
    candidates = [
        output_root / "imgs" / img_name,
        output_root / img_name,
        output_root / "raw" / "imgs" / img_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(output_root.glob(f"**/{img_name}"))
    return matches[0] if matches else None


def pick_fixed_exemplars(reference_rows: list[dict], seed: int, k: int) -> list[dict]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for row in reference_rows:
        answer_type = str(row.get("answer_type", "")).upper() or "UNKNOWN"
        by_type.setdefault(answer_type, []).append(row)
    for rows in by_type.values():
        rng.shuffle(rows)

    picked = []
    preferred = ["CLOSED", "OPEN"]
    per_type = max(1, k // max(1, len([t for t in preferred if t in by_type])))
    for answer_type in preferred:
        picked.extend(by_type.get(answer_type, [])[:per_type])
    if len(picked) < k:
        rest = [row for rows in by_type.values() for row in rows if row not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: k - len(picked)])
    return picked[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="BoKelvin/SLAKE")
    parser.add_argument("--output-root", type=Path, default=Path("data/vqa/slake"))
    parser.add_argument("--lang", default="en", choices=["en", "zh", "all"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fixed-k", type=int, default=6)
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    raw_dir = output_root / "raw"
    output_root.mkdir(parents=True, exist_ok=True)

    json_paths = {
        split: download_file(args.repo_id, f"{split}.json", raw_dir)
        for split in ("train", "validation", "test")
    }
    imgs_zip = download_file(args.repo_id, "imgs.zip", raw_dir)
    extract_images(imgs_zip, output_root)

    samples = []
    missing = []
    stats = {}
    for hf_split, json_path in json_paths.items():
        rows = load_json(json_path)
        for row in rows:
            q_lang = str(row.get("q_lang", ""))
            if args.lang != "all" and q_lang != args.lang:
                continue
            img_name = str(row["img_name"])
            image_path = find_image(output_root, img_name)
            if image_path is None:
                missing.append(img_name)
                continue
            split = SPLIT_TO_INTERNAL[hf_split]
            qid = str(row.get("qid", len(samples)))
            sample_id = f"slake_{q_lang}_{qid}"
            rel_image = image_path.relative_to(output_root)
            answer_type = str(row.get("answer_type", ""))
            metadata = {
                "source_split": hf_split,
                "qid": qid,
                "img_id": str(row.get("img_id", "")),
                "img_name": img_name,
                "q_lang": q_lang,
                "answer_type": answer_type,
                "base_type": row.get("base_type", ""),
                "content_type": row.get("content_type", ""),
                "modality": row.get("modality", ""),
                "location": row.get("location", ""),
                "triple": row.get("triple", ""),
            }
            sample = {
                "id": sample_id,
                "split": split,
                "image_path": str(rel_image),
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
                "answer_type": answer_type,
                "metadata": metadata,
            }
            samples.append(sample)
            stats[split] = stats.get(split, 0) + 1

    if missing and not args.allow_missing_images:
        raise FileNotFoundError(f"Missing {len(missing)} SLAKE images, first examples: {missing[:10]}")

    manifest = {
        "name": "slake",
        "task_type": "vqa",
        "source": args.repo_id,
        "lang": args.lang,
        "seed": args.seed,
        "n_samples": len(samples),
        "splits": stats,
        "samples": samples,
    }
    suffix = args.lang if args.lang != "all" else "all"
    manifest_path = output_root / f"manifest_{suffix}.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    reference_rows = [row for row in samples if row["split"] == "reference"]
    fixed = pick_fixed_exemplars(reference_rows, args.seed, args.fixed_k)
    fixed_path = output_root / f"fixed_exemplars_{suffix}_seed{args.seed}.json"
    with fixed_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "answer_type": row.get("answer_type", ""),
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
        "splits": stats,
        "missing_images": len(missing),
    }, indent=2))


if __name__ == "__main__":
    main()
