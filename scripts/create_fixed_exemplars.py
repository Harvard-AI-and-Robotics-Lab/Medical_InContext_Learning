#!/usr/bin/env python
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import get_dataset


def main():
    parser = argparse.ArgumentParser(description="Create fixed train-only ICL exemplars for a dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-csv", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    dataset = get_dataset(
        args.dataset,
        args.data_root,
        split="all",
        manifest_csv=args.manifest_csv,
    )
    refs = list(dataset.get_reference_pool())
    if len(refs) < args.k:
        raise ValueError(f"Need at least {args.k} train/reference samples, found {len(refs)}.")

    rng = random.Random(args.seed)
    chosen = rng.sample(sorted(refs, key=lambda sample: str(sample.id)), args.k)
    rows = [
        {
            "id": sample.id,
            "label": int(sample.label),
            "label_name": sample.label_name,
            "split": sample.split,
            "image_path": sample.image_path,
            "metadata": sample.metadata or {},
        }
        for sample in chosen
    ]

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} fixed exemplars to {output_path}")
    for row in rows:
        print(f"{row['id']}\t{row['label_name']}\t{row['split']}")


if __name__ == "__main__":
    main()
