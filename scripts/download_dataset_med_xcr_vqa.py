#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "X-iZhang/Medical-CXR-VQA"
DEFAULT_OUTPUT_DIR = Path("data/Medical_CXR_VQA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Medical-CXR-VQA dataset from Hugging Face."
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
    return parser.parse_args()


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
