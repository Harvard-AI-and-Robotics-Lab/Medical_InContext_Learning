import argparse
import json
from pathlib import Path

import numpy as np


def load_shard(path: Path):
    with open(path / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embeddings = np.load(path / "global_embeddings.npy").astype(np.float32)
    if embeddings.shape[0] != len(metadata["ids"]):
        raise ValueError(f"Row count mismatch in {path}")
    return metadata, embeddings


def main():
    parser = argparse.ArgumentParser(description="Merge deterministic feature shards into one retrieval directory.")
    parser.add_argument("--shard-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    base_meta = None
    expected_total = None
    for shard_dir in args.shard_dirs:
        meta, emb = load_shard(shard_dir)
        if base_meta is None:
            base_meta = meta
            expected_total = int(meta.get("n_total_samples", 0)) or None
        for key in ["encoder_name", "encoder_version", "preprocessing_hash", "embedding_dim", "image_size", "model_id"]:
            if meta.get(key) != base_meta.get(key):
                raise ValueError(f"Shard metadata mismatch for {key}: {shard_dir}")
        for row_idx, original_idx in enumerate(meta["original_indices"]):
            rows.append((int(original_idx), meta["ids"][row_idx], meta["labels"][row_idx], meta["splits"][row_idx], emb[row_idx]))

    rows.sort(key=lambda row: row[0])
    indices = [row[0] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate original indices across shards")
    if expected_total is not None and indices != list(range(expected_total)):
        missing = sorted(set(range(expected_total)) - set(indices))[:10]
        raise ValueError(f"Merged indices do not cover 0..{expected_total - 1}; first missing={missing}")

    ids = [row[1] for row in rows]
    labels = [row[2] for row in rows]
    splits = [row[3] for row in rows]
    embeddings = np.stack([row[4] for row in rows]).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "global_embeddings.npy", embeddings)
    out_meta = {
        "ids": ids,
        "labels": labels,
        "splits": splits,
        "encoder_name": base_meta.get("encoder_name", ""),
        "encoder_version": base_meta.get("encoder_version", ""),
        "preprocessing_hash": base_meta.get("preprocessing_hash", ""),
        "n_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "image_size": base_meta.get("image_size", 0),
        "model_id": base_meta.get("model_id", ""),
        "spatial_storage_format": "none",
        "spatial_count": 0,
        "spatial_dtype": None,
        "merged_from_shards": [str(p) for p in args.shard_dirs],
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2)
    print(f"Merged {embeddings.shape[0]} embeddings to {args.output_dir}")


if __name__ == "__main__":
    main()
