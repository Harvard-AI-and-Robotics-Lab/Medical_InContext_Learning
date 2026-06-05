import argparse
import json
from pathlib import Path

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), eps)


def main():
    parser = argparse.ArgumentParser(
        description="Convert exported DINOv3 patch-token features into one patch-mean global embedding per image."
    )
    parser.add_argument("--dinov3-features-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--normalize", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.dinov3_features_dir / "metadata.json", "r", encoding="utf-8") as f:
        src_meta = json.load(f)

    n = int(src_meta["n_samples"])
    spatial_dir = args.dinov3_features_dir / "spatial_features"
    patch_mean_embeddings = []

    for idx in range(n):
        spatial_path = spatial_dir / f"{idx}.npy"
        if not spatial_path.exists():
            raise FileNotFoundError(f"Missing spatial feature file: {spatial_path}")
        patch_tokens = np.load(spatial_path).astype(np.float32)
        if patch_tokens.ndim != 2:
            raise ValueError(f"Expected 2D patch-token array for {spatial_path}, got {patch_tokens.shape}")
        pooled = patch_tokens.mean(axis=0)
        if args.normalize:
            pooled = l2_normalize(pooled)
        patch_mean_embeddings.append(pooled.astype(np.float32))

    embeddings = np.stack(patch_mean_embeddings, axis=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "global_embeddings.npy", embeddings)

    metadata = dict(src_meta)
    metadata.update(
        {
            "encoder_name": "dinov3_patch_mean_global",
            "encoder_version": f"{src_meta.get('encoder_version', '')}+patch_mean_global",
            "embedding_dim": int(embeddings.shape[1]),
            "pooling": "mean_over_patch_tokens_then_l2_normalize" if args.normalize else "mean_over_patch_tokens",
            "source_features_dir": str(args.dinov3_features_dir),
        }
    )
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved DINOv3 patch-mean features to {args.output_dir}")


if __name__ == "__main__":
    main()
