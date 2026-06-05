import argparse
import json
from pathlib import Path

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def load_feature_dir(path: Path):
    with open(path / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embeddings = np.load(path / "global_embeddings.npy").astype(np.float32)
    return metadata, embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Build equal-weight fused retrieval features from two global embedding directories."
    )
    parser.add_argument("--feature-a", required=True, type=Path)
    parser.add_argument("--feature-b", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weight-a", type=float, default=0.5)
    parser.add_argument("--weight-b", type=float, default=0.5)
    parser.add_argument("--encoder-name", default="clip_dinov3cls05")
    parser.add_argument(
        "--score-definition",
        default="cosine(concat(sqrt(0.5)*L2(A), sqrt(0.5)*L2(B))) == 0.5*A_cosine + 0.5*B_cosine",
    )
    args = parser.parse_args()

    meta_a, emb_a = load_feature_dir(args.feature_a)
    meta_b, emb_b = load_feature_dir(args.feature_b)

    if meta_a["ids"] != meta_b["ids"]:
        raise ValueError("Feature directories have different sample id order.")
    if meta_a["labels"] != meta_b["labels"]:
        raise ValueError("Feature directories have different labels.")
    if meta_a["splits"] != meta_b["splits"]:
        raise ValueError("Feature directories have different splits.")

    wa = float(args.weight_a)
    wb = float(args.weight_b)
    if wa <= 0 or wb <= 0:
        raise ValueError("Fusion weights must be positive.")
    total = wa + wb
    wa /= total
    wb /= total

    fused = np.concatenate(
        [np.sqrt(wa) * l2_normalize(emb_a), np.sqrt(wb) * l2_normalize(emb_b)],
        axis=1,
    ).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "global_embeddings.npy", fused)

    metadata = {
        "ids": meta_a["ids"],
        "labels": meta_a["labels"],
        "splits": meta_a["splits"],
        "encoder_name": args.encoder_name,
        "encoder_version": f"{meta_a.get('encoder_version', '')}+{meta_b.get('encoder_version', '')}+weighted-fusion",
        "preprocessing_hash": f"a:{meta_a.get('preprocessing_hash', '')}|b:{meta_b.get('preprocessing_hash', '')}",
        "n_samples": int(fused.shape[0]),
        "embedding_dim": int(fused.shape[1]),
        "image_size": {
            "a": meta_a.get("image_size", ""),
            "b": meta_b.get("image_size", ""),
        },
        "model_id": {
            "a": meta_a.get("model_id", ""),
            "b": meta_b.get("model_id", ""),
        },
        "weight_a": wa,
        "weight_b": wb,
        "feature_a": str(args.feature_a),
        "feature_b": str(args.feature_b),
        "score_definition": args.score_definition,
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved fused features to {args.output_dir}")


if __name__ == "__main__":
    main()
