import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml
from tqdm import tqdm

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(1, str(repo_root))

from config import load_config
from datasets import get_dataset
from encoders import get_encoder
from scripts.extract_features import collect_samples


def build_encoder_config(cfg, raw_cfg, encoder_name):
    raw_encoder = raw_cfg.get("encoder", {})
    encoder_cfg = {
        "model_id": getattr(cfg.encoder, "model_id", None),
        "image_size": getattr(cfg.encoder, "image_size", 224),
        "device": getattr(cfg.encoder, "device", "cuda"),
        "batch_size": getattr(cfg.encoder, "batch_size", 32),
        "return_spatial": bool(raw_encoder.get("return_spatial", False)),
    }
    if encoder_name == "clip":
        encoder_cfg["model_id"] = raw_encoder.get("model_id", "openai/clip-vit-large-patch14")
        encoder_cfg["image_size"] = int(raw_encoder.get("image_size", 224))
    if "precision" in raw_encoder:
        encoder_cfg["precision"] = raw_encoder["precision"]
    if "torch_dtype" in raw_encoder:
        encoder_cfg["torch_dtype"] = raw_encoder["torch_dtype"]
    return encoder_cfg


def coerce_label(sample):
    label = getattr(sample, "label", -1)
    if isinstance(label, (np.integer,)):
        return int(label)
    if isinstance(label, (np.floating,)):
        return float(label)
    return label


def main():
    parser = argparse.ArgumentParser(description="Extract global features for one deterministic dataset shard.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--encoders", nargs="+", default=None)
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    cfg = load_config(args.config)

    dataset_names = args.datasets or cfg.datasets
    encoder_names = args.encoders or [cfg.encoder.name]
    dataset_kwargs = {}
    manifest_csv = raw_cfg.get("manifest_csv", "")
    if manifest_csv:
        dataset_kwargs["manifest_csv"] = manifest_csv

    for dataset_name in dataset_names:
        dataset = get_dataset(dataset_name, cfg.data_root, split="all", **dataset_kwargs)
        all_samples = collect_samples(dataset)
        shard_pairs = [(idx, sample) for idx, sample in enumerate(all_samples) if idx % args.num_shards == args.shard_index]
        if not shard_pairs:
            raise RuntimeError(f"No samples selected for shard {args.shard_index}/{args.num_shards}")

        for encoder_name in encoder_names:
            encoder_cfg = build_encoder_config(cfg, raw_cfg, encoder_name)
            encoder_kwargs = {
                "model_id": encoder_cfg["model_id"],
                "device": encoder_cfg["device"],
                "image_size": encoder_cfg["image_size"],
                "return_spatial": encoder_cfg["return_spatial"],
            }
            if "precision" in encoder_cfg:
                encoder_kwargs["precision"] = encoder_cfg["precision"]
            if "torch_dtype" in encoder_cfg:
                encoder_kwargs["torch_dtype"] = encoder_cfg["torch_dtype"]
            encoder = get_encoder(name=encoder_name, **encoder_kwargs)

            out_path = args.output_root / f"shard_{args.shard_index:02d}" / dataset_name / encoder_name
            out_path.mkdir(parents=True, exist_ok=True)

            ids, labels, splits, original_indices = [], [], [], []
            global_embeddings = []
            batch_size = int(encoder_cfg.get("batch_size", 32))
            desc = f"{dataset_name}/{encoder_name}/shard{args.shard_index}"
            for start in tqdm(range(0, len(shard_pairs), batch_size), desc=desc):
                batch_pairs = shard_pairs[start:start + batch_size]
                batch_indices = [idx for idx, _sample in batch_pairs]
                batch_samples = [sample for _idx, sample in batch_pairs]
                try:
                    batch_images = [sample.load_image() for sample in batch_samples]
                    batch_outputs = encoder.encode_batch(batch_images, batch_size=batch_size)
                    if len(batch_outputs) != len(batch_samples):
                        raise RuntimeError(f"encoder returned {len(batch_outputs)} outputs for {len(batch_samples)} samples")
                except Exception as exc:
                    print(f"Batch error on {desc} indices {batch_indices[:3]}..{batch_indices[-1]}: {exc}", flush=True)
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    batch_outputs = []
                    kept_samples = []
                    kept_indices = []
                    for idx, sample in zip(batch_indices, batch_samples):
                        try:
                            batch_outputs.append(encoder.encode_image(sample.load_image()))
                            kept_samples.append(sample)
                            kept_indices.append(idx)
                        except Exception as sample_exc:
                            print(f"Error processing {sample.id}: {sample_exc}", flush=True)
                    batch_samples = kept_samples
                    batch_indices = kept_indices

                for original_idx, sample, output in zip(batch_indices, batch_samples, batch_outputs):
                    ids.append(sample.id)
                    labels.append(coerce_label(sample))
                    splits.append(sample.split)
                    original_indices.append(int(original_idx))
                    global_embeddings.append(np.asarray(output.global_embedding, dtype=np.float32))

            embeddings = np.stack(global_embeddings).astype(np.float32)
            np.save(out_path / "global_embeddings.npy", embeddings)
            metadata = {
                "ids": ids,
                "labels": labels,
                "splits": splits,
                "original_indices": original_indices,
                "encoder_name": encoder_name,
                "encoder_version": getattr(encoder, "encoder_version", ""),
                "preprocessing_hash": encoder.preprocessing_hash() if hasattr(encoder, "preprocessing_hash") else "",
                "n_samples": len(ids),
                "n_total_samples": len(all_samples),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                "image_size": int(encoder_cfg.get("image_size", 0)),
                "model_id": encoder_cfg.get("model_id", ""),
                "spatial_storage_format": "none",
                "spatial_count": 0,
                "spatial_dtype": None,
            }
            with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved shard {args.shard_index}/{args.num_shards}: {len(ids)} embeddings to {out_path}", flush=True)


if __name__ == "__main__":
    main()
