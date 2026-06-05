import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(1, str(repo_root))

from config import load_config
from datasets import get_dataset
from scripts.extract_features import collect_samples


class CLIPShardDataset(Dataset):
    def __init__(self, shard_pairs, image_size):
        self.shard_pairs = shard_pairs
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def __len__(self):
        return len(self.shard_pairs)

    def __getitem__(self, idx):
        original_idx, sample = self.shard_pairs[idx]
        with Image.open(sample.image_path) as img:
            tensor = self.transform(img.convert("RGB"))
        return {
            "original_idx": int(original_idx),
            "id": sample.id,
            "label": int(getattr(sample, "label", -1)),
            "split": sample.split,
            "pixel_values": tensor,
        }


def collate(batch):
    return {
        "original_indices": [item["original_idx"] for item in batch],
        "ids": [item["id"] for item in batch],
        "labels": [item["label"] for item in batch],
        "splits": [item["split"] for item in batch],
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Fast CLIP global feature extraction for one deterministic shard.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    cfg = load_config(args.config)
    raw_encoder = raw_cfg.get("encoder", {})
    model_id = raw_encoder.get("model_id", "openai/clip-vit-large-patch14")
    image_size = int(raw_encoder.get("image_size", 224))
    batch_size = int(args.batch_size or raw_encoder.get("batch_size", 256))

    dataset_kwargs = {}
    if raw_cfg.get("manifest_csv"):
        dataset_kwargs["manifest_csv"] = raw_cfg["manifest_csv"]
    dataset = get_dataset("chexpert", cfg.data_root, split="all", **dataset_kwargs)
    all_samples = collect_samples(dataset)
    shard_pairs = [(idx, sample) for idx, sample in enumerate(all_samples) if idx % args.num_shards == args.shard_index]
    if not shard_pairs:
        raise RuntimeError(f"No samples selected for shard {args.shard_index}/{args.num_shards}")

    from transformers import CLIPModel
    model = CLIPModel.from_pretrained(model_id).to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    loader = DataLoader(
        CLIPShardDataset(shard_pairs, image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        collate_fn=collate,
    )

    ids, labels, splits, original_indices = [], [], [], []
    embeddings = []
    desc = f"chexpert/clip/shard{args.shard_index}"
    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc):
            pixel_values = batch["pixel_values"].to("cuda", non_blocking=True)
            outputs = model.vision_model(pixel_values=pixel_values, output_hidden_states=False)
            cls = outputs.last_hidden_state[:, 0, :]
            cls = cls.detach().float().cpu().numpy().astype(np.float32, copy=False)
            cls /= np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)
            embeddings.append(cls)
            ids.extend(batch["ids"])
            labels.extend(batch["labels"])
            splits.extend(batch["splits"])
            original_indices.extend(batch["original_indices"])

    embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    out_path = args.output_root / f"shard_{args.shard_index:02d}" / "chexpert" / "clip"
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "global_embeddings.npy", embeddings)
    metadata = {
        "ids": ids,
        "labels": labels,
        "splits": splits,
        "original_indices": original_indices,
        "encoder_name": "clip",
        "encoder_version": "clip-vit-large-patch14-v1",
        "preprocessing_hash": "clip-fast-dataloader-224-v1",
        "n_samples": len(ids),
        "n_total_samples": len(all_samples),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "embedding_dim": int(embeddings.shape[1]),
        "image_size": image_size,
        "model_id": model_id,
        "spatial_storage_format": "none",
        "spatial_count": 0,
        "spatial_dtype": None,
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved CLIP shard {args.shard_index}/{args.num_shards}: {len(ids)} embeddings to {out_path}", flush=True)


if __name__ == "__main__":
    main()
