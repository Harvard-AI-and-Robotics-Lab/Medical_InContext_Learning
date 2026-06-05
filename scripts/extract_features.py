import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from datasets import get_dataset, CLASSIFICATION_DATASETS, VQA_DATASETS
from encoders import get_encoder


def collect_samples(dataset):
    groups = []
    if hasattr(dataset, "get_reference_pool"):
        groups.append(dataset.get_reference_pool())
    if hasattr(dataset, "get_validation_samples"):
        groups.append(dataset.get_validation_samples())
    if hasattr(dataset, "get_test_samples"):
        groups.append(dataset.get_test_samples())

    collected = []
    seen = set()
    for group in groups:
        for sample in group:
            if sample.id in seen:
                continue
            seen.add(sample.id)
            collected.append(sample)

    if collected:
        return collected
    return list(dataset.samples)


def extract_for_dataset(dataset_name, data_root, encoder_name, encoder_cfg, output_dir, dataset_kwargs=None):
    dataset_kwargs = dataset_kwargs or {}
    dataset = get_dataset(dataset_name, data_root, split="all", **dataset_kwargs)
    all_samples = collect_samples(dataset)

    if len(all_samples) == 0:
        print(f"No samples found for {dataset_name}, skipping.")
        return

    encoder_kwargs = {
        "model_id": encoder_cfg.get("model_id", None),
        "device": encoder_cfg.get("device", "cuda"),
        "image_size": encoder_cfg.get("image_size", 518),
    }
    if "return_spatial" in encoder_cfg:
        encoder_kwargs["return_spatial"] = encoder_cfg["return_spatial"]
    if "precision" in encoder_cfg:
        encoder_kwargs["precision"] = encoder_cfg["precision"]
    if "torch_dtype" in encoder_cfg:
        encoder_kwargs["torch_dtype"] = encoder_cfg["torch_dtype"]
    encoder = get_encoder(name=encoder_name, **encoder_kwargs)

    out_path = Path(output_dir) / dataset_name / encoder_name
    out_path.mkdir(parents=True, exist_ok=True)

    ids = []
    global_embeddings = []
    spatial_features_dir = out_path / "spatial_features"
    have_full_spatial = True
    spatial_count = 0
    spatial_dtype = None
    labels = []
    splits = []
    batch_size = int(encoder_cfg.get("batch_size", 8))

    for start in tqdm(range(0, len(all_samples), batch_size), desc=f"{dataset_name}/{encoder_name}"):
        batch_samples = all_samples[start:start + batch_size]
        try:
            batch_images = [sample.load_image() for sample in batch_samples]
            batch_outputs = encoder.encode_batch(batch_images, batch_size=batch_size)
            for sample, output in zip(batch_samples, batch_outputs):
                ids.append(sample.id)
                global_embeddings.append(output.global_embedding)
                if output.spatial_features is not None:
                    spatial_features_dir.mkdir(parents=True, exist_ok=True)
                    spatial_path = spatial_features_dir / f"{len(ids) - 1}.npy"
                    spatial_array = output.spatial_features.astype(np.float16, copy=False)
                    np.save(spatial_path, spatial_array)
                    spatial_count += 1
                    spatial_dtype = str(spatial_array.dtype)
                else:
                    have_full_spatial = False
                if hasattr(sample, 'label'):
                    labels.append(sample.label)
                else:
                    labels.append(-1)
                splits.append(sample.split)
        except Exception as e:
            print(f"Batch error on {dataset_name}/{encoder_name} samples {[s.id for s in batch_samples]}: {e}")
            for sample in batch_samples:
                try:
                    img = sample.load_image()
                    output = encoder.encode_image(img)
                    ids.append(sample.id)
                    global_embeddings.append(output.global_embedding)
                    if output.spatial_features is not None:
                        spatial_features_dir.mkdir(parents=True, exist_ok=True)
                        spatial_path = spatial_features_dir / f"{len(ids) - 1}.npy"
                        spatial_array = output.spatial_features.astype(np.float16, copy=False)
                        np.save(spatial_path, spatial_array)
                        spatial_count += 1
                        spatial_dtype = str(spatial_array.dtype)
                    else:
                        have_full_spatial = False
                    if hasattr(sample, 'label'):
                        labels.append(sample.label)
                    else:
                        labels.append(-1)
                    splits.append(sample.split)
                except Exception as sample_exc:
                    print(f"Error processing {sample.id}: {sample_exc}")
                    continue

    global_embeddings = np.array(global_embeddings)
    np.save(out_path / "global_embeddings.npy", global_embeddings)

    if spatial_count and not (have_full_spatial and spatial_count == len(ids)):
        print(
            f"Spatial feature export incomplete for {dataset_name}/{encoder_name}: "
            f"only {spatial_count} of {len(ids)} samples produced spatial features."
        )

    metadata = {
        "ids": ids,
        "labels": labels,
        "splits": splits,
        "encoder_name": encoder_name,
        "encoder_version": encoder.encoder_version if hasattr(encoder, 'encoder_version') else "",
        "preprocessing_hash": encoder.preprocessing_hash() if hasattr(encoder, 'preprocessing_hash') else "",
        "n_samples": len(ids),
        "embedding_dim": int(global_embeddings.shape[1]) if len(global_embeddings.shape) > 1 else 0,
        "image_size": int(encoder_cfg.get("image_size", 0)),
        "model_id": encoder_cfg.get("model_id", ""),
        "spatial_storage_format": "per_sample_npy" if spatial_count else "none",
        "spatial_count": spatial_count,
        "spatial_dtype": spatial_dtype,
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(ids)} embeddings to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--encoders", nargs="+", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    cfg = load_config(args.config)

    dataset_names = args.datasets or cfg.datasets
    encoder_names = args.encoders or [cfg.encoder.name]

    encoder_configs = {
        "dinov3": {
            "model_id": cfg.encoder.model_id,
            "image_size": cfg.encoder.image_size,
            "device": cfg.encoder.device,
            "batch_size": cfg.encoder.batch_size,
            "return_spatial": raw_cfg.get("encoder", {}).get("return_spatial", True),
        },
        "clip": {
            "model_id": "openai/clip-vit-large-patch14",
            "image_size": 224,
            "device": cfg.encoder.device,
            "batch_size": cfg.encoder.batch_size,
            "return_spatial": raw_cfg.get("encoder", {}).get("return_spatial", True),
        },
        "biomedclip": {
            "model_id": cfg.encoder.model_id,
            "image_size": 224,
            "device": cfg.encoder.device,
            "batch_size": cfg.encoder.batch_size,
        },
        "openclip": {
            "model_id": cfg.encoder.model_id,
            "image_size": 224,
            "device": cfg.encoder.device,
            "batch_size": cfg.encoder.batch_size,
            "precision": getattr(cfg.encoder, "precision", "fp32"),
        },
        "siglip2": {
            "model_id": cfg.encoder.model_id,
            "image_size": cfg.encoder.image_size,
            "device": cfg.encoder.device,
            "batch_size": cfg.encoder.batch_size,
            "torch_dtype": getattr(cfg.encoder, "torch_dtype", "float16"),
        },
        "mae": {
            "model_id": "facebook/vit-mae-large",
            "image_size": 224,
            "device": cfg.encoder.device,
            "batch_size": cfg.encoder.batch_size,
        },
    }

    output_dir = Path(cfg.features_root) if hasattr(cfg, 'features_root') else Path(cfg.output_root) / "features"
    dataset_kwargs = {}
    manifest_json = raw_cfg.get("manifest_json", "")
    if manifest_json:
        dataset_kwargs["manifest_json"] = manifest_json
    manifest_csv = raw_cfg.get("manifest_csv", "")
    if manifest_csv:
        dataset_kwargs["manifest_csv"] = manifest_csv

    for ds_name in dataset_names:
        for enc_name in encoder_names:
            enc_cfg = encoder_configs.get(enc_name, {"device": cfg.encoder.device})
            extract_for_dataset(ds_name, cfg.data_root, enc_name, enc_cfg, output_dir, dataset_kwargs=dataset_kwargs)


if __name__ == "__main__":
    main()
