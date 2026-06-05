import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


EXPECTED_GENERATION = {
    "temperature": 1.0,
    "max_tokens": 512,
    "response_format": "json_object",
    "enable_thinking": False,
    "parallel_requests": 256,
}

CHEXPERT5_EXPECTED_GENERATION = {
    **EXPECTED_GENERATION,
    "max_tokens": 1024,
    "parallel_requests": 128,
}


def expected_generation_for_config(path: Path, cfg: dict):
    datasets = {str(x).lower() for x in cfg.get("datasets", [])}
    prompt_dataset = str(cfg.get("prompt_dataset_name", "")).lower()
    if path.name.startswith("chexpert5_") or prompt_dataset == "chexpert" or "chexpert" in datasets:
        return CHEXPERT5_EXPECTED_GENERATION
    return EXPECTED_GENERATION


def check_manifest(manifest_csv: Path, fixed_exemplars_json: Path):
    df = pd.read_csv(manifest_csv, dtype={"id": str})
    required = {"id", "label", "image_path", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    label_name_col = "class_name" if "class_name" in df.columns else "label_name" if "label_name" in df.columns else None
    if label_name_col is None:
        raise ValueError("Manifest must include either class_name or label_name.")

    ids_by_split = {split: set(g["id"]) for split, g in df.groupby("split")}
    if ids_by_split.get("train", set()) & ids_by_split.get("test", set()):
        raise ValueError("Train/test id overlap detected.")
    if ids_by_split.get("val", set()) & ids_by_split.get("test", set()):
        raise ValueError("Val/test id overlap detected.")
    if "patient_id" in df.columns and df.groupby("patient_id")["split"].nunique().max() > 1:
        raise ValueError("Patient-level leakage detected across splits.")

    fixed = json.load(open(fixed_exemplars_json, "r", encoding="utf-8"))
    fixed_ids = {str(x["id"]) for x in fixed}
    if not fixed_ids.issubset(ids_by_split.get("train", set())):
        bad = sorted(fixed_ids.difference(ids_by_split.get("train", set())))
        raise ValueError(f"Fixed exemplars are not all train samples: {bad}")

    print("Manifest OK")
    print(pd.crosstab(df[label_name_col], df["split"]).to_string())
    print("Fixed exemplars:", ", ".join(sorted(fixed_ids)))


def check_config(path: Path):
    cfg = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
    inf = cfg.get("inference", {})
    if not inf:
        return
    errors = []
    expected_generation = expected_generation_for_config(path, cfg)
    for key, expected in expected_generation.items():
        if inf.get(key) != expected:
            errors.append(f"{key}={inf.get(key)!r}, expected {expected!r}")
    if inf.get("top_p") is not None:
        errors.append(f"top_p={inf.get('top_p')!r}, expected unset/null")
    chat_kwargs = inf.get("chat_template_kwargs") or {}
    if chat_kwargs.get("enable_thinking") is not False:
        errors.append("chat_template_kwargs.enable_thinking must be false")
    if int(cfg.get("k", cfg.get("retrieval", {}).get("k", 6))) != int(cfg.get("retrieval", {}).get("k", 6)):
        errors.append("top-level k and retrieval.k disagree")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    print(f"Config OK: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, default=Path("manifests/lag_manifest.csv"))
    parser.add_argument("--fixed-exemplars-json", type=Path, default=Path("manifests/fixed_exemplars_seed3407.json"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs/final"))
    args = parser.parse_args()

    check_manifest(args.manifest_csv, args.fixed_exemplars_json)
    for path in sorted(args.config_dir.glob("*.yaml")):
        if path.name.startswith("extract_") or path.name.startswith("resnet"):
            continue
        if path.name.startswith("chexpert_"):
            print(f"Skipping legacy CheXpert config: {path}")
            continue
        check_config(path)


if __name__ == "__main__":
    main()
