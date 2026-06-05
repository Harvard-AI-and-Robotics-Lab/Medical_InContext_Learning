#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import get_dataset
from inference import OutputParser
from prompting import get_prompter


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen36", "gemma4", "medgemma27b")
CONFIG_STEMS = ("noicl_fixed6_random6", "clip_top6", "dinov3_cls_top6", "clip_dinov3cls_top6")


def expect(cond: bool, message: str):
    if not cond:
        raise AssertionError(message)


def load_yaml(path: Path) -> dict:
    expect(path.exists(), f"missing config {path.relative_to(ROOT)}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_configs():
    for model in MODELS:
        for stem in CONFIG_STEMS:
            path = ROOT / "configs" / "final" / f"slake_{model}_{stem}.yaml"
            cfg = load_yaml(path)
            inf = cfg.get("inference", {})
            expect(cfg.get("data_root") == "data/vqa", f"{path.name}: data_root must be data/vqa")
            expect(cfg.get("datasets") == ["slake"], f"{path.name}: datasets must be [slake]")
            expect(cfg.get("k") == 6, f"{path.name}: k must be 6")
            expect(inf.get("temperature") == 1.0, f"{path.name}: temperature must be 1.0")
            expect(inf.get("top_p") is None, f"{path.name}: top_p must be null")
            expect(inf.get("max_tokens") == 512, f"{path.name}: max_tokens must be 512")
            expect(inf.get("response_format") == "json_object", f"{path.name}: response_format must be json_object")
            expect(inf.get("image_max_side") is None, f"{path.name}: image_max_side must be null for original image bytes")
            if stem == "noicl_fixed6_random6":
                expect(cfg.get("methods") == ["zero_shot", "fixed_random_6", "random_icl"], f"{path.name}: bad noicl methods")
                expect("fixed_exemplars_json" in cfg, f"{path.name}: fixed_exemplars_json missing")
            elif stem == "clip_dinov3cls_top6":
                expect(cfg.get("methods") == ["rg_icl_dual_global_similarity"], f"{path.name}: bad fused method")
            else:
                expect(cfg.get("methods") == ["rg_icl_global_similarity"], f"{path.name}: bad retrieval method")

    for name in ("slake_extract_clip_global.yaml", "slake_extract_dinov3_global.yaml"):
        cfg = load_yaml(ROOT / "configs" / "final" / name)
        expect(cfg.get("datasets") == ["slake"], f"{name}: datasets must be [slake]")


def validate_data():
    manifest = ROOT / "data" / "vqa" / "slake" / "manifest_en.json"
    fixed = ROOT / "data" / "vqa" / "slake" / "fixed_exemplars_en_seed3407.json"
    expect(manifest.exists(), "SLAKE manifest_en.json missing; run scripts/prepare_slake_dataset.py")
    expect(fixed.exists(), "SLAKE fixed exemplars missing; run scripts/prepare_slake_dataset.py")

    data = json.loads(manifest.read_text(encoding="utf-8"))
    splits = data.get("splits", {})
    expect(splits.get("reference", 0) > 0, "reference split is empty")
    expect(splits.get("test", 0) > 0, "test split is empty")

    dataset = get_dataset(
        "slake",
        str(ROOT / "data" / "vqa"),
        split="all",
        manifest_json=str(manifest),
        lang="en",
    )
    expect(len(dataset.get_reference_pool()) == splits.get("reference"), "reference count mismatch")
    expect(len(dataset.get_test_samples()) == splits.get("test"), "test count mismatch")
    for sample in dataset.samples[:20]:
        expect(Path(sample.image_path).exists(), f"missing image for {sample.id}: {sample.image_path}")

    fixed_rows = json.loads(fixed.read_text(encoding="utf-8"))
    ref_by_id = {sample.id: sample for sample in dataset.get_reference_pool()}
    fixed_refs = [ref_by_id[row["id"]] for row in fixed_rows]
    query = dataset.get_test_samples()[0]
    get_prompter("zero_shot").build_vqa_prompt(query)
    get_prompter("fixed_random_6").build_vqa_prompt(query, fixed_references=fixed_refs)
    get_prompter("naive_icl", k=6, seed=3407).build_vqa_prompt(
        query,
        reference_pool=dataset.get_reference_pool(),
        k=6,
        rng_seed=3407,
    )
    parsed = OutputParser().parse_vqa('{"answer":"CT","confidence":0.9,"evidence":"visible axial slices"}', "smoke")
    expect(parsed.parse_success and parsed.answer == "CT", "VQA JSON parser failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()
    validate_configs()
    if not args.skip_data:
        validate_data()
    print("OK: SLAKE VQA setup is aligned")


if __name__ == "__main__":
    main()
