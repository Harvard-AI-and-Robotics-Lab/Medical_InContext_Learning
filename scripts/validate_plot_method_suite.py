#!/usr/bin/env python3
"""Smoke-test the reproducible plot method matrix without launching models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FINAL_CONFIGS = ROOT / "configs" / "final"

DATASETS = {
    "lag": {
        "qwen_prefix": "qwen36",
        "gemma_prefix": "gemma4",
        "suffix": "",
        "lora_dataset": "lag_project",
        "lora_output_name": "lag_project",
    },
    "tbx11k": {
        "qwen_prefix": "tbx11k_qwen36",
        "gemma_prefix": "tbx11k_gemma4",
        "suffix": "",
        "lora_dataset": "tbx11k",
        "lora_output_name": "tbx11k",
    },
    "ddr_512": {
        "qwen_prefix": "ddr_qwen36",
        "gemma_prefix": "ddr_gemma4",
        "suffix": "_512",
        "lora_dataset": "ddr_512",
        "lora_output_name": "ddr512",
    },
    "breakhis_binary": {
        "qwen_prefix": "breakhis_binary_qwen36",
        "gemma_prefix": "breakhis_binary_gemma4",
        "suffix": "",
        "lora_dataset": "breakhis_binary",
        "lora_output_name": "breakhis_binary",
    },
}

NOICL_ROWS = ("zero_shot", "fixed_random_6", "random_icl")
RETRIEVAL_ROWS = {
    "clip_top6": "rg_icl_global_similarity",
    "dinov3_cls_top6": "rg_icl_global_similarity",
    "clip_dinov3cls_top6": "rg_icl_dual_global_similarity",
}
FORMAL_SYNC_FILES = (
    "scripts/train_gemma4_language_lora.py",
    "scripts/eval_gemma4_language_lora.py",
    "scripts/eval_gemma4_lora_icl.py",
    "scripts/eval_gemma4_lora_vllm_api.py",
    "src/prompting/rg_icl.py",
)


class CheckError(Exception):
    pass


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise CheckError(f"Missing config: {path.relative_to(ROOT)}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise CheckError(f"Config is not a mapping: {path.relative_to(ROOT)}")
    return data


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def validate_inference_contract(path: Path, cfg: dict) -> None:
    inf = cfg.get("inference", {})
    rel = path.relative_to(ROOT)
    expect(inf.get("temperature") == 1.0, f"{rel}: temperature must be 1.0")
    expect(inf.get("top_p") is None, f"{rel}: top_p must be null")
    expect(inf.get("max_tokens") == 512, f"{rel}: max_tokens must be 512")
    expect(inf.get("seed") == 3407, f"{rel}: seed must be 3407")
    expect(inf.get("response_format") == "json_object", f"{rel}: response_format must be json_object")
    expect(inf.get("enable_thinking") is False, f"{rel}: enable_thinking must be false")
    expect(cfg.get("k") == 6, f"{rel}: top-level k must be 6")
    expect(cfg.get("retrieval", {}).get("k") == 6, f"{rel}: retrieval.k must be 6")


def config_path(prefix: str, stem: str, suffix: str) -> Path:
    return FINAL_CONFIGS / f"{prefix}_{stem}{suffix}.yaml"


def validate_vlm_matrix() -> tuple[int, list[str]]:
    rows: list[str] = []
    for dataset, spec in DATASETS.items():
        for model_label, prefix_key in (("qwen36", "qwen_prefix"), ("gemma4", "gemma_prefix")):
            prefix = spec[prefix_key]
            suffix = spec["suffix"]

            noicl = config_path(prefix, "noicl_fixed6_random6", suffix)
            cfg = load_yaml(noicl)
            validate_inference_contract(noicl, cfg)
            methods = tuple(cfg.get("methods", ()))
            expect(methods == NOICL_ROWS, f"{noicl.relative_to(ROOT)}: methods must be {NOICL_ROWS}")
            rows.extend(f"{dataset}:{model_label}:{method}" for method in NOICL_ROWS)

            for stem, method in RETRIEVAL_ROWS.items():
                path = config_path(prefix, stem, suffix)
                cfg = load_yaml(path)
                validate_inference_contract(path, cfg)
                methods = tuple(cfg.get("methods", ()))
                if stem == "clip_dinov3cls_top6":
                    allowed = (("rg_icl_dual_global_similarity",), ("rg_icl_global_similarity",))
                    expect(methods in allowed, f"{path.relative_to(ROOT)}: methods must be one of {allowed}")
                    features_dir = str(cfg.get("retrieval_features_dir", ""))
                    expect("clip_dinov3cls" in features_dir, f"{path.relative_to(ROOT)}: fused row must use CLIP+DINO features")
                else:
                    expect(methods == (method,), f"{path.relative_to(ROOT)}: methods must be ({method!r},)")
                rows.append(f"{dataset}:{model_label}:{stem}")

        rows.append(f"{dataset}:gemma4_language_lora")

    expected = len(DATASETS) * 13
    expect(len(rows) == expected, f"Expected {expected} plotted rows across datasets, found {len(rows)}")
    return expected, rows


def validate_lora_scripts() -> None:
    train = (ROOT / "scripts" / "train_gemma4_language_lora.py").read_text(encoding="utf-8")
    eval_hf = (ROOT / "scripts" / "eval_gemma4_language_lora.py").read_text(encoding="utf-8")
    eval_icl = (ROOT / "scripts" / "eval_gemma4_lora_icl.py").read_text(encoding="utf-8")
    eval_vllm = (ROOT / "scripts" / "eval_gemma4_lora_vllm_api.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts" / "run_plot_method_suite.sh").read_text(encoding="utf-8")

    for dataset in ("breakhis_binary", "tbx11k", "ddr_512", "lag_project"):
        expect(f'"{dataset}":' in train, f"LoRA train preset missing {dataset}")
    expect('"lag_project": {\n        "dataset_name": "lag_project"' in train, "lag_project preset is malformed")
    expect('"image_max_side": 512' in train, "LoRA train script must include 512-side presets")
    expect('add_argument("--max_new_tokens", type=int, default=512)' in eval_hf, "HF LoRA eval default must be 512")
    expect('add_argument("--max_new_tokens", type=int, default=512)' in eval_icl, "LoRA+ICL eval default must be 512")
    expect('add_argument("--max_tokens", type=int, default=512)' in eval_vllm, "vLLM LoRA eval default must be 512")
    expect('lora_dataset="lag_project"' in runner, "runner must map lag to LoRA dataset lag_project")
    expect('lora_output_name="lag_project"' in runner, "runner must map lag output to lag_project")


def validate_medgemma_configs() -> int:
    paths = sorted(FINAL_CONFIGS.glob("*medgemma27b*.yaml"))
    expect(len(paths) == 24, f"Expected 24 MedGemma configs copied from formal repo, found {len(paths)}")
    required = [
        "tbx11k_medgemma27b_noicl_fixed6_random6.yaml",
        "tbx11k_medgemma27b_clip_top6.yaml",
        "tbx11k_medgemma27b_dinov3_cls_top6.yaml",
        "tbx11k_medgemma27b_clip_dinov3cls_top6.yaml",
        "ddr_medgemma27b_noicl_fixed6_random6_512.yaml",
        "ddr_medgemma27b_clip_dinov3cls_top6_512.yaml",
        "breakhis_binary_mag40_medgemma27b_noicl_fixed6_random6_subtype.yaml",
        "breakhis_binary_mag400_medgemma27b_clip_dinov3cls05_top6_subtype.yaml",
    ]
    for name in required:
        expect((FINAL_CONFIGS / name).exists(), f"Missing MedGemma config {name}")
    return len(paths)


def validate_against_formal(formal_repo: Path) -> None:
    formal_repo = formal_repo.resolve()
    expect(formal_repo.exists(), f"Formal repo does not exist: {formal_repo}")

    for spec in DATASETS.values():
        for prefix_key in ("qwen_prefix", "gemma_prefix"):
            prefix = spec[prefix_key]
            suffix = spec["suffix"]
            for stem in ("noicl_fixed6_random6", "clip_top6", "dinov3_cls_top6", "clip_dinov3cls_top6"):
                rel = Path("configs") / "final" / f"{prefix}_{stem}{suffix}.yaml"
                mine = ROOT / rel
                formal = formal_repo / rel
                expect(formal.exists(), f"Formal config missing: {formal}")
                expect(mine.read_bytes() == formal.read_bytes(), f"{rel} differs from formal repo")

    for rel in FORMAL_SYNC_FILES:
        mine = ROOT / rel
        formal = formal_repo / rel
        expect(formal.exists(), f"Formal file missing: {formal}")
        expect(mine.read_bytes() == formal.read_bytes(), f"{rel} differs from formal repo")

    medgemma_names = sorted(path.name for path in FINAL_CONFIGS.glob("*medgemma27b*.yaml"))
    expect(len(medgemma_names) == 24, f"Local repo has {len(medgemma_names)} MedGemma configs, expected 24")
    for name in medgemma_names:
        formal = formal_repo / "configs" / "final" / name
        mine = FINAL_CONFIGS / formal.name
        expect(mine.exists(), f"Missing copied MedGemma config: {formal.name}")
        expect(formal.exists(), f"Formal MedGemma config missing: {formal.name}")
        expect(mine.read_bytes() == formal.read_bytes(), f"MedGemma config differs from formal repo: {formal.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-repo", type=Path, default=None, help="Optional formal repo to byte-compare synced files.")
    args = parser.parse_args()

    try:
        plotted_rows, _ = validate_vlm_matrix()
        validate_lora_scripts()
        medgemma_count = validate_medgemma_configs()
        if args.formal_repo is not None:
            validate_against_formal(args.formal_repo)
    except CheckError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("OK: plot method suite is aligned")
    print(f"OK: {plotted_rows} plotted rows checked across {len(DATASETS)} datasets")
    print("OK: each dataset expands to 13 non-supervised rows")
    print(f"OK: {medgemma_count} MedGemma configs present")
    if args.formal_repo is not None:
        print(f"OK: synced files match formal repo {args.formal_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
