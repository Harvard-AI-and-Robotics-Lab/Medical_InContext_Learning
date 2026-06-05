#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"

MODELS = {
    "qwen36": {"label": "Qwen3.6", "base_color": "#F1DCA7", "ret_color": "#F5A623"},
    "gemma4": {"label": "Gemma4", "base_color": "#B7D8F0", "ret_color": "#4F9BD9"},
    "medgemma27b": {"label": "MedGemma27B", "base_color": "#D8C7F0", "ret_color": "#8E63C7"},
}

METHODS = [
    ("noicl_fixed6_random6", "zero_shot", "No-ICL", "base"),
    ("noicl_fixed6_random6", "fixed_random_6", "Fixed-6", "base"),
    ("noicl_fixed6_random6", "random_icl", "Random-6", "base"),
    ("clip_top6", "rg_icl_global_similarity", "CLIP", "retrieval"),
    ("dinov3_cls_top6", "rg_icl_global_similarity", "DINOv3", "retrieval"),
    ("clip_dinov3cls_top6", "rg_icl_dual_global_similarity", "CLIP+DINO", "retrieval"),
]

LORA_SUMMARY = (
    ROOT
    / "outputs/lora/gemma4_vqa_rad_vqa_lora_language_projector_r16_a32_lr1e-4_seed3407_max512"
    / "test_generation_vllm_merged_max512/vqa_summary.json"
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_from_main_summary(path: Path, method: str) -> dict:
    data = read_json(path)["vqa_rad"][method]
    by_type = data.get("by_answer_type", {})
    return {
        "n": int(data["n_samples"]),
        "overall_exact": float(data["normalized_exact_match"]),
        "closed_exact": float(by_type.get("CLOSED", {}).get("normalized_exact_match", math.nan)),
        "open_exact": float(by_type.get("OPEN", {}).get("normalized_exact_match", math.nan)),
        "parse_success": float(data.get("parse_success", math.nan)),
    }


def metric_from_lora_summary(path: Path) -> dict:
    data = read_json(path)
    by_type = data.get("by_answer_type", {})
    return {
        "n": int(data["n"]),
        "overall_exact": float(data["normalized_exact_accuracy"]),
        "closed_exact": float(by_type.get("CLOSED", {}).get("normalized_exact_accuracy", math.nan)),
        "open_exact": float(by_type.get("OPEN", {}).get("normalized_exact_accuracy", math.nan)),
        "parse_success": float(data.get("parse_success_rate", math.nan)),
    }


def collect_rows() -> pd.DataFrame:
    rows = []
    for model_key, model_cfg in MODELS.items():
        for suffix, method_key, method_label, kind in METHODS:
            path = ROOT / "outputs" / "final" / f"vqa_rad_{model_key}_{suffix}" / "vqa_summary.json"
            metrics = metric_from_main_summary(path, method_key)
            rows.append(
                {
                    "dataset": "vqa_rad",
                    "label": f"{model_cfg['label']} {method_label}",
                    "model": model_cfg["label"],
                    "method": method_label,
                    "source": str(path.relative_to(ROOT)),
                    "color": model_cfg["ret_color"] if kind == "retrieval" else model_cfg["base_color"],
                    **metrics,
                }
            )

    if LORA_SUMMARY.exists():
        rows.append(
            {
                "dataset": "vqa_rad",
                "label": "Gemma4 LoRA SFT",
                "model": "Gemma4",
                "method": "LoRA SFT",
                "source": str(LORA_SUMMARY.relative_to(ROOT)),
                "color": "#9C6ADE",
                **metric_from_lora_summary(LORA_SUMMARY),
            }
        )

    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "vqa_rad_auto_metrics.csv"
    pdf_path = OUT_DIR / "vqa_rad_auto_metrics_horizontal.pdf"
    png_path = OUT_DIR / "vqa_rad_auto_metrics_horizontal.png"
    df.to_csv(csv_path, index=False)

    metrics = [
        ("overall_exact", "Overall normalized exact"),
        ("closed_exact", "Closed exact"),
        ("open_exact", "Open exact"),
    ]
    y = list(range(len(df)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, max(7.5, 0.43 * len(df) + 1.8)), sharey=True)
    labels = df["label"].tolist()

    for ax, (column, title) in zip(axes, metrics):
        vals = df[column].astype(float).tolist()
        ax.barh(y, vals, color=df["color"].tolist(), edgecolor="#555555", linewidth=0.9, alpha=0.92)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlim(0, 1.04)
        ax.grid(axis="x", color="#E6E6E6", linewidth=1.0)
        ax.set_axisbelow(True)
        for idx, val in enumerate(vals):
            if not math.isnan(val):
                ax.text(min(val + 0.012, 1.01), idx, f"{val:.3f}", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#4A4A4A")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=10)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].invert_yaxis()
    fig.suptitle("VQA-RAD main experiment overview (automatic exact metrics)", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965], w_pad=2.5)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"csv": str(csv_path), "pdf": str(pdf_path), "png": str(png_path), "rows": len(df)}, indent=2))


def main() -> None:
    plot(collect_rows())


if __name__ == "__main__":
    main()
