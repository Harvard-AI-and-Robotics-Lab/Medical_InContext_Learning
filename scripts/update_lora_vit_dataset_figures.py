#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"

COLOR_MAP = {
    "qwen_base": "#F1DCA7",
    "qwen_ret": "#F5A623",
    "gemma_base": "#B7D8F0",
    "gemma_ret": "#4F9BD9",
    "resnet": "#EF5350",
    "vit": "#7E57C2",
    "lora": "#2E7D32",
}


def add_or_replace(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    df = df[df["key"] != row["key"]].copy()
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def color_values(df: pd.DataFrame) -> list[str]:
    return [COLOR_MAP.get(str(value), str(value)) for value in df["color"]]


def plot_accuracy_auc(df: pd.DataFrame, output_prefix: Path, title: str):
    metrics = [metric for metric in ["accuracy", "auc"] if metric in df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(8.5, 0.58 * len(df) * len(metrics)), 4.4))
    if len(metrics) == 1:
        axes = [axes]

    colors = color_values(df)
    labels = df["label"].astype(str).tolist()

    for ax, metric in zip(axes, metrics):
        values = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        x = np.arange(len(df))
        ax.bar(x, values, color=colors, edgecolor="#555555", linewidth=0.8)
        ax.set_title(metric.upper() if metric == "auc" else "Accuracy", fontweight="bold")
        valid = values[np.isfinite(values)]
        if len(valid):
            ax.set_ylim(max(0.0, float(valid.min()) - 0.08), min(1.02, float(valid.max()) + 0.08))
        else:
            ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.text(i, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            else:
                ax.text(i, 0.03, "N/A", ha="center", va="bottom", fontsize=8, color="#666666")

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def update_tbx11k():
    source = FIG_DIR / "tbx11k_overview.csv"
    df = pd.read_csv(source)
    df = add_or_replace(
        df,
        {
            "key": "vit224",
            "label": "ViT224 Fine-tuned",
            "color": "vit",
            "accuracy": 0.9894444444444445,
            "auc": 0.9996366666666666,
        },
    )
    df = add_or_replace(
        df,
        {
            "key": "gemma4_lora",
            "label": "Gemma4 LoRA SFT",
            "color": "lora",
            "accuracy": 0.98,
            "auc": np.nan,
        },
    )
    out_csv = FIG_DIR / "tbx11k_overview_with_lora_vit.csv"
    df.to_csv(out_csv, index=False)
    plot_accuracy_auc(df, FIG_DIR / "tbx11k_main_experiment_overview_with_lora_vit", "TBX11K test performance")


def update_breakhis_binary_mean():
    source = FIG_DIR / "breakhis_binary_mean_accuracy.csv"
    df = pd.read_csv(source)
    df = add_or_replace(
        df,
        {
            "key": "vit224_full",
            "label": "ViT224 Fine-tuned (full)",
            "color": "vit",
            "accuracy": 0.9043478260869565,
        },
    )
    df = add_or_replace(
        df,
        {
            "key": "gemma4_lora_full",
            "label": "Gemma4 LoRA SFT (full)",
            "color": "lora",
            "accuracy": 0.8843478260869565,
        },
    )
    out_csv = FIG_DIR / "breakhis_binary_mean_accuracy_with_lora_vit.csv"
    df.to_csv(out_csv, index=False)
    plot_accuracy_auc(
        df,
        FIG_DIR / "breakhis_binary_mean_accuracy_with_lora_vit",
        "BreakHis 2-class test accuracy",
    )


def main():
    update_tbx11k()
    update_breakhis_binary_mean()


if __name__ == "__main__":
    main()
