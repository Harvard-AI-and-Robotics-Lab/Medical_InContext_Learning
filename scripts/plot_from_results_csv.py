import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_accuracy_auc(csv_path: Path, output_prefix: Path, title: str):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"{csv_path} must contain a label column.")
    metrics = [m for m in ["accuracy", "auc"] if m in df.columns]
    if not metrics:
        raise ValueError(f"{csv_path} has no accuracy/auc columns.")

    fig, axes = plt.subplots(1, len(metrics), figsize=(max(7, 0.55 * len(df) * len(metrics)), 4.2))
    if len(metrics) == 1:
        axes = [axes]
    colors = df["color"].tolist() if "color" in df.columns else ["#4F9BD9"] * len(df)

    for ax, metric in zip(axes, metrics):
        values = df[metric].astype(float).to_numpy()
        x = np.arange(len(df))
        ax.bar(x, values, color=colors, edgecolor="#555555", linewidth=0.8)
        ax.set_title(metric.upper() if metric == "auc" else metric.capitalize(), fontweight="bold")
        ax.set_ylim(max(0.0, np.nanmin(values) - 0.08), min(1.02, np.nanmax(values) + 0.08))
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"], rotation=0, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for i, v in enumerate(values):
            ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--title", default="LAG glaucoma classification")
    args = parser.parse_args()
    plot_accuracy_auc(args.csv, args.output_prefix, args.title)


if __name__ == "__main__":
    main()
