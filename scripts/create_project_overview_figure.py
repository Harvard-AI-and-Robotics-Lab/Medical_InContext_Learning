#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figs" / "medical_icl_project_overview.png"


COLORS = {
    "ink": "#17212b",
    "muted": "#5b6775",
    "line": "#c9d3df",
    "panel": "#f8fafc",
    "blue": "#2f80ed",
    "teal": "#12a594",
    "green": "#2e7d32",
    "orange": "#f2994a",
    "purple": "#7b61ff",
    "red": "#d64550",
}


def panel(ax, x, y, w, h, title, label):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.2,
            edgecolor=COLORS["line"],
            facecolor=COLORS["panel"],
        )
    )
    ax.text(x + 0.018, y + h - 0.036, label, fontsize=11, weight="bold", color=COLORS["blue"], va="top")
    ax.text(x + 0.07, y + h - 0.036, title, fontsize=13, weight="bold", color=COLORS["ink"], va="top")


def text(ax, x, y, s, size=9, color=None, weight=None, ha="left", va="top", width=None):
    if width:
        s = "\n".join(textwrap.wrap(s, width=width))
    ax.text(x, y, s, fontsize=size, color=color or COLORS["ink"], weight=weight, ha=ha, va=va)


def pill(ax, x, y, w, h, s, fc="#ffffff", ec=None, color=None, size=8.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.018",
            linewidth=0.9,
            edgecolor=ec or COLORS["line"],
            facecolor=fc,
        )
    )
    ax.text(x + w / 2, y + h / 2, s, fontsize=size, color=color or COLORS["ink"], ha="center", va="center")


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.3, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color or COLORS["muted"],
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def image_icon(ax, x, y, w, h, fc="#e8eef7", ec=None, kind="xray"):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=0.9, edgecolor=ec or COLORS["line"], facecolor=fc))
    if kind == "xray":
        ax.add_patch(Rectangle((x + 0.025 * w, y + 0.06 * h), 0.95 * w, 0.88 * h, facecolor="#24313f", edgecolor="none"))
        ax.add_patch(Circle((x + 0.38 * w, y + 0.56 * h), 0.18 * h, color="#d6e0eb", alpha=0.75))
        ax.add_patch(Circle((x + 0.62 * w, y + 0.56 * h), 0.18 * h, color="#d6e0eb", alpha=0.75))
    elif kind == "fundus":
        ax.add_patch(Circle((x + w / 2, y + h / 2), 0.36 * h, color="#cc5f3f", alpha=0.95))
        ax.add_patch(Circle((x + 0.62 * w, y + 0.58 * h), 0.08 * h, color="#f0c65d"))
    elif kind == "path":
        for i in range(5):
            ax.add_patch(Circle((x + (0.2 + 0.13 * i) * w, y + (0.35 + 0.08 * (i % 2)) * h), 0.07 * h, color="#b06aa0", alpha=0.75))
    else:
        ax.add_patch(Circle((x + w / 2, y + h / 2), 0.34 * h, color="#3f7ecb", alpha=0.85))


def dot_bank(ax, x, y, color):
    for r in range(3):
        for c in range(8):
            ax.add_patch(Circle((x + c * 0.012, y - r * 0.013), 0.0035, color=color, alpha=0.85))


def main():
    fig = plt.figure(figsize=(18, 10.125), dpi=220)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.text(
        0.5,
        0.965,
        "Retrieval-Guided Medical VLM In-Context Learning Overview",
        ha="center",
        va="top",
        fontsize=23,
        weight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.5,
        0.925,
        "Model-agnostic final protocol: train-only references, top-6 visual retrieval, JSON predictions, supervised and LoRA baselines",
        ha="center",
        va="top",
        fontsize=11,
        color=COLORS["muted"],
    )

    panel(ax, 0.035, 0.585, 0.29, 0.295, "Data and Splits", "A")
    for i, kind in enumerate(["xray", "fundus", "path", "retina"]):
        image_icon(ax, 0.06 + i * 0.057, 0.765, 0.045, 0.06, kind=kind)
    arrow(ax, 0.285, 0.795, 0.305, 0.795)
    pill(ax, 0.2, 0.684, 0.095, 0.035, "manifests", fc="#ffffff")
    pill(ax, 0.065, 0.686, 0.092, 0.03, "train", fc="#e7f4ee", color=COLORS["green"])
    pill(ax, 0.065, 0.648, 0.092, 0.03, "val", fc="#eef4ff", color=COLORS["blue"])
    pill(ax, 0.065, 0.61, 0.092, 0.03, "test", fc="#fff3e4", color=COLORS["orange"])
    text(ax, 0.18, 0.653, "Reference pool is locked to train split only.", size=8.8, color=COLORS["muted"], width=36)
    text(ax, 0.18, 0.613, "CheXpert-5 labels, 320 x 320", size=8.8, weight="bold")
    text(
        ax,
        0.18,
        0.592,
        "atelectasis, cardiomegaly,\nconsolidation, edema, pleural effusion",
        size=6.9,
        color=COLORS["muted"],
    )

    panel(ax, 0.355, 0.585, 0.29, 0.295, "Feature Banks", "B")
    image_icon(ax, 0.382, 0.772, 0.045, 0.06, kind="xray")
    arrow(ax, 0.43, 0.802, 0.475, 0.802, color=COLORS["blue"])
    for i, (name, color) in enumerate([("CLIP global", COLORS["blue"]), ("DINOv3 CLS", COLORS["purple"]), ("0.5 CLIP + 0.5 DINOv3", COLORS["teal"])]):
        y = 0.82 - i * 0.07
        pill(ax, 0.485, y - 0.022, 0.13, 0.034, name, fc="#ffffff", ec=color, color=color, size=8)
        dot_bank(ax, 0.49, y - 0.038, color)
    text(ax, 0.382, 0.658, "Cosine similarity ranks training images for each query.", size=9, color=COLORS["muted"], width=38)
    pill(ax, 0.382, 0.61, 0.09, 0.035, "k = 6", fc="#fff8e8", ec=COLORS["orange"], color=COLORS["orange"])
    pill(ax, 0.488, 0.61, 0.12, 0.035, "top-6 neighbors", fc="#fff8e8", ec=COLORS["orange"], color=COLORS["orange"])

    panel(ax, 0.675, 0.585, 0.29, 0.295, "ICL Prompt Construction", "C")
    image_icon(ax, 0.705, 0.76, 0.048, 0.062, kind="xray")
    text(ax, 0.704, 0.743, "query image", size=8, color=COLORS["muted"])
    arrow(ax, 0.76, 0.79, 0.81, 0.79)
    for i in range(6):
        image_icon(ax, 0.815 + (i % 3) * 0.042, 0.779 - (i // 3) * 0.052, 0.032, 0.039, kind="xray")
    pill(ax, 0.705, 0.686, 0.095, 0.031, "No ICL", fc="#ffffff")
    pill(ax, 0.812, 0.686, 0.095, 0.031, "Fixed-6", fc="#ffffff")
    pill(ax, 0.705, 0.64, 0.095, 0.031, "Random-6", fc="#ffffff")
    pill(ax, 0.812, 0.64, 0.12, 0.031, "Retrieved top-6", fc="#ffffff")
    text(ax, 0.705, 0.604, "Compact prompt: target image, optional references, label schema, strict JSON instruction.", size=8.8, color=COLORS["muted"], width=46)

    panel(ax, 0.035, 0.235, 0.45, 0.305, "Final CheXpert-5 VLM Runs", "D")
    for row, lane in enumerate(["Base VLM A", "Base VLM B"]):
        y = 0.445 - row * 0.12
        pill(ax, 0.06, y, 0.09, 0.04, lane, fc="#eef4ff", ec=COLORS["blue"], color=COLORS["blue"], size=8.5)
        x0 = 0.17
        for i, s in enumerate(["no/fixed/random", "CLIP top-6", "DINOv3 top-6", "CLIP+DINO top-6"]):
            pill(ax, x0 + i * 0.072, y, 0.064, 0.04, s, fc="#ffffff", size=7.2)
        arrow(ax, 0.462, y + 0.02, 0.475, y + 0.02, color=COLORS["muted"])
    pill(ax, 0.31, 0.276, 0.145, 0.045, "JSON findings + probabilities", fc="#e9f7f5", ec=COLORS["teal"], color=COLORS["teal"], size=8.5)
    text(ax, 0.06, 0.296, "Eight listed configs collapse to two anonymous VLM lanes and four shared method groups.", size=9, color=COLORS["muted"], width=52)

    panel(ax, 0.515, 0.235, 0.215, 0.305, "Baselines", "E")
    for i, (name, color) in enumerate([("ResNet50 fine-tune", COLORS["red"]), ("ViT fine-tune", COLORS["purple"]), ("Gemma4 language LoRA", COLORS["green"])]):
        y = 0.44 - i * 0.078
        image_icon(ax, 0.545, y - 0.006, 0.035, 0.044, kind="xray")
        arrow(ax, 0.585, y + 0.015, 0.615, y + 0.015, color=color)
        pill(ax, 0.62, y, 0.085, 0.036, name, fc="#ffffff", ec=color, color=color, size=7.6)
    text(ax, 0.545, 0.272, "Supervised classifiers and a language-side adapter baseline.", size=8.8, color=COLORS["muted"], width=32)

    panel(ax, 0.76, 0.235, 0.205, 0.305, "Outputs and Metrics", "F")
    pill(ax, 0.785, 0.465, 0.07, 0.04, "predictions", fc="#ffffff")
    pill(ax, 0.872, 0.465, 0.065, 0.04, "raw ledger", fc="#ffffff")
    arrow(ax, 0.82, 0.455, 0.82, 0.414, color=COLORS["teal"])
    arrow(ax, 0.902, 0.455, 0.902, 0.414, color=COLORS["teal"])
    pill(ax, 0.79, 0.37, 0.065, 0.034, "accuracy", fc="#eef4ff", color=COLORS["blue"])
    pill(ax, 0.868, 0.37, 0.065, 0.034, "macro AUC", fc="#eef4ff", color=COLORS["blue"])
    pill(ax, 0.79, 0.326, 0.065, 0.034, "F1", fc="#eef4ff", color=COLORS["blue"])
    pill(ax, 0.868, 0.326, 0.065, 0.034, "calibration", fc="#eef4ff", color=COLORS["blue"])
    text(ax, 0.792, 0.292, "Final tables and paper plots summarize model-agnostic method rows.", size=8.8, color=COLORS["muted"], width=30)

    ax.text(
        0.035,
        0.06,
        "Protocol anchors: seed 3407; k = 6; train-only references; CheXpert-5 target schema; JSON response parsing; metrics from predictions and probabilities.",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
