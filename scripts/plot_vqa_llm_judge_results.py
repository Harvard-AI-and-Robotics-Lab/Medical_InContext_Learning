#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    ("zero_shot", "No-ICL", "base"),
    ("fixed6", "Fixed-6", "base"),
    ("random6", "Random-6", "base"),
    ("clip_top6", "CLIP", "retrieval"),
    ("dinov3_cls_top6", "DINOv3", "retrieval"),
    ("clip_dinov3cls_top6", "CLIP+DINO", "retrieval"),
]

DATASETS = {
    "slake": {
        "title": "SLAKE VQA LLM-judge results",
        "judge_dir": ROOT / "outputs/judge/slake_main",
        "lora_summary": ROOT / "outputs/judge/slake_gemma4_lora_projector512_gpt54mini_full_summary.json",
    },
    "pathvqa": {
        "title": "PathVQA LLM-judge results",
        "judge_dir": ROOT / "outputs/judge/pathvqa_main",
        "lora_summary": ROOT / "outputs/judge/pathvqa_gemma4_lora_projector512_gpt54mini_full_summary.json",
    },
    "vqamed2019": {
        "title": "VQA-Med2019 LLM-judge results",
        "judge_dir": ROOT / "outputs/judge/vqamed2019_main",
        "lora_summary": ROOT / "outputs/judge/vqamed2019_gemma4_lora_projector512_gpt54mini_full_summary.json",
    },
    "vqa_rad": {
        "title": "VQA-RAD LLM-judge results",
        "judge_dir": ROOT / "outputs/judge/vqa_rad_main",
        "lora_summary": ROOT / "outputs/judge/vqa_rad_gemma4_lora_projector512_gpt54mini_full_summary.json",
    },
}

SCORE_KEYS = [
    ("semantic_accuracy", "semantic_100", "Semantic = 100"),
    ("completeness", "completeness_100", "Completeness = 100"),
    ("factuality", "factuality_100", "Factuality = 100"),
    ("conciseness", "conciseness_100", "Conciseness = 100"),
]


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_score(value):
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return None
    if 0 <= score <= 100:
        return score
    return None


def summary_to_jsonl_path(path: Path) -> Path:
    name = path.name
    if name.endswith("_summary.json"):
        return path.with_name(name[: -len("_summary.json")] + ".jsonl")
    return path.with_suffix(".jsonl")


def metric_100_rates(summary_path_: Path, summary: dict) -> dict[str, float]:
    rates = {}
    for score_key, column, _title in SCORE_KEYS:
        rates[column] = float(summary.get(f"{score_key}_100_accuracy", math.nan))

    if all(not math.isnan(value) for value in rates.values()):
        return rates

    jsonl_path = summary_to_jsonl_path(summary_path_)
    if not jsonl_path.exists():
        rates["semantic_100"] = float(summary.get("exact_match_accuracy", rates["semantic_100"]))
        return rates

    counts = {score_key: 0 for score_key, _column, _title in SCORE_KEYS}
    n_valid = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("judge_parse_success"):
                continue
            parsed = row.get("parsed_judge", {})
            scores = {score_key: normalize_score(parsed.get(score_key)) for score_key, _column, _title in SCORE_KEYS}
            if any(score is None for score in scores.values()):
                continue
            n_valid += 1
            for score_key, score in scores.items():
                if score == 100:
                    counts[score_key] += 1

    if not n_valid:
        rates["semantic_100"] = float(summary.get("exact_match_accuracy", rates["semantic_100"]))
        return rates

    for score_key, column, _title in SCORE_KEYS:
        rates[column] = counts[score_key] / n_valid
    return rates


def summary_path(dataset_key: str, model_key: str, method_key: str) -> Path:
    cfg = DATASETS[dataset_key]
    return cfg["judge_dir"] / f"{dataset_key}_{model_key}_{method_key}_gpt54mini_summary.json"


def row_from_summary(dataset_key: str, label: str, model: str, method: str, color: str, path: Path):
    data = read_json(path)
    if data is None:
        return None
    rates = metric_100_rates(path, data)
    return {
        "dataset": dataset_key,
        "label": label,
        "model": model,
        "method": method,
        "n": int(data.get("n", 0)),
        "exact": rates["semantic_100"],
        "semantic_100": rates["semantic_100"],
        "completeness_100": rates["completeness_100"],
        "factuality_100": rates["factuality_100"],
        "conciseness_100": rates["conciseness_100"],
        "sem_ge80": float(data.get("semantic_accuracy_ge80_accuracy", math.nan)),
        "mean_semantic": float(data.get("mean_semantic_accuracy", math.nan)),
        "mean_completeness": float(data.get("mean_completeness", math.nan)),
        "mean_factuality": float(data.get("mean_factuality", math.nan)),
        "mean_conciseness": float(data.get("mean_conciseness", math.nan)),
        "parse_success": float(data.get("parse_success_rate", math.nan)),
        "open_exact": float(data.get("by_answer_type", {}).get("OPEN", {}).get("exact_match_accuracy", math.nan)),
        "closed_exact": float(data.get("by_answer_type", {}).get("CLOSED", {}).get("exact_match_accuracy", math.nan)),
        "source": str(path.relative_to(ROOT)),
        "color": color,
    }


def collect_rows(dataset_key: str) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    for model_key, model_cfg in MODELS.items():
        for method_key, method_label, kind in METHODS:
            path = summary_path(dataset_key, model_key, method_key)
            color = model_cfg["ret_color"] if kind == "retrieval" else model_cfg["base_color"]
            row = row_from_summary(
                dataset_key,
                f"{model_cfg['label']} {method_label}",
                model_cfg["label"],
                method_label,
                color,
                path,
            )
            if row is None:
                missing.append(str(path.relative_to(ROOT)))
                continue
            rows.append(row)

    lora_path = DATASETS[dataset_key].get("lora_summary")
    if lora_path is not None:
        row = row_from_summary(
            dataset_key,
            "Gemma4 LoRA SFT",
            "Gemma4",
            "LoRA SFT",
            "#1F4E8C",
            lora_path,
        )
        if row is None:
            missing.append(str(lora_path.relative_to(ROOT)))
        else:
            rows.append(row)

    return pd.DataFrame(rows), missing


def plot_dataset(dataset_key: str, df: pd.DataFrame) -> Path:
    cfg = DATASETS[dataset_key]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"vqa_llm_judge_{dataset_key}.csv"
    pdf_path = OUT_DIR / f"vqa_llm_judge_{dataset_key}_horizontal.pdf"
    png_path = OUT_DIR / f"vqa_llm_judge_{dataset_key}_horizontal.png"
    df.to_csv(csv_path, index=False)

    if df.empty:
        raise RuntimeError(f"No rows to plot for {dataset_key}")

    labels = list(df["label"])
    y = list(range(len(df)))
    height = max(7.0, 0.45 * len(df) + 1.8)
    fig, axes = plt.subplots(1, 4, figsize=(27, height), sharey=True)
    metrics = [(column, title) for _score_key, column, title in SCORE_KEYS]

    for ax, (metric, title) in zip(axes, metrics):
        vals = df[metric].astype(float).tolist()
        colors = df["color"].tolist()
        ax.barh(y, vals, color=colors, edgecolor="#555555", linewidth=0.9, alpha=0.92)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlim(0, 1.04)
        ax.grid(axis="x", color="#E6E6E6", linewidth=1.0)
        ax.set_axisbelow(True)
        for idx, val in enumerate(vals):
            if math.isnan(val):
                continue
            x = min(val + 0.012, 1.01)
            ax.text(x, idx, f"{val:.3f}", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#4A4A4A")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=10.5)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].invert_yaxis()
    fig.suptitle(cfg["title"], fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=2.5)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    args = parser.parse_args()

    dataset_keys = args.dataset or sorted(DATASETS)
    for dataset_key in dataset_keys:
        df, missing = collect_rows(dataset_key)
        path = plot_dataset(dataset_key, df)
        print(json.dumps({
            "dataset": dataset_key,
            "n_rows": len(df),
            "plot": str(path),
            "csv": str(OUT_DIR / f"vqa_llm_judge_{dataset_key}.csv"),
            "missing": missing,
        }, indent=2))


if __name__ == "__main__":
    main()
