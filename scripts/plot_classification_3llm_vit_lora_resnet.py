#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.output_parser import OutputParser  # noqa: E402
from src.metrics.classification import ClassificationMetrics  # noqa: E402


OUT_DIR = ROOT / "outputs" / "figures"
MAGNIFICATIONS = ["40", "100", "200", "400"]

DATASETS = {
    "breakhis_binary": {
        "title": "BreakHis 2-class",
        "task_dataset": "breakhis_binary",
        "label_names": ["benign", "malignant"],
        "prompt_dataset_name": "breakhis_binary",
        "lora_predictions": ROOT
        / "outputs/lora/gemma4_language_lora_breakhis_binary_r16_a32_lr1e-4_seed3407/test_generation_max512_predictions.json",
        "lora_note": "Gemma4 LoRA max_new_tokens=512; pooled full test; AUC from generated probability.",
    },
    "tbx11k": {
        "title": "TBX11K",
        "task_dataset": "tbx11k",
        "label_names": ["healthy", "sick but non-TB", "TB"],
        "prompt_dataset_name": "tbx11k",
        "lora_predictions": ROOT
        / "outputs/lora/gemma4_language_lora_tbx11k_r16_a32_lr1e-4_seed3407/test_generation_max512_predictions.json",
        "lora_note": "Gemma4 LoRA max_new_tokens=512; AUC from generated class probabilities.",
    },
    "ddr_512": {
        "title": "DDR-512",
        "task_dataset": "ddr",
        "label_names": [
            "no_dr",
            "mild_npdr",
            "moderate_npdr",
            "severe_npdr",
            "proliferative_dr",
            "ungradable",
        ],
        "prompt_dataset_name": "ddr",
        "lora_predictions": ROOT
        / "outputs/lora/gemma4_language_lora_ddr512_r16_a32_lr1e-4_seed3407_epoch5_es2/test_generation_vllm_merged_max512_predictions.json",
        "lora_note": "Gemma4 LoRA merged-vLLM max_tokens=512; AUC from generated class probabilities.",
    },
    "lag": {
        "title": "LAG",
        "task_dataset": "lag_project",
        "label_names": ["non_glaucoma", "glaucoma"],
        "prompt_dataset_name": "lag",
        "lora_predictions": ROOT
        / "outputs/lora/gemma4_language_lora_lag_project_r16_a32_lr1e-4_seed3407/test_generation_vllm_merged_max512_predictions.json",
        "lora_note": "Gemma4 LoRA merged-vLLM max_tokens=512; AUC from generated glaucoma probability.",
    },
}

MODELS = {
    "qwen36": {"label": "Qwen3.6", "base_color": "#F1DCA7", "ret_color": "#F5A623"},
    "gemma4": {"label": "Gemma4", "base_color": "#B7D8F0", "ret_color": "#4F9BD9"},
    "medgemma27b": {"label": "MedGemma27B", "base_color": "#D8C7F0", "ret_color": "#8E63C7"},
}

BASE_METHODS = [
    ("zero_shot", "No-ICL"),
    ("fixed_random_6", "Fixed-6"),
    ("random_icl", "Random-6"),
]

RETRIEVAL_METHODS = [
    ("clip", "CLIP"),
    ("dinov3", "DINOv3"),
    ("fusion", "CLIP+DINO"),
]

SUPERVISED = [
    ("resnet", "ResNet50", "#EF5350"),
    ("vit", "ViT224", "#7E57C2"),
    ("lora", "Gemma4 LoRA", "#2E7D32"),
]


def read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_metrics(metrics_path: Path) -> tuple[str, dict] | tuple[None, None]:
    data = read_json(metrics_path)
    if not isinstance(data, dict):
        return None, None
    for method, metrics in data.items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            return method, metrics
    return None, None


def metric_from_path(path: Path, method: str | None = None) -> dict | None:
    data = read_json(path)
    if not isinstance(data, dict):
        return None
    if method is not None:
        value = data.get(method)
        return value if isinstance(value, dict) else None
    _, value = first_metrics(path)
    return value


def metric_value(metrics: dict, metric_name: str) -> float:
    if metric_name == "auc":
        value = metrics.get("auc", metrics.get("macro_auc"))
    else:
        value = metrics.get(metric_name)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def run_path(dataset_key: str, model_key: str, run_kind: str, mag: str | None = None) -> Path:
    if dataset_key == "breakhis_binary":
        if run_kind == "base":
            name = f"breakhis_binary_mag{mag}_{model_key}_noicl_fixed6_random6_subtype"
        elif run_kind == "clip":
            name = f"breakhis_binary_mag{mag}_{model_key}_clip_top6_subtype"
        elif run_kind == "dinov3":
            name = f"breakhis_binary_mag{mag}_{model_key}_dinov3_cls_top6_subtype"
        elif run_kind == "fusion":
            name = f"breakhis_binary_mag{mag}_{model_key}_clip_dinov3cls05_top6_subtype"
        else:
            raise ValueError(run_kind)
        return ROOT / "outputs/final" / name / "breakhis_binary/metrics.json"

    if dataset_key == "tbx11k":
        stem = "tbx11k"
        suffix = ""
        task_dir = "tbx11k"
    elif dataset_key == "ddr_512":
        stem = "ddr"
        suffix = "_512"
        task_dir = "ddr"
    else:
        raise ValueError(dataset_key)

    if run_kind == "base":
        name = f"{stem}_{model_key}_noicl_fixed6_random6{suffix}"
    elif run_kind == "clip":
        name = f"{stem}_{model_key}_clip_top6{suffix}"
    elif run_kind == "dinov3":
        name = f"{stem}_{model_key}_dinov3_cls_top6{suffix}"
    elif run_kind == "fusion":
        name = f"{stem}_{model_key}_clip_dinov3cls_top6{suffix}"
    else:
        raise ValueError(run_kind)
    return ROOT / "outputs/final" / name / task_dir / "metrics.json"


def aggregate_breakhis(model_key: str, run_kind: str, method: str | None) -> tuple[float, float, str]:
    rows = []
    missing = []
    for mag in MAGNIFICATIONS:
        path = run_path("breakhis_binary", model_key, run_kind, mag)
        metrics = metric_from_path(path, method)
        if metrics is None:
            missing.append(str(path.relative_to(ROOT)))
            continue
        rows.append(metrics)
    if missing or not rows:
        return float("nan"), float("nan"), "; ".join(missing)
    return (
        float(np.mean([metric_value(row, "accuracy") for row in rows])),
        float(np.mean([metric_value(row, "auc") for row in rows])),
        "Simple mean over BreakHis magnifications 40/100/200/400.",
    )


def add_main_rows_for_dataset(dataset_key: str, rows: list[dict], missing: list[dict]) -> None:
    if dataset_key == "lag":
        add_lag_main_rows(rows, missing)
        return

    for model_key, model_cfg in MODELS.items():
        for method, method_label in BASE_METHODS:
            if dataset_key == "breakhis_binary":
                acc, auc, note = aggregate_breakhis(model_key, "base", method)
                source = "BreakHis magnification subtype runs"
            else:
                path = run_path(dataset_key, model_key, "base")
                metrics = metric_from_path(path, method)
                if metrics is None:
                    fallback = fallback_overview_metric(dataset_key, model_key, method)
                    if fallback is None:
                        acc = auc = float("nan")
                        note = "Missing metrics."
                        source = str(path.relative_to(ROOT))
                        missing.append(missing_row(dataset_key, model_cfg["label"], method_label, source))
                    else:
                        acc = fallback["accuracy"]
                        auc = fallback["auc"]
                        note = fallback["note"]
                        source = fallback["source"]
                else:
                    acc = metric_value(metrics, "accuracy")
                    auc = metric_value(metrics, "auc")
                    note = ""
                    source = str(path.relative_to(ROOT))
            rows.append(
                row(
                    dataset_key,
                    f"{model_cfg['label']} {method_label}",
                    model_cfg["label"],
                    method_label,
                    "main",
                    acc,
                    auc,
                    model_cfg["base_color"],
                    source,
                    note,
                )
            )

        for run_kind, method_label in RETRIEVAL_METHODS:
            if dataset_key == "breakhis_binary":
                acc, auc, note = aggregate_breakhis(model_key, run_kind, None)
                source = "BreakHis magnification subtype runs"
            else:
                path = run_path(dataset_key, model_key, run_kind)
                metrics = metric_from_path(path)
                if metrics is None:
                    acc = auc = float("nan")
                    note = "Missing metrics."
                    source = str(path.relative_to(ROOT))
                    missing.append(missing_row(dataset_key, model_cfg["label"], method_label, source))
                else:
                    acc = metric_value(metrics, "accuracy")
                    auc = metric_value(metrics, "auc")
                    note = ""
                    source = str(path.relative_to(ROOT))
            rows.append(
                row(
                    dataset_key,
                    f"{model_cfg['label']} {method_label}",
                    model_cfg["label"],
                    method_label,
                    "main",
                    acc,
                    auc,
                    model_cfg["ret_color"],
                    source,
                    note,
                )
            )


def fallback_overview_metric(dataset_key: str, model_key: str, method: str) -> dict | None:
    if method != "zero_shot":
        return None
    if dataset_key != "tbx11k":
        return None
    key = {"qwen36": "qwen_zero", "gemma4": "gemma_zero"}.get(model_key)
    if key is None:
        return None
    path = ROOT / "outputs/figures/tbx11k_overview.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    hit = df[df["key"].astype(str) == key]
    if hit.empty:
        return None
    record = hit.iloc[0]
    return {
        "accuracy": float(record["accuracy"]),
        "auc": float(record["auc"]),
        "source": f"{path.relative_to(ROOT)}:{key}",
        "note": "Fallback from existing TBX11K overview because zero_shot is absent from the noicl_fixed6_random6 metrics file.",
    }


def add_lag_main_rows(rows: list[dict], missing: list[dict]) -> None:
    lag_csv = ROOT / "results/tables/figure_data/main_experiment_overview.csv"
    df = pd.read_csv(lag_csv)
    label_map = {
        ("Qwen3.6", "No-ICL"): "Qwen3.6\nNo-ICL",
        ("Qwen3.6", "Fixed-6"): "Qwen3.6\nFixed-6",
        ("Qwen3.6", "Random-6"): "Qwen3.6\nRandom-6",
        ("Qwen3.6", "CLIP"): "Qwen3.6\nCLIP",
        ("Qwen3.6", "CLIP+DINO"): "Qwen3.6\nCLIP+DINO",
        ("Gemma4", "No-ICL"): "Gemma4\nNo-ICL",
        ("Gemma4", "Fixed-6"): "Gemma4\nFixed-6",
        ("Gemma4", "Random-6"): "Gemma4\nRandom-6",
        ("Gemma4", "CLIP"): "Gemma4\nCLIP",
        ("Gemma4", "CLIP+DINO"): "Gemma4\nCLIP+DINO",
    }
    for model_key, cfg in MODELS.items():
        for method_label in ["No-ICL", "Fixed-6", "Random-6", "CLIP", "DINOv3", "CLIP+DINO"]:
            label = label_map.get((cfg["label"], method_label))
            if method_label == "DINOv3" and cfg["label"] == "Gemma4":
                add_lag_gemma_dinov3(rows)
                continue
            if label is None:
                missing.append(missing_row("lag", cfg["label"], method_label, "No LAG run found."))
                continue
            hit = df[df["label"].astype(str) == label]
            if hit.empty:
                missing.append(missing_row("lag", cfg["label"], method_label, f"Missing row {label} in {lag_csv}"))
                continue
            record = hit.iloc[0]
            color = cfg["base_color"] if method_label in {"No-ICL", "Fixed-6", "Random-6"} else cfg["ret_color"]
            rows.append(
                row(
                    "lag",
                    f"{cfg['label']} {method_label}",
                    cfg["label"],
                    method_label,
                    "main",
                    float(record["accuracy"]),
                    float(record["auc"]),
                    color,
                    f"{lag_csv.relative_to(ROOT)}:{label}",
                    str(record.get("note", "")),
                )
            )


def add_lag_gemma_dinov3(rows: list[dict]) -> None:
    path = ROOT / "results/tables/figure_data/dinov3_embedding_ablation_gemma4.csv"
    df = pd.read_csv(path)
    hit = df[df["label"].astype(str) == "CLS/global"]
    if hit.empty:
        return
    record = hit.iloc[0]
    rows.append(
        row(
            "lag",
            "Gemma4 DINOv3",
            "Gemma4",
            "DINOv3",
            "main",
            float(record["accuracy"]),
            float(record["auc"]),
            MODELS["gemma4"]["ret_color"],
            f"{path.relative_to(ROOT)}:CLS/global",
            "LAG DINOv3 CLS/global ablation row; no Qwen/MedGemma DINOv3 LAG run found.",
        )
    )


def add_supervised_rows(dataset_key: str, rows: list[dict], missing: list[dict]) -> None:
    if dataset_key == "breakhis_binary":
        add_breakhis_supervised(rows, missing)
        add_lora_row(dataset_key, rows, missing)
        return

    if dataset_key == "tbx11k":
        paths = {
            "resnet": ROOT / "outputs/final/resnet50_tbx11k_384_seed3407/metrics.json",
            "vit": ROOT / "outputs/final/vit224_tbx11k_seed3407/metrics.json",
        }
    elif dataset_key == "ddr_512":
        paths = {
            "resnet": ROOT / "outputs/final/resnet50_ddr_crop_pad_512_384_seed3407/metrics.json",
            "vit": ROOT / "outputs/final/vit224_ddr_crop_pad_512_seed3407/metrics.json",
        }
    elif dataset_key == "lag":
        add_lag_supervised(rows, missing)
        add_lora_row(dataset_key, rows, missing)
        return
    else:
        raise ValueError(dataset_key)

    for key, label, color in SUPERVISED:
        if key == "lora":
            add_lora_row(dataset_key, rows, missing)
            continue
        path = paths[key]
        metrics = read_json(path)
        test = metrics.get("test") if isinstance(metrics, dict) else None
        if not isinstance(test, dict):
            missing.append(missing_row(dataset_key, label, "supervised", str(path.relative_to(ROOT))))
            acc = auc = float("nan")
        else:
            acc = metric_value(test, "accuracy")
            auc = metric_value(test, "auc")
        rows.append(
            row(dataset_key, label, label, "Fine-tuned", "supervised", acc, auc, color, str(path.relative_to(ROOT)), "")
        )


def add_breakhis_supervised(rows: list[dict], missing: list[dict]) -> None:
    for key, label, color in SUPERVISED:
        if key == "lora":
            continue
        if key == "vit":
            path = ROOT / "outputs/final/vit224_breakhis_binary_seed3407/metrics.json"
            metrics = read_json(path)
            test = metrics.get("test") if isinstance(metrics, dict) else None
            if not isinstance(test, dict):
                missing.append(missing_row("breakhis_binary", label, "supervised", str(path.relative_to(ROOT))))
                acc = auc = float("nan")
            else:
                acc = metric_value(test, "accuracy")
                auc = metric_value(test, "auc")
            note = "Pooled full BreakHis binary test; not magnification mean."
            source = str(path.relative_to(ROOT))
        else:
            vals = []
            miss = []
            for mag in MAGNIFICATIONS:
                path = ROOT / f"outputs/final/resnet50_breakhis_binary_mag{mag}_384_seed3407/metrics.json"
                metrics = read_json(path)
                test = metrics.get("test") if isinstance(metrics, dict) else None
                if not isinstance(test, dict):
                    miss.append(str(path.relative_to(ROOT)))
                    continue
                vals.append(test)
            if miss or not vals:
                missing.append(missing_row("breakhis_binary", label, "supervised", "; ".join(miss)))
                acc = auc = float("nan")
                note = "Missing BreakHis magnification ResNet metrics."
            else:
                acc = float(np.mean([metric_value(v, "accuracy") for v in vals]))
                auc = float(np.mean([metric_value(v, "auc") for v in vals]))
                note = "Simple mean over BreakHis magnifications 40/100/200/400."
            source = "outputs/final/resnet50_breakhis_binary_mag*_384_seed3407/metrics.json"
        rows.append(row("breakhis_binary", label, label, "Fine-tuned", "supervised", acc, auc, color, source, note))


def add_lag_supervised(rows: list[dict], missing: list[dict]) -> None:
    table_path = ROOT / "results/tables/current_experiment_summary_main_table.csv"
    df = pd.read_csv(table_path)
    hit = df[(df["Model"].astype(str) == "ResNet50") & (df["Method"].astype(str) == "test_0p5")]
    if hit.empty:
        missing.append(missing_row("lag", "ResNet50", "Fine-tuned", str(table_path.relative_to(ROOT))))
    else:
        record = hit.iloc[0]
        rows.append(
            row(
                "lag",
                "ResNet50",
                "ResNet50",
                "Fine-tuned",
                "supervised",
                float(record["Acc"]),
                float(record["AUC"]),
                "#EF5350",
                f"{table_path.relative_to(ROOT)}:ResNet50/test_0p5",
                "LAG ResNet50 row comes from curated current experiment table.",
            )
        )
    missing.append(missing_row("lag", "ViT224", "Fine-tuned", "No vit224_lag metrics file found."))


def add_lora_row(dataset_key: str, rows: list[dict], missing: list[dict]) -> None:
    cfg = DATASETS[dataset_key]
    pred_path = cfg["lora_predictions"]
    metrics = compute_lora_metrics(pred_path, cfg["label_names"], cfg["prompt_dataset_name"])
    if metrics is None:
        missing.append(missing_row(dataset_key, "Gemma4 LoRA", "SFT", str(pred_path.relative_to(ROOT))))
        acc = auc = float("nan")
    else:
        acc = metric_value(metrics, "accuracy")
        auc = metric_value(metrics, "auc")
    rows.append(
        row(
            dataset_key,
            "Gemma4 LoRA",
            "Gemma4 LoRA",
            "SFT",
            "lora",
            acc,
            auc,
            "#2E7D32",
            str(pred_path.relative_to(ROOT)),
            cfg["lora_note"],
        )
    )


def compute_lora_metrics(pred_path: Path, label_names: list[str], prompt_dataset_name: str) -> dict | None:
    data = read_json(pred_path)
    if not isinstance(data, list) or not data:
        return None

    parser = OutputParser()
    y_true = []
    y_pred = []
    y_prob = []
    for record in data:
        y_true.append(int(record["label_idx"]))
        y_pred.append(int(record["prediction_idx"]))
        parsed = record.get("parsed")
        if not isinstance(parsed, dict) or not parsed.get("class_probabilities"):
            parsed_obj = parser.parse_classification(
                raw_response=str(record.get("raw_response", "")),
                query_id=str(record.get("query_id", record.get("index", ""))),
                label_names=label_names,
                is_multi_label=False,
                dataset_name=prompt_dataset_name,
            )
            parsed = parsed_obj.to_dict()
        probs = parsed.get("class_probabilities") or []
        if len(probs) != len(label_names):
            probs = fallback_probs(int(record["prediction_idx"]), len(label_names), float(parsed.get("confidence", 0.5)))
        y_prob.append([float(x) for x in probs])

    metrics = ClassificationMetrics()
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    if len(label_names) == 2:
        result = metrics.compute_binary(y_true_arr, y_pred_arr, y_prob_arr[:, 1])
    else:
        result = metrics.compute_multiclass(y_true_arr, y_pred_arr, y_prob_arr, len(label_names))
    return result.to_dict()


def fallback_probs(pred_idx: int, n_classes: int, confidence: float) -> list[float]:
    confidence = max(0.0, min(1.0, confidence))
    if n_classes <= 1:
        return [1.0]
    rest = (1.0 - confidence) / (n_classes - 1)
    probs = [rest] * n_classes
    if 0 <= pred_idx < n_classes:
        probs[pred_idx] = confidence
    return probs


def row(
    dataset_key: str,
    label: str,
    model: str,
    method: str,
    family: str,
    accuracy: float,
    auc: float,
    color: str,
    source: str,
    note: str,
) -> dict:
    return {
        "dataset": dataset_key,
        "dataset_title": DATASETS[dataset_key]["title"],
        "label": label,
        "model": model,
        "method": method,
        "family": family,
        "accuracy": accuracy,
        "auc": auc,
        "color": color,
        "source": source,
        "note": note,
    }


def missing_row(dataset_key: str, model: str, method: str, source: str) -> dict:
    return {
        "dataset": dataset_key,
        "dataset_title": DATASETS[dataset_key]["title"],
        "model": model,
        "method": method,
        "missing_source": source,
    }


def build_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    missing: list[dict] = []
    for dataset_key in DATASETS:
        add_main_rows_for_dataset(dataset_key, rows, missing)
        add_supervised_rows(dataset_key, rows, missing)
    return pd.DataFrame(rows), pd.DataFrame(missing)


def plot_dataset(df: pd.DataFrame, dataset_key: str) -> None:
    sub = df[df["dataset"] == dataset_key].copy()
    output_prefix = OUT_DIR / f"classification_3llm_vit_lora_resnet_{dataset_key}_accuracy_auc_angled"
    plot_accuracy_auc(sub, output_prefix, f"{DATASETS[dataset_key]['title']} test performance")


def plot_dataset_horizontal(df: pd.DataFrame, dataset_key: str) -> None:
    sub = df[df["dataset"] == dataset_key].copy()
    labels = sub["label"].astype(str).tolist()
    y = np.arange(len(sub))
    colors = sub["color"].astype(str).tolist()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.5, max(8.5, 0.42 * len(sub))),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    for ax, metric, title in zip(axes, ["accuracy", "auc"], ["Accuracy", "AUC"]):
        values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        ax.barh(y, np.nan_to_num(values, nan=0.0), color=colors, edgecolor="#4A4A4A", linewidth=0.8)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0.0, 1.08)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.grid(axis="x", alpha=0.24, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.text(
                    min(value + 0.012, 1.055),
                    i,
                    f"{value:.3f}",
                    ha="left",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#4A4A4A",
                )
            else:
                ax.text(0.02, i, "N/A", ha="left", va="center", fontsize=8.5, color="#6A6A6A")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle(f"{DATASETS[dataset_key]['title']} main experiment overview", fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.19, right=0.98, top=0.90, bottom=0.08, wspace=0.08)
    output_prefix = OUT_DIR / f"classification_3llm_vit_lora_resnet_{dataset_key}_accuracy_auc_horizontal"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_accuracy_auc(df: pd.DataFrame, output_prefix: Path, title: str) -> None:
    labels = df["label"].astype(str).tolist()
    x = np.arange(len(df))
    colors = df["color"].astype(str).tolist()

    fig, axes = plt.subplots(2, 1, figsize=(max(11.5, 0.55 * len(df)), 7.6), sharex=True)
    for ax, metric in zip(axes, ["accuracy", "auc"]):
        values = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        plot_vals = np.nan_to_num(values, nan=0.0)
        ax.bar(x, plot_vals, color=colors, edgecolor="#4A4A4A", linewidth=0.6)
        valid = values[np.isfinite(values)]
        if len(valid):
            lo = max(0.0, float(valid.min()) - 0.08)
            hi = min(1.02, float(valid.max()) + 0.08)
            ax.set_ylim(lo, hi)
        else:
            ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("AUC" if metric == "auc" else "Accuracy", fontweight="bold")
        ax.grid(axis="y", alpha=0.24, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        y0, y1 = ax.get_ylim()
        offset = (y1 - y0) * 0.018
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.text(i, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=6.4)
            else:
                ax.text(i, y0 + offset, "N/A", ha="center", va="bottom", fontsize=6.4, color="#6A6A6A")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_combined(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(18, 17.5), sharey=False)
    for row_idx, dataset_key in enumerate(DATASETS):
        sub = df[df["dataset"] == dataset_key].copy()
        labels = sub["label"].astype(str).tolist()
        x = np.arange(len(sub))
        colors = sub["color"].astype(str).tolist()
        for col_idx, metric in enumerate(["accuracy", "auc"]):
            ax = axes[row_idx, col_idx]
            values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            ax.bar(x, np.nan_to_num(values, nan=0.0), color=colors, edgecolor="#4A4A4A", linewidth=0.45)
            valid = values[np.isfinite(values)]
            if len(valid):
                ax.set_ylim(max(0.0, float(valid.min()) - 0.08), min(1.02, float(valid.max()) + 0.08))
            else:
                ax.set_ylim(0.0, 1.0)
            ax.set_title(
                f"{DATASETS[dataset_key]['title']} - {'AUC' if metric == 'auc' else 'Accuracy'}",
                fontsize=10,
                fontweight="bold",
            )
            ax.grid(axis="y", alpha=0.22, linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=5.8)
    fig.suptitle("Classification main experiments + supervised baselines + Gemma4 LoRA", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    prefix = OUT_DIR / "classification_3llm_vit_lora_resnet_accuracy_auc_angled"
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    df, missing = build_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUT_DIR / "classification_3llm_vit_lora_resnet_accuracy_auc.csv"
    missing_path = OUT_DIR / "classification_3llm_vit_lora_resnet_missing_data.csv"
    df.to_csv(table_path, index=False)
    missing.to_csv(missing_path, index=False)
    for dataset_key in DATASETS:
        dataset_csv = OUT_DIR / f"classification_3llm_vit_lora_resnet_{dataset_key}.csv"
        df[df["dataset"] == dataset_key].to_csv(dataset_csv, index=False)
        plot_dataset(df, dataset_key)
        if dataset_key in {"breakhis_binary", "tbx11k", "ddr_512"}:
            plot_dataset_horizontal(df, dataset_key)
    plot_combined(df)
    print(f"Wrote {len(df)} rows to {table_path}")
    print(f"Wrote {len(missing)} missing-data rows to {missing_path}")


if __name__ == "__main__":
    main()
