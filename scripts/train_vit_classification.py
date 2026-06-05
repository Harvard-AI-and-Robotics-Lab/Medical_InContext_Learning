#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification


@dataclass
class SingleLabelMetrics:
    n: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_auc: float
    quadratic_weighted_kappa: float
    brier: float
    nll: float
    confusion_matrix: list


@dataclass
class MultiLabelMetrics:
    n: int
    macro_auc: float
    auc_per_label: dict
    mean_label_accuracy: float
    micro_f1: float
    macro_f1: float
    brier: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_image_path(row: pd.Series, data_root: Path) -> Path:
    raw_path = "" if pd.isna(row.get("image_path", "")) else str(row.get("image_path", ""))
    if raw_path:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate
        candidate = data_root / raw_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve image path for row id={row.get('id')}: {raw_path}")


def infer_single_labels(df: pd.DataFrame) -> list[str]:
    label_rows = (
        df[["label", "label_name"]]
        .drop_duplicates()
        .sort_values("label")
        .reset_index(drop=True)
    )
    return [str(x) for x in label_rows["label_name"].tolist()]


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        data_root: Path,
        split: str,
        transform,
        task: str,
        target_labels: list[str] | None = None,
    ):
        df = pd.read_csv(manifest_csv, dtype={"id": str, "patient_id": str})
        self.rows = df[df["split"] == split].copy().reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.task = task
        self.target_labels = target_labels or []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        image_path = resolve_image_path(row, self.data_root)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.task == "multilabel":
            label_arr = row[self.target_labels].astype(float).to_numpy(dtype=np.float32)
            label = torch.tensor(label_arr, dtype=torch.float32)
        else:
            label = torch.tensor(int(row["label"]), dtype=torch.long)
        return {
            "image": image,
            "label": label,
            "id": str(row["id"]),
            "image_path": str(image_path),
        }


def processor_mean_std(processor):
    mean = getattr(processor, "image_mean", None) or [0.5, 0.5, 0.5]
    std = getattr(processor, "image_std", None) or [0.5, 0.5, 0.5]
    return list(mean), list(std)


def build_transforms(processor, image_size: int, augment: bool):
    mean, std = processor_mean_std(processor)
    train_steps = [
        transforms.Resize((image_size, image_size), antialias=True),
    ]
    if augment:
        train_steps.extend(
            [
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=10, translate=(0.03, 0.03), scale=(0.95, 1.05))],
                    p=0.5,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06, hue=0.01),
            ]
        )
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(train_steps), eval_tf


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "id": [b["id"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }


def build_model_and_optimizer(
    model_name: str,
    labels: list[str],
    task: str,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
):
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    problem_type = "multi_label_classification" if task == "multilabel" else "single_label_classification"
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        problem_type=problem_type,
        ignore_mismatched_sizes=True,
    )
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if name.startswith("classifier."):
            head_params.append(param)
        else:
            backbone_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    return model, optimizer


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def predict_single(model, loader, device):
    model.eval()
    ids, paths, labels, probs = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(pixel_values=images).logits
            batch_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            ids.extend(batch["id"])
            paths.extend(batch["image_path"])
            labels.extend(batch["label"].numpy().astype(int).tolist())
            probs.extend(batch_probs.tolist())
    return ids, paths, np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)


def predict_multilabel(model, loader, device):
    model.eval()
    ids, paths, labels, probs = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(pixel_values=images).logits
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
            ids.extend(batch["id"])
            paths.extend(batch["image_path"])
            labels.extend(batch["label"].numpy().tolist())
            probs.extend(batch_probs.tolist())
    return ids, paths, np.asarray(labels, dtype=float), np.asarray(probs, dtype=float)


def single_metrics_from_probs(y_true, probs, labels: list[str]) -> SingleLabelMetrics:
    n_classes = len(labels)
    preds = probs.argmax(axis=1)
    y_true_onehot = np.zeros((len(y_true), n_classes), dtype=float)
    for idx, label in enumerate(y_true):
        y_true_onehot[idx, int(label)] = 1.0
    aucs = []
    for class_idx in range(n_classes):
        y_bin = (np.asarray(y_true) == class_idx).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        aucs.append(float(roc_auc_score(y_bin, probs[:, class_idx])))
    macro_auc = float(np.mean(aucs)) if aucs else math.nan
    eps_probs = np.clip(probs, 1e-7, 1.0)
    eps_probs = eps_probs / eps_probs.sum(axis=1, keepdims=True)
    qwk = float(cohen_kappa_score(y_true, preds, weights="quadratic")) if n_classes > 2 else math.nan
    return SingleLabelMetrics(
        n=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, preds)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, preds)),
        macro_f1=float(f1_score(y_true, preds, average="macro", labels=list(range(n_classes)), zero_division=0)),
        weighted_f1=float(f1_score(y_true, preds, average="weighted", labels=list(range(n_classes)), zero_division=0)),
        macro_auc=macro_auc,
        quadratic_weighted_kappa=qwk,
        brier=float(np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1))),
        nll=float(log_loss(y_true, eps_probs, labels=list(range(n_classes)))),
        confusion_matrix=confusion_matrix(y_true, preds, labels=list(range(n_classes))).tolist(),
    )


def multilabel_metrics_from_probs(y_true, probs, labels: list[str]) -> MultiLabelMetrics:
    preds = (probs >= 0.5).astype(int)
    aucs = {}
    vals = []
    for idx, label in enumerate(labels):
        y = y_true[:, idx]
        if len(np.unique(y)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y, probs[:, idx]))
            vals.append(auc)
        aucs[label] = auc
    return MultiLabelMetrics(
        n=int(len(y_true)),
        macro_auc=float(np.mean(vals)) if vals else math.nan,
        auc_per_label=aucs,
        mean_label_accuracy=float(np.mean([accuracy_score(y_true[:, i], preds[:, i]) for i in range(len(labels))])),
        micro_f1=float(f1_score(y_true.reshape(-1), preds.reshape(-1), average="micro", zero_division=0)),
        macro_f1=float(f1_score(y_true, preds, average="macro", zero_division=0)),
        brier=float(np.mean((probs - y_true) ** 2)),
    )


def save_single_predictions(path: Path, ids, image_paths, y_true, probs, labels: list[str]) -> None:
    preds = probs.argmax(axis=1)
    rows = []
    for sample_id, image_path, label, prob_vec, pred in zip(ids, image_paths, y_true, probs, preds):
        rows.append(
            {
                "id": sample_id,
                "image_path": image_path,
                "ground_truth_label": int(label),
                "ground_truth_name": labels[int(label)],
                "predicted_label_idx": int(pred),
                "predicted_label_name": labels[int(pred)],
                "confidence": float(prob_vec[int(pred)]),
                "class_probabilities": {name: float(prob_vec[idx]) for idx, name in enumerate(labels)},
            }
        )
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def save_multilabel_predictions(path: Path, ids, image_paths, y_true, probs, labels: list[str]) -> None:
    preds = (probs >= 0.5).astype(int)
    rows = []
    for sample_id, image_path, y, p, d in zip(ids, image_paths, y_true, probs, preds):
        rows.append(
            {
                "id": sample_id,
                "image_path": image_path,
                "ground_truth": {label: int(y[idx]) for idx, label in enumerate(labels)},
                "predicted": {label: int(d[idx]) for idx, label in enumerate(labels)},
                "probabilities": {label: float(p[idx]) for idx, label in enumerate(labels)},
                "threshold": 0.5,
            }
        )
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="google/vit-base-patch16-224")
    parser.add_argument("--task", choices=["single", "multilabel"], default="single")
    parser.add_argument("--target-labels", default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--selection-metric", default="accuracy")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    manifest_df = pd.read_csv(args.manifest_csv)
    if args.task == "multilabel":
        labels = [x.strip() for x in args.target_labels.split(",") if x.strip()]
        if not labels:
            raise ValueError("--target-labels is required for multilabel training")
        missing = [label for label in labels if label not in manifest_df.columns]
        if missing:
            raise ValueError(f"Missing target label columns in manifest: {missing}")
    else:
        labels = infer_single_labels(manifest_df)
    (args.output_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    train_tf, eval_tf = build_transforms(processor, args.image_size, augment=not args.no_augment)
    task_key = "multilabel" if args.task == "multilabel" else "single"
    train_ds = ManifestDataset(args.manifest_csv, args.data_root, "train", train_tf, task_key, labels)
    val_ds = ManifestDataset(args.manifest_csv, args.data_root, "val", eval_tf, task_key, labels)
    test_ds = ManifestDataset(args.manifest_csv, args.data_root, "test", eval_tf, task_key, labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer = build_model_and_optimizer(
        args.model_name,
        labels,
        task_key,
        args.backbone_lr,
        args.head_lr,
        args.weight_decay,
    )
    model.to(device)

    if args.task == "multilabel":
        pos = train_ds.rows[labels].astype(float).sum(axis=0).to_numpy(dtype=np.float32)
        neg = len(train_ds) - pos
        pos_weight = np.clip(neg / np.maximum(pos, 1.0), 1.0, 20.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    else:
        n_classes = len(labels)
        class_counts = train_ds.rows["label"].astype(int).value_counts().reindex(range(n_classes), fill_value=0).to_numpy()
        class_weights = class_counts.sum() / np.maximum(class_counts, 1)
        class_weights = class_weights / class_weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = build_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    history = []
    best_path = args.output_dir / "best_vit.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(pixel_values=images).logits
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))

        if args.task == "multilabel":
            val_ids, val_paths, val_true, val_probs = predict_multilabel(model, val_loader, device)
            val_metrics = multilabel_metrics_from_probs(val_true, val_probs, labels)
        else:
            val_ids, val_paths, val_true, val_probs = predict_single(model, val_loader, device)
            val_metrics = single_metrics_from_probs(val_true, val_probs, labels)

        score = getattr(val_metrics, args.selection_metric)
        if math.isnan(float(score)):
            score = getattr(val_metrics, "accuracy", getattr(val_metrics, "macro_f1", 0.0))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else math.nan,
            "lr_backbone": float(optimizer.param_groups[0]["lr"]),
            "lr_head": float(optimizer.param_groups[1]["lr"]),
            "val": asdict(val_metrics),
        }
        history.append(row)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"epoch={epoch} loss={row['train_loss']:.4f} "
            f"selection_{args.selection_metric}={float(score):.4f}",
            flush=True,
        )
        if float(score) > best_score:
            best_score = float(score)
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "labels": labels, "epoch": epoch}, best_path)
        else:
            stale_epochs += 1
            if epoch >= args.min_epochs and stale_epochs >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
                break

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        best_epoch = int(checkpoint.get("epoch", best_epoch))

    if args.task == "multilabel":
        val_ids, val_paths, val_true, val_probs = predict_multilabel(model, val_loader, device)
        test_ids, test_paths, test_true, test_probs = predict_multilabel(model, test_loader, device)
        val_metrics = multilabel_metrics_from_probs(val_true, val_probs, labels)
        test_metrics = multilabel_metrics_from_probs(test_true, test_probs, labels)
        save_multilabel_predictions(args.output_dir / "val_predictions.json", val_ids, val_paths, val_true, val_probs, labels)
        save_multilabel_predictions(args.output_dir / "test_predictions.json", test_ids, test_paths, test_true, test_probs, labels)
    else:
        val_ids, val_paths, val_true, val_probs = predict_single(model, val_loader, device)
        test_ids, test_paths, test_true, test_probs = predict_single(model, test_loader, device)
        val_metrics = single_metrics_from_probs(val_true, val_probs, labels)
        test_metrics = single_metrics_from_probs(test_true, test_probs, labels)
        save_single_predictions(args.output_dir / "val_predictions.json", val_ids, val_paths, val_true, val_probs, labels)
        save_single_predictions(args.output_dir / "test_predictions.json", test_ids, test_paths, test_true, test_probs, labels)

    metrics = {
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "labels": labels,
        "val": asdict(val_metrics),
        "test": asdict(test_metrics),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
