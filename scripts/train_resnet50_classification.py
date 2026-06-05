#!/usr/bin/env python
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
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


@dataclass
class Metrics:
    n: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_auc: float
    brier: float
    nll: float
    confusion_matrix: list


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def infer_labels(df: pd.DataFrame) -> list[str]:
    label_rows = (
        df[["label", "label_name"]]
        .drop_duplicates()
        .sort_values("label")
        .reset_index(drop=True)
    )
    return [str(x) for x in label_rows["label_name"].tolist()]


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


class ManifestClassificationDataset(Dataset):
    def __init__(self, manifest_csv: Path, data_root: Path, split: str, transform):
        df = pd.read_csv(manifest_csv, dtype={"id": str, "patient_id": str})
        df = df[df["split"] == split].copy().reset_index(drop=True)
        self.rows = df
        self.data_root = data_root
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        image_path = resolve_image_path(row, self.data_root)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "id": str(row["id"]),
            "image_path": str(image_path),
        }


def build_transforms(image_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=10, translate=(0.03, 0.03), scale=(0.95, 1.05))],
                p=0.5,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, eval_tf


def build_model(n_classes: int, backbone_lr: float, head_lr: float, weight_decay: float):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    head_params = list(model.fc.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    return model, optimizer


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "id": [b["id"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }


def predict(model, loader, device):
    model.eval()
    ids, paths, labels, probs = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            batch_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            ids.extend(batch["id"])
            paths.extend(batch["image_path"])
            labels.extend(batch["label"].numpy().astype(int).tolist())
            probs.extend(batch_probs.tolist())
    return ids, paths, np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)


def metrics_from_probs(y_true, probs, labels: list[str]) -> Metrics:
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
    return Metrics(
        n=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, preds)),
        macro_f1=float(f1_score(y_true, preds, average="macro", labels=list(range(n_classes)), zero_division=0)),
        weighted_f1=float(f1_score(y_true, preds, average="weighted", labels=list(range(n_classes)), zero_division=0)),
        macro_auc=macro_auc,
        brier=float(np.mean(np.sum((probs - y_true_onehot) ** 2, axis=1))),
        nll=float(log_loss(y_true, eps_probs, labels=list(range(n_classes)))),
        confusion_matrix=confusion_matrix(y_true, preds, labels=list(range(n_classes))).tolist(),
    )


def save_predictions(path: Path, ids, image_paths, y_true, probs, labels: list[str]) -> None:
    rows = []
    preds = probs.argmax(axis=1)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selection-metric", choices=["accuracy", "macro_auc", "macro_f1"], default="accuracy")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    manifest_df = pd.read_csv(args.manifest_csv)
    labels = infer_labels(manifest_df)
    n_classes = len(labels)
    (args.output_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    train_tf, eval_tf = build_transforms(args.image_size)
    train_ds = ManifestClassificationDataset(args.manifest_csv, args.data_root, "train", train_tf)
    val_ds = ManifestClassificationDataset(args.manifest_csv, args.data_root, "val", eval_tf)
    test_ds = ManifestClassificationDataset(args.manifest_csv, args.data_root, "test", eval_tf)

    class_counts = train_ds.rows["label"].astype(int).value_counts().reindex(range(n_classes), fill_value=0).to_numpy()
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()

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
    model, optimizer = build_model(n_classes, args.backbone_lr, args.head_lr, args.weight_decay)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    history = []
    best_path = args.output_dir / "best_resnet50.pt"

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(images)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))

            val_ids, val_paths, val_true, val_probs = predict(model, val_loader, device)
            val_metrics = metrics_from_probs(val_true, val_probs, labels)
            score = getattr(val_metrics, args.selection_metric)
            if math.isnan(score):
                score = val_metrics.accuracy
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else math.nan,
                "val": asdict(val_metrics),
            }
            history.append(row)
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(
                f"epoch={epoch} loss={row['train_loss']:.4f} "
                f"val_acc={val_metrics.accuracy:.4f} val_macro_f1={val_metrics.macro_f1:.4f} "
                f"val_macro_auc={val_metrics.macro_auc}",
                flush=True,
            )
            if score > best_score:
                best_score = float(score)
                best_epoch = epoch
                stale_epochs = 0
                torch.save({"model": model.state_dict(), "labels": labels, "epoch": epoch}, best_path)
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
                    break

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        best_epoch = int(checkpoint.get("epoch", best_epoch))

    val_ids, val_paths, val_true, val_probs = predict(model, val_loader, device)
    test_ids, test_paths, test_true, test_probs = predict(model, test_loader, device)
    val_metrics = metrics_from_probs(val_true, val_probs, labels)
    test_metrics = metrics_from_probs(test_true, test_probs, labels)

    save_predictions(args.output_dir / "val_predictions.json", val_ids, val_paths, val_true, val_probs, labels)
    save_predictions(args.output_dir / "test_predictions.json", test_ids, test_paths, test_true, test_probs, labels)

    metrics = {
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "val": asdict(val_metrics),
        "test": asdict(test_metrics),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
