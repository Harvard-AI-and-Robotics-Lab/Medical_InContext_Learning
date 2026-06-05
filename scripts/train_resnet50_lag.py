#!/usr/bin/env python
import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


LABELS = ["non_glaucoma", "glaucoma"]


@dataclass
class Metrics:
    n: int
    threshold: float
    accuracy: float
    sensitivity: float
    specificity: float
    auc: float
    average_precision: float
    brier: float
    nll: float
    confusion_matrix: list


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
        if ":\\" in raw_path or "://" in raw_path:
            win_path = PureWindowsPath(raw_path)
            return data_root / str(row["class_name"]) / "image" / win_path.name
    return data_root / str(row["class_name"]) / "image" / f"{row['id']}.jpg"


class LAGManifestDataset(Dataset):
    def __init__(self, manifest_csv: Path, data_root: Path, split: str, transform):
        df = pd.read_csv(manifest_csv, dtype={"id": str})
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
            "label": torch.tensor(label, dtype=torch.float32),
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
                [transforms.RandomAffine(degrees=12, translate=(0.03, 0.03), scale=(0.95, 1.05))],
                p=0.6,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01),
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


def build_model(backbone_lr: float, head_lr: float, weight_decay: float):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
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
            logits = model(images).squeeze(1)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
            ids.extend(batch["id"])
            paths.extend(batch["image_path"])
            labels.extend(batch["label"].numpy().astype(int).tolist())
            probs.extend(batch_probs.tolist())
    return ids, paths, np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)


def metrics_from_probs(y_true, probs, threshold: float) -> Metrics:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    eps_probs = np.clip(probs, 1e-6, 1 - 1e-6)
    return Metrics(
        n=int(len(y_true)),
        threshold=float(threshold),
        accuracy=float((preds == y_true).mean()),
        sensitivity=float(tp / (tp + fn)) if (tp + fn) else math.nan,
        specificity=float(tn / (tn + fp)) if (tn + fp) else math.nan,
        auc=float(roc_auc_score(y_true, probs)),
        average_precision=float(average_precision_score(y_true, probs)),
        brier=float(brier_score_loss(y_true, probs)),
        nll=float(log_loss(y_true, eps_probs, labels=[0, 1])),
        confusion_matrix=cm.tolist(),
    )


def select_youden_threshold(y_true, probs):
    candidates = np.unique(np.concatenate([[0.5], probs]))
    best_threshold = 0.5
    best_score = -float("inf")
    for threshold in candidates:
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        score = sens + spec - 1.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def save_predictions(path: Path, ids, image_paths, y_true, probs, threshold: float) -> None:
    rows = []
    preds = (probs >= threshold).astype(int)
    for sample_id, image_path, label, prob, pred in zip(ids, image_paths, y_true, probs, preds):
        rows.append(
            {
                "id": sample_id,
                "image_path": image_path,
                "ground_truth_label": int(label),
                "ground_truth_name": LABELS[int(label)],
                "probability": float(prob),
                "predicted_label_idx": int(pred),
                "predicted_label_name": LABELS[int(pred)],
                "threshold": float(threshold),
            }
        )
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    train_tf, eval_tf = build_transforms(args.image_size)
    train_ds = LAGManifestDataset(args.manifest_csv, args.data_root, "train", train_tf)
    val_ds = LAGManifestDataset(args.manifest_csv, args.data_root, "val", eval_tf)
    test_ds = LAGManifestDataset(args.manifest_csv, args.data_root, "test", eval_tf)

    train_labels = train_ds.rows["label"].astype(int).to_numpy()
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

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
    model, optimizer = build_model(args.backbone_lr, args.head_lr, args.weight_decay)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_auc = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    history = []
    best_path = args.output_dir / "best_resnet50.pt"

    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "train": len(train_ds),
                "val": len(val_ds),
                "test": len(test_ds),
                "pos_weight": float(pos_weight.item()),
                "image_size": args.image_size,
                "batch_size": args.batch_size,
            }
        ),
        flush=True,
    )

    if args.eval_only:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        val_ids, val_paths, val_y, val_probs = predict(model, val_loader, device)
        threshold = float(checkpoint["val_threshold_youden"])
        test_ids, test_paths, test_y, test_probs = predict(model, test_loader, device)
        final_metrics = {
            "best_epoch": int(checkpoint["epoch"]),
            "val_threshold_youden": threshold,
            "val_0p5": asdict(metrics_from_probs(val_y, val_probs, 0.5)),
            "val_youden": asdict(metrics_from_probs(val_y, val_probs, threshold)),
            "test_0p5": asdict(metrics_from_probs(test_y, test_probs, 0.5)),
            "test_youden": asdict(metrics_from_probs(test_y, test_probs, threshold)),
        }
        (args.output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
        save_predictions(args.output_dir / "val_predictions.json", val_ids, val_paths, val_y, val_probs, threshold)
        save_predictions(args.output_dir / "test_predictions.json", test_ids, test_paths, test_y, test_probs, threshold)
        print(json.dumps({"event": "eval_done", "metrics": final_metrics}), flush=True)
        return

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        val_ids, val_paths, val_y, val_probs = predict(model, val_loader, device)
        val_metrics_05 = metrics_from_probs(val_y, val_probs, 0.5)
        val_threshold = select_youden_threshold(val_y, val_probs)
        val_metrics_youden = metrics_from_probs(val_y, val_probs, val_threshold)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_0p5": asdict(val_metrics_05),
            "val_youden": asdict(val_metrics_youden),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if val_metrics_05.auc > best_auc:
            best_auc = val_metrics_05.auc
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "val_threshold_youden": val_threshold,
                    "args": vars(args),
                },
                best_path,
            )
            save_predictions(args.output_dir / "val_predictions_best.json", val_ids, val_paths, val_y, val_probs, val_threshold)
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print(json.dumps({"event": "early_stop", "epoch": epoch, "best_epoch": best_epoch, "best_auc": best_auc}), flush=True)
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_ids, val_paths, val_y, val_probs = predict(model, val_loader, device)
    threshold = float(checkpoint["val_threshold_youden"])
    test_ids, test_paths, test_y, test_probs = predict(model, test_loader, device)

    final_metrics = {
        "best_epoch": int(checkpoint["epoch"]),
        "val_threshold_youden": threshold,
        "val_0p5": asdict(metrics_from_probs(val_y, val_probs, 0.5)),
        "val_youden": asdict(metrics_from_probs(val_y, val_probs, threshold)),
        "test_0p5": asdict(metrics_from_probs(test_y, test_probs, 0.5)),
        "test_youden": asdict(metrics_from_probs(test_y, test_probs, threshold)),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    save_predictions(args.output_dir / "val_predictions.json", val_ids, val_paths, val_y, val_probs, threshold)
    save_predictions(args.output_dir / "test_predictions.json", test_ids, test_paths, test_y, test_probs, threshold)
    print(json.dumps({"event": "done", "metrics": final_metrics}), flush=True)


if __name__ == "__main__":
    main()
