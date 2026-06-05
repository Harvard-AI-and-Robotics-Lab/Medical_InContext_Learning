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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

LABELS = [
    "no_finding", "enlarged_cardiomediastinum", "cardiomegaly", "lung_opacity", "lung_lesion",
    "edema", "consolidation", "pneumonia", "atelectasis", "pneumothorax", "pleural_effusion",
    "pleural_other", "fracture", "support_devices",
]

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
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_image_path(row: pd.Series, data_root: Path) -> Path:
    p = Path(str(row["image_path"]))
    if p.exists(): return p
    q = data_root / p
    if q.exists(): return q
    raise FileNotFoundError(f"missing image for {row.get('id')}: {p}")


class CheXpertManifestDataset(Dataset):
    def __init__(self, manifest_csv: Path, data_root: Path, split: str, transform):
        df = pd.read_csv(manifest_csv, dtype={"id": str, "patient_id": str})
        self.rows = df[df["split"] == split].copy().reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        image_path = resolve_image_path(row, self.data_root)
        image = Image.open(image_path).convert("RGB")
        if self.transform: image = self.transform(image)
        labels = row[LABELS].astype(float).to_numpy(dtype=np.float32)
        return {"image": image, "label": torch.tensor(labels), "id": str(row["id"]), "image_path": str(image_path)}


def build_transforms(image_size: int):
    mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    train_tf=transforms.Compose([
        transforms.Resize((image_size,image_size), antialias=True),
        transforms.RandomApply([transforms.RandomAffine(degrees=7, translate=(0.02,0.02), scale=(0.97,1.03))], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), transforms.Normalize(mean,std),
    ])
    eval_tf=transforms.Compose([transforms.Resize((image_size,image_size), antialias=True), transforms.ToTensor(), transforms.Normalize(mean,std)])
    return train_tf, eval_tf


def collate(batch):
    return {"image": torch.stack([b["image"] for b in batch]), "label": torch.stack([b["label"] for b in batch]), "id": [b["id"] for b in batch], "image_path": [b["image_path"] for b in batch]}


def build_model(n_labels: int, backbone_lr: float, head_lr: float, weight_decay: float):
    model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features=model.fc.in_features
    model.fc=nn.Linear(in_features, n_labels)
    backbone=[p for n,p in model.named_parameters() if not n.startswith("fc.")]
    head=list(model.fc.parameters())
    opt=torch.optim.AdamW([{"params": backbone, "lr": backbone_lr}, {"params": head, "lr": head_lr}], weight_decay=weight_decay)
    return model,opt


def predict(model, loader, device):
    model.eval(); ids=[]; paths=[]; ys=[]; probs=[]
    with torch.inference_mode():
        for batch in loader:
            images=batch["image"].to(device, non_blocking=True)
            logits=model(images)
            p=torch.sigmoid(logits).detach().cpu().numpy()
            ids.extend(batch["id"]); paths.extend(batch["image_path"]); ys.extend(batch["label"].numpy().tolist()); probs.extend(p.tolist())
    return ids, paths, np.asarray(ys, dtype=float), np.asarray(probs, dtype=float)


def metrics_from_probs(y_true, probs) -> MultiLabelMetrics:
    preds=(probs >= 0.5).astype(int)
    aucs={}
    vals=[]
    for idx,label in enumerate(LABELS):
        y=y_true[:,idx]
        if len(np.unique(y)) < 2:
            auc=float("nan")
        else:
            auc=float(roc_auc_score(y, probs[:,idx]))
            vals.append(auc)
        aucs[label]=auc
    return MultiLabelMetrics(
        n=int(len(y_true)),
        macro_auc=float(np.mean(vals)) if vals else math.nan,
        auc_per_label=aucs,
        mean_label_accuracy=float(np.mean([accuracy_score(y_true[:,i], preds[:,i]) for i in range(len(LABELS))])),
        micro_f1=float(f1_score(y_true.reshape(-1), preds.reshape(-1), average="micro", zero_division=0)),
        macro_f1=float(f1_score(y_true, preds, average="macro", zero_division=0)),
        brier=float(np.mean((probs-y_true)**2)),
    )


def save_predictions(path: Path, ids, image_paths, y_true, probs):
    preds=(probs>=0.5).astype(int)
    rows=[]
    for sid,img,y,p,d in zip(ids,image_paths,y_true,probs,preds):
        rows.append({"id": sid, "image_path": img, "ground_truth": {l:int(y[i]) for i,l in enumerate(LABELS)}, "predicted": {l:int(d[i]) for i,l in enumerate(LABELS)}, "probabilities": {l:float(p[i]) for i,l in enumerate(LABELS)}})
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("."))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--backbone-lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    args=ap.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    train_tf, eval_tf=build_transforms(args.image_size)
    train_ds=CheXpertManifestDataset(args.manifest_csv,args.data_root,"train",train_tf)
    val_ds=CheXpertManifestDataset(args.manifest_csv,args.data_root,"val",eval_tf)
    test_ds=CheXpertManifestDataset(args.manifest_csv,args.data_root,"test",eval_tf)
    pos=train_ds.rows[LABELS].astype(float).sum(axis=0).to_numpy(dtype=np.float32)
    neg=len(train_ds)-pos
    pos_weight=np.clip(neg/np.maximum(pos,1.0), 1.0, 20.0)
    loaders={
        "train": DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0,collate_fn=collate),
        "val": DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0,collate_fn=collate),
        "test": DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,persistent_workers=args.num_workers>0,collate_fn=collate),
    }
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,opt=build_model(len(LABELS), args.backbone_lr, args.head_lr, args.weight_decay)
    model.to(device)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    scaler=torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    best=-float("inf"); best_epoch=-1; stale=0; hist=[]; best_path=args.output_dir/"best_resnet50.pt"
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch in loaders["train"]:
            images=batch["image"].to(device, non_blocking=True); targets=batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type=="cuda"):
                loss=criterion(model(images), targets)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        _,_,vy,vp=predict(model, loaders["val"], device)
        vm=metrics_from_probs(vy,vp)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": asdict(vm)})
        (args.output_dir/"history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
        print(f"epoch={epoch} loss={np.mean(losses):.4f} val_macro_auc={vm.macro_auc:.4f} val_macro_f1={vm.macro_f1:.4f}", flush=True)
        score=vm.macro_auc if not math.isnan(vm.macro_auc) else vm.macro_f1
        if score > best:
            best=float(score); best_epoch=epoch; stale=0; torch.save({"model":model.state_dict(),"epoch":epoch,"labels":LABELS}, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True); break
    ckpt=torch.load(best_path, map_location=device); model.load_state_dict(ckpt["model"]); best_epoch=int(ckpt.get("epoch",best_epoch))
    vi,vp_paths,vy,vp=predict(model, loaders["val"], device)
    ti,tp_paths,ty,tp=predict(model, loaders["test"], device)
    vm=metrics_from_probs(vy,vp); tm=metrics_from_probs(ty,tp)
    save_predictions(args.output_dir/"val_predictions.json", vi, vp_paths, vy, vp)
    save_predictions(args.output_dir/"test_predictions.json", ti, tp_paths, ty, tp)
    out={"best_epoch": best_epoch, "selection_metric":"macro_auc", "labels": LABELS, "val": asdict(vm), "test": asdict(tm)}
    (args.output_dir/"metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)

if __name__ == "__main__":
    main()
