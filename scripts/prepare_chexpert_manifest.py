#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

LABELS = [
    "no_finding",
    "enlarged_cardiomediastinum",
    "cardiomegaly",
    "lung_opacity",
    "lung_lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural_effusion",
    "pleural_other",
    "fracture",
    "support_devices",
]

COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def parse_label(value) -> int:
    if pd.isna(value):
        return 0
    try:
        return 1 if float(value) == 1.0 else 0
    except (TypeError, ValueError):
        return 0


def clean_id(path: str, split: str) -> str:
    stem = path
    for prefix in ["CheXpert-v1.0-small/", "CheXpert-v1.0/", "CheXpert/"]:
        stem = stem.replace(prefix, "")
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return f"chexpert_{split}_{stem}"


def parse_patient_study_view(path: str) -> tuple[str, str, str]:
    parts = Path(path).parts
    patient = next((x for x in parts if x.startswith("patient")), "")
    study = next((x for x in parts if x.startswith("study")), "")
    view = Path(path).stem
    return patient, study, view


def labels_from_row(row: pd.Series) -> dict[str, int]:
    return {label: parse_label(row.get(col, 0)) for label, col in zip(LABELS, COLS)}


def add_kaggle_rows(rows: list[dict], csv_path: Path, kaggle_root: Path, split: str) -> None:
    df = pd.read_csv(csv_path)
    for row in df.to_dict("records"):
        src_path = str(row["Path"])
        rel = src_path.replace("CheXpert-v1.0-small/", "")
        image_path = kaggle_root / rel
        if not image_path.exists():
            raise FileNotFoundError(f"missing Kaggle image: {image_path}")
        patient, study, view = parse_patient_study_view(rel)
        out = {
            "id": clean_id(src_path, split),
            "split": split,
            "image_path": str(image_path),
            "source_path": src_path,
            "patient_id": patient,
            "study_id": study,
            "view": view,
            "frontal_lateral": row.get("Frontal/Lateral", ""),
            "ap_pa": row.get("AP/PA", ""),
        }
        out.update(labels_from_row(pd.Series(row)))
        rows.append(out)


def add_test_rows(rows: list[dict], labels_csv: Path, test_root: Path) -> None:
    labels = pd.read_csv(labels_csv)
    by_study = {str(row["Study"]): row for row in labels.to_dict("records")}
    image_paths = sorted((test_root / "CheXpert" / "test").glob("patient*/study*/*.jpg"))
    missing_labels = []
    for image_path in image_paths:
        rel = image_path.relative_to(test_root).as_posix()
        parts = image_path.parts
        patient = next(x for x in parts if x.startswith("patient"))
        study = next(x for x in parts if x.startswith("study"))
        study_key = f"CheXpert-v1.0/test/{patient}/{study}"
        row = by_study.get(study_key)
        if row is None:
            missing_labels.append(study_key)
            continue
        out = {
            "id": clean_id(rel, "test"),
            "split": "test",
            "image_path": str(image_path),
            "source_path": study_key,
            "patient_id": patient,
            "study_id": study,
            "view": image_path.stem,
            "frontal_lateral": "Frontal" if "frontal" in image_path.stem else "Lateral",
            "ap_pa": "",
        }
        out.update(labels_from_row(pd.Series(row)))
        rows.append(out)
    if missing_labels:
        raise RuntimeError(f"missing labels for {len(set(missing_labels))} studies, e.g. {missing_labels[:5]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-root", type=Path, default=Path("data/raw/chexpert_kaggle"))
    ap.add_argument("--test-root", type=Path, default=Path("data/raw/chexlocalize_azure"))
    ap.add_argument("--test-labels", type=Path, default=Path("data/raw/chexpert_test_labels/groundtruth.csv"))
    ap.add_argument("--output", type=Path, default=Path("manifests/chexpert_official_split.csv"))
    args = ap.parse_args()

    rows: list[dict] = []
    add_kaggle_rows(rows, args.kaggle_root / "train.csv", args.kaggle_root, "train")
    add_kaggle_rows(rows, args.kaggle_root / "valid.csv", args.kaggle_root, "val")
    add_test_rows(rows, args.test_labels, args.test_root)

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows to {args.output}")
    print(df["split"].value_counts().to_string())
    print("test patients", df[df["split"] == "test"]["patient_id"].nunique())
    print("test studies", df[df["split"] == "test"][["patient_id", "study_id"]].drop_duplicates().shape[0])
    print("test images", (df["split"] == "test").sum())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
