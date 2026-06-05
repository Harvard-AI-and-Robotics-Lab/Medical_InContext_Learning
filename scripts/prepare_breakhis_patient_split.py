import argparse
import json
import os
import random
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


LABELS = [
    "adenosis",
    "fibroadenoma",
    "phyllodes_tumor",
    "tubular_adenoma",
    "ductal_carcinoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
]

CODE_TO_LABEL = {
    ("B", "A"): "adenosis",
    ("B", "F"): "fibroadenoma",
    ("B", "PT"): "phyllodes_tumor",
    ("B", "TA"): "tubular_adenoma",
    ("M", "DC"): "ductal_carcinoma",
    ("M", "LC"): "lobular_carcinoma",
    ("M", "MC"): "mucinous_carcinoma",
    ("M", "PC"): "papillary_carcinoma",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FILENAME_RE = re.compile(r"^SOB_([BM])_([A-Z]+)-(.+)-(\d+)-(\d+)$", re.IGNORECASE)


def strict_patient_code(raw_case_id: str) -> str:
    """Return the BreaKHis patient code shared across related lesion suffixes.

    BreaKHis filenames can use suffixes such as 14-21998AB, 14-21998CD, and
    14-21998EF for related cases. For leakage-free patient-level splitting, all
    suffix variants must stay in the same split.
    """

    return re.sub(r"[A-Za-z]+$", "", str(raw_case_id))


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        output_root = output_dir.resolve()
        members = tar.getmembers()
        for member in members:
            target = (output_dir / member.name).resolve()
            if not str(target).startswith(str(output_root)):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(output_dir)


def parse_breakhis_image(path: Path, root: Path) -> dict | None:
    if path.suffix.lower() not in IMAGE_EXTS:
        return None
    match = FILENAME_RE.match(path.stem)
    if not match:
        return None
    binary_code = match.group(1).upper()
    tumor_code = match.group(2).upper()
    raw_case_id = match.group(3)
    patient_id = strict_patient_code(raw_case_id)
    magnification = match.group(4)
    label_name = CODE_TO_LABEL.get((binary_code, tumor_code))
    if label_name is None:
        return None
    label = LABELS.index(label_name)
    rel_path = path.relative_to(root)
    return {
        "id": path.stem,
        "patient_id": patient_id,
        "raw_case_id": raw_case_id,
        "binary_label": "benign" if binary_code == "B" else "malignant",
        "tumor_code": tumor_code,
        "tumor_type": label_name,
        "magnification": magnification,
        "label": label,
        "label_name": label_name,
        "image_path": str(rel_path),
    }


def scan_images(root: Path) -> list[dict]:
    rows = []
    seen = set()
    for path in sorted(root.rglob("*")):
        parsed = parse_breakhis_image(path, root)
        if parsed is None:
            continue
        if parsed["id"] in seen:
            raise ValueError(f"Duplicate image id found: {parsed['id']}")
        seen.add(parsed["id"])
        rows.append(parsed)
    if not rows:
        raise ValueError(f"No BreaKHis images found under {root}")
    return rows


def patient_major_label(rows: list[dict]) -> dict[str, int]:
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)
    patient_labels = {}
    for patient_id, patient_rows in by_patient.items():
        counts = Counter(r["label"] for r in patient_rows)
        patient_labels[patient_id] = counts.most_common(1)[0][0]
    return patient_labels


def patient_label_records(rows: list[dict]) -> dict[str, dict]:
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)

    records = {}
    for patient_id, patient_rows in by_patient.items():
        label_counts = Counter(r["label"] for r in patient_rows)
        records[patient_id] = {
            "labels": set(label_counts.keys()),
            "label_counts": label_counts,
            "n_images": len(patient_rows),
        }
    return records


def multilabel_split_score(
    assignment: dict[str, str],
    patient_records: dict[str, dict],
    required_labels: dict[str, set[int]],
    target_image_counts: dict[str, float],
    split_ratios: dict[str, float],
) -> tuple[float, int]:
    splits = ("train", "val", "test")
    split_patients = {split: [] for split in splits}
    for patient_id, split in assignment.items():
        split_patients[split].append(patient_records[patient_id])

    missing_count = 0
    score = 0.0
    for split in splits:
        covered = set()
        for record in split_patients[split]:
            covered.update(record["labels"])
        missing_count += len(required_labels[split] - covered)

        image_count = sum(record["n_images"] for record in split_patients[split])
        score += ((image_count - target_image_counts[split]) / 100.0) ** 2

    total_label_counts = Counter()
    for record in patient_records.values():
        total_label_counts.update(record["label_counts"])

    for label in range(len(LABELS)):
        total = total_label_counts[label]
        if total <= 0:
            continue
        scale = max(20.0, total * 0.05)
        for split in splits:
            if label not in required_labels[split]:
                continue
            observed = sum(record["label_counts"][label] for record in split_patients[split])
            expected = split_ratios[split] * total
            score += ((observed - expected) / scale) ** 2

    return score + missing_count * 1_000_000.0, missing_count


def allocate_multilabel_stratified_splits(
    rows: list[dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    """Best-effort patient-level multi-label stratification.

    BreaKHis has only two strict patient codes with adenosis, so it is
    impossible to place every subtype in train/val/test while preserving
    patient-level separation. We require train and test to cover every subtype
    that has at least two patients, and require validation coverage for
    subtypes with at least three patients.
    """

    patient_records = patient_label_records(rows)
    patient_ids = sorted(patient_records)
    n_patients = len(patient_ids)
    n_train = round(n_patients * train_ratio)
    n_val = round(n_patients * val_ratio)
    if n_train + n_val > n_patients:
        n_val = max(0, n_patients - n_train)
    n_test = n_patients - n_train - n_val

    label_patient_counts = {
        label: sum(label in record["labels"] for record in patient_records.values())
        for label in range(len(LABELS))
    }
    all_possible_labels = {
        label for label, count in label_patient_counts.items() if count >= 1
    }
    train_required = set(all_possible_labels)
    test_required = {
        label for label, count in label_patient_counts.items() if count >= 2
    }
    val_required = {
        label for label, count in label_patient_counts.items() if count >= 3
    }
    required_labels = {
        "train": train_required,
        "val": val_required,
        "test": test_required,
    }

    n_images = len(rows)
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    target_image_counts = {
        "train": train_ratio * n_images,
        "val": val_ratio * n_images,
        "test": test_ratio * n_images,
    }
    split_ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }

    rng = random.Random(seed)
    best_assignment = None
    best_score = float("inf")
    best_missing = 10**9

    # Exact patient-count splits are maintained by swapping two patients from
    # different splits during local search.
    for restart in range(500):
        shuffled = list(patient_ids)
        rng.shuffle(shuffled)
        assignment = {}
        for idx, patient_id in enumerate(shuffled):
            if idx < n_train:
                assignment[patient_id] = "train"
            elif idx < n_train + n_val:
                assignment[patient_id] = "val"
            else:
                assignment[patient_id] = "test"

        score, missing = multilabel_split_score(
            assignment,
            patient_records,
            required_labels,
            target_image_counts,
            split_ratios,
        )
        temperature = 0.1
        for _ in range(700):
            a, b = rng.sample(patient_ids, 2)
            if assignment[a] == assignment[b]:
                continue
            split_a, split_b = assignment[a], assignment[b]
            assignment[a], assignment[b] = split_b, split_a
            new_score, new_missing = multilabel_split_score(
                assignment,
                patient_records,
                required_labels,
                target_image_counts,
                split_ratios,
            )
            accept = new_score < score
            if not accept and temperature > 1e-12:
                accept = rng.random() < pow(2.718281828459045, (score - new_score) / temperature)
            if accept:
                score, missing = new_score, new_missing
            else:
                assignment[a], assignment[b] = split_a, split_b
            temperature *= 0.995

        if missing < best_missing or (missing == best_missing and score < best_score):
            best_assignment = dict(assignment)
            best_score = score
            best_missing = missing
            if best_missing == 0:
                break

    if best_assignment is None:
        raise RuntimeError("Failed to create a multi-label stratified split.")
    if best_missing != 0:
        missing_by_split = {}
        for split, required in required_labels.items():
            covered = set()
            for patient_id, assigned_split in best_assignment.items():
                if assigned_split == split:
                    covered.update(patient_records[patient_id]["labels"])
            missing_by_split[split] = [LABELS[label] for label in sorted(required - covered)]
        raise RuntimeError(
            "Could not satisfy required subtype coverage under strict patient-level constraints: "
            f"{missing_by_split}"
        )
    return best_assignment


def allocate_patient_splits(
    rows: list[dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    split_mode: str = "global_random",
) -> dict[str, str]:
    patient_labels = patient_major_label(rows)
    rng = random.Random(seed)
    patient_split = {}

    if split_mode == "global_random":
        patients = sorted(patient_labels)
        rng.shuffle(patients)
        n = len(patients)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        train_ids = patients[:n_train]
        val_ids = patients[n_train:n_train + n_val]
        test_ids = patients[n_train + n_val:]
        for pid in train_ids:
            patient_split[pid] = "train"
        for pid in val_ids:
            patient_split[pid] = "val"
        for pid in test_ids:
            patient_split[pid] = "test"
        return patient_split

    if split_mode == "patient_multilabel_stratified_best_effort":
        return allocate_multilabel_stratified_splits(
            rows=rows,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

    if split_mode != "stratified_min_per_class":
        raise ValueError(f"Unknown split_mode: {split_mode}")

    patients_by_label = defaultdict(list)
    for patient_id, label in patient_labels.items():
        patients_by_label[label].append(patient_id)

    for label, patients in sorted(patients_by_label.items()):
        patients = list(patients)
        rng.shuffle(patients)
        n = len(patients)
        test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
        if n >= 3:
            n_val = max(1, round(n * val_ratio))
            n_test = max(1, round(n * test_ratio))
            if n_val + n_test >= n:
                n_val = 1
                n_test = 1
            n_train = n - n_val - n_test
        else:
            n_train = max(1, round(n * train_ratio))
            n_val = max(0, round(n * val_ratio))
            if n_train + n_val > n:
                n_val = max(0, n - n_train)
            n_test = n - n_train - n_val
        train_ids = patients[:n_train]
        val_ids = patients[n_train:n_train + n_val]
        test_ids = patients[n_train + n_val:n_train + n_val + n_test]
        for pid in train_ids:
            patient_split[pid] = "train"
        for pid in val_ids:
            patient_split[pid] = "val"
        for pid in test_ids:
            patient_split[pid] = "test"
    return patient_split


def summarize(df: pd.DataFrame) -> dict:
    split_label = pd.crosstab(df["split"], df["label_name"]).to_dict()
    split_counts = df["split"].value_counts().to_dict()
    patient_counts = df.drop_duplicates("patient_id")["split"].value_counts().to_dict()
    return {
        "n_images": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "image_split_counts": {k: int(v) for k, v in split_counts.items()},
        "patient_split_counts": {k: int(v) for k, v in patient_counts.items()},
        "split_by_label": split_label,
        "magnification_counts": df["magnification"].value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/raw/BreaKHis_v1.tar.gz"))
    parser.add_argument("--extract-dir", type=Path, default=Path("data/raw/BreaKHis_v1_extracted"))
    parser.add_argument("--manifest-csv", type=Path, default=Path("manifests/breakhis_patient_split_seed3407.csv"))
    parser.add_argument("--manifest-json", type=Path, default=Path("data/breakhis/manifest.json"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--split-mode",
        choices=[
            "global_random",
            "stratified_min_per_class",
            "patient_multilabel_stratified_best_effort",
        ],
        default="global_random",
    )
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    if not args.skip_extract:
        if not args.archive.exists():
            raise FileNotFoundError(args.archive)
        print(f"Extracting {args.archive} to {args.extract_dir}")
        safe_extract_tar(args.archive, args.extract_dir)

    rows = scan_images(args.extract_dir)
    patient_split = allocate_patient_splits(
        rows,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        split_mode=args.split_mode,
    )
    for row in rows:
        row["split"] = patient_split[row["patient_id"]]

    df = pd.DataFrame(rows).sort_values(["split", "label", "patient_id", "id"]).reset_index(drop=True)

    train_patients = set(df.loc[df["split"] == "train", "patient_id"])
    val_patients = set(df.loc[df["split"] == "val", "patient_id"])
    test_patients = set(df.loc[df["split"] == "test", "patient_id"])
    if train_patients & val_patients or train_patients & test_patients or val_patients & test_patients:
        raise RuntimeError("Patient-level leakage detected across splits.")

    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.manifest_csv, index=False)

    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    json_root = args.manifest_json.parent
    for row in df.to_dict("records"):
        image_abs = (args.extract_dir / row["image_path"]).resolve()
        image_rel_to_json_root = os.path.relpath(image_abs, json_root.resolve())
        samples.append(
            {
                "id": row["id"],
                "image_path": image_rel_to_json_root,
                "split": row["split"],
                "label": int(row["label"]),
                "metadata": {
                    "patient_id": row["patient_id"],
                    "raw_case_id": row["raw_case_id"],
                    "binary_label": row["binary_label"],
                    "tumor_type": row["tumor_type"],
                    "magnification": row["magnification"],
                },
            }
        )
    with open(args.manifest_json, "w", encoding="utf-8") as f:
        json.dump({"name": "breakhis", "labels": LABELS, "samples": samples}, f, indent=2)

    summary = summarize(df)
    summary_path = args.manifest_csv.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote CSV manifest: {args.manifest_csv}")
    print(f"Wrote JSON manifest: {args.manifest_json}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
