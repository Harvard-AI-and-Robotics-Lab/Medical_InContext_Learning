from pathlib import Path
import json
import csv
import pandas as pd
from .base import ClassificationDataset, ClassificationSample


class CheXpertDataset(ClassificationDataset):
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

    def __init__(self, data_root: str, split: str = "all", manifest_csv: str = "", target_label_names: list | None = None):
        self.manifest_csv = manifest_csv
        self.target_label_names = list(target_label_names or self.LABELS)
        unknown = [name for name in self.target_label_names if name not in self.LABELS]
        if unknown:
            raise ValueError(f"Unknown CheXpert target labels: {unknown}. Available: {self.LABELS}")
        self.target_label_indices = [self.LABELS.index(name) for name in self.target_label_names]
        super().__init__(data_root=data_root, split=split)

    @property
    def name(self) -> str:
        return "chexpert"

    @property
    def label_names(self) -> list:
        return self.target_label_names

    @property
    def n_classes(self) -> int:
        return len(self.target_label_names)

    @property
    def is_multi_label(self) -> bool:
        return True

    def _parse_chexpert_label(self, value):
        if value == "" or value == "-1":
            return 0
        try:
            v = float(value)
            return 1 if v == 1.0 else 0
        except (ValueError, TypeError):
            return 0

    def _load(self):
        if self.manifest_csv:
            df = pd.read_csv(self.manifest_csv, dtype={"id": str, "patient_id": str, "study_id": str})
            if self.split != "all":
                df = df[df["split"] == self.split].copy()
            for row in df.to_dict("records"):
                image_path = str(row["image_path"])
                if not Path(image_path).is_absolute():
                    image_path = str(Path(self.data_root) / image_path)
                full_multi_label = [self._parse_chexpert_label(row.get(label_name, 0)) for label_name in self.LABELS]
                multi_label = [full_multi_label[idx] for idx in self.target_label_indices]
                self.samples.append(ClassificationSample(
                    id=str(row["id"]),
                    image_path=image_path,
                    split=str(row["split"]),
                    label=-1,
                    label_name="multi_label",
                    multi_label=multi_label,
                    metadata={
                        "patient_id": str(row.get("patient_id", "")),
                        "study_id": str(row.get("study_id", "")),
                        "view": str(row.get("view", "")),
                        "source_path": str(row.get("source_path", "")),
                        "source_manifest": self.manifest_csv,
                        "all_label_names": self.LABELS,
                        "target_label_names": self.target_label_names,
                        "target_label_indices": self.target_label_indices,
                    },
                ))
            return

        root = self.data_root / "chexpert"
        manifest_path = root / "manifest.json"

        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            for entry in manifest["samples"]:
                if self.split == "all" or entry["split"] == self.split:
                    self.samples.append(ClassificationSample(
                        id=entry["id"],
                        image_path=str(root / entry["image_path"]),
                        split=entry["split"],
                        label=-1,
                        label_name="multi_label",
                        multi_label=[entry["multi_label"][idx] for idx in self.target_label_indices],
                        metadata=entry.get("metadata", {}),
                    ))
            return

        for split_dir in ["reference", "test"]:
            split_path = root / split_dir
            if not split_path.exists():
                continue
            if self.split != "all" and split_dir != self.split:
                continue

            csv_path = split_path / "labels.csv"
            if not csv_path.exists():
                continue

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_rel = row.get("Path", row.get("path", ""))
                    img_path = split_path / img_rel
                    if not img_path.exists():
                        continue

                    full_multi_label = []
                    for label_name in self.LABELS:
                        col_name = label_name.replace("_", " ").title()
                        alt_col = label_name
                        value = row.get(col_name, row.get(alt_col, "0"))
                        full_multi_label.append(self._parse_chexpert_label(value))
                    multi_label = [full_multi_label[idx] for idx in self.target_label_indices]

                    sample_id = f"chexpert_{split_dir}_{Path(img_rel).stem}"
                    self.samples.append(ClassificationSample(
                        id=sample_id,
                        image_path=str(img_path),
                        split=split_dir,
                        label=-1,
                        label_name="multi_label",
                        multi_label=multi_label,
                    ))


    def get_reference_pool(self):
        return [s for s in self.samples if s.split == "train"]

    def get_validation_samples(self):
        return [s for s in self.samples if s.split == "val"]
