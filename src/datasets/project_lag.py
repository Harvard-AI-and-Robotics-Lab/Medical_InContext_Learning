from pathlib import Path, PureWindowsPath

import pandas as pd

from .base import ClassificationDataset, ClassificationSample


class ProjectLAGDataset(ClassificationDataset):
    LABELS = ["non_glaucoma", "glaucoma"]

    def __init__(self, data_root: str, split: str = "all", manifest_csv: str = ""):
        self.manifest_csv = manifest_csv
        super().__init__(data_root=data_root, split=split)

    @property
    def name(self) -> str:
        return "lag_project"

    @property
    def label_names(self) -> list:
        return self.LABELS

    @property
    def n_classes(self) -> int:
        return 2

    def _resolve_image_path(self, row) -> str:
        raw_path = row.get("image_path", "")
        raw_path = "" if pd.isna(raw_path) else str(raw_path)
        if raw_path:
            candidate = Path(raw_path)
            if candidate.exists():
                return str(candidate)
            if ":\\" in raw_path or "://" in raw_path:
                win_path = PureWindowsPath(raw_path)
                sample_id = str(row["id"])
                class_name = str(row["class_name"])
                return str(Path(self.data_root) / class_name / "image" / f"{sample_id}.jpg")
        sample_id = str(row["id"])
        class_name = str(row["class_name"])
        return str(Path(self.data_root) / class_name / "image" / f"{sample_id}.jpg")

    def _load(self):
        if not self.manifest_csv:
            raise ValueError("ProjectLAGDataset requires manifest_csv.")

        df = pd.read_csv(self.manifest_csv, dtype={"id": str})
        if self.split != "all":
            df = df[df["split"] == self.split].copy()

        split_map = {"train": "train", "val": "val", "test": "test"}
        for row in df.to_dict("records"):
            label_idx = int(row["label"])
            label_name = self.LABELS[label_idx]
            self.samples.append(
                ClassificationSample(
                    id=str(row["id"]),
                    image_path=self._resolve_image_path(row),
                    split=split_map.get(str(row["split"]), str(row["split"])),
                    label=label_idx,
                    label_name=label_name,
                    metadata={
                        "class_name": str(row.get("class_name", "")),
                        "label_path": row.get("label_path", ""),
                        "attention_map_path": row.get("attention_map_path", ""),
                        "source_manifest": self.manifest_csv,
                    },
                )
            )

    def get_reference_pool(self):
        return [s for s in self.samples if s.split == "train"]

    def get_validation_samples(self):
        return [s for s in self.samples if s.split == "val"]
