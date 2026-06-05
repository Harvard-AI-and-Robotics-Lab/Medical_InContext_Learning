from pathlib import Path

import pandas as pd

from .base import ClassificationDataset, ClassificationSample


class TBX11KDataset(ClassificationDataset):
    LABELS = [
        "healthy",
        "sick but non-TB",
        "TB",
    ]

    def __init__(self, data_root: str, split: str = "all", manifest_csv: str = "", target_label_names: list | None = None):
        self.manifest_csv = manifest_csv
        if target_label_names and list(target_label_names) != self.LABELS:
            raise ValueError(f"TBX11K target_label_names must be {self.LABELS}, got {target_label_names}")
        super().__init__(data_root=data_root, split=split)

    @property
    def name(self) -> str:
        return "tbx11k"

    @property
    def label_names(self) -> list:
        return self.LABELS

    @property
    def n_classes(self) -> int:
        return len(self.LABELS)

    def _load(self):
        if not self.manifest_csv:
            raise ValueError("TBX11KDataset requires manifest_csv.")

        df = pd.read_csv(self.manifest_csv, dtype={"id": str})
        if self.split != "all":
            df = df[df["split"] == self.split].copy()

        for row in df.to_dict("records"):
            label_idx = int(row["label"])
            image_path = str(row["image_path"])
            if not Path(image_path).is_absolute():
                image_path = str(Path(self.data_root) / image_path)
            self.samples.append(
                ClassificationSample(
                    id=str(row["id"]),
                    image_path=image_path,
                    split=str(row["split"]),
                    label=label_idx,
                    label_name=self.LABELS[label_idx],
                    metadata={
                        "source_rel_path": str(row.get("source_rel_path", "")),
                        "official_split": str(row.get("official_split", "")),
                        "source_manifest": self.manifest_csv,
                    },
                )
            )

    def get_reference_pool(self):
        return [s for s in self.samples if s.split == "train"]

    def get_validation_samples(self):
        return [s for s in self.samples if s.split == "val"]
