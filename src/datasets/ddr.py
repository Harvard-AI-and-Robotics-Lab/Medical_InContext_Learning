from pathlib import Path
import json
import pandas as pd
from .base import ClassificationDataset, ClassificationSample


class DDRDataset(ClassificationDataset):
    LABELS = [
        "no_dr",
        "mild_npdr",
        "moderate_npdr",
        "severe_npdr",
        "proliferative_dr",
        "ungradable",
    ]

    def __init__(self, data_root: str, split: str = "all", manifest_csv: str = ""):
        self.manifest_csv = manifest_csv
        super().__init__(data_root=data_root, split=split)

    @property
    def name(self) -> str:
        return "ddr"

    @property
    def label_names(self) -> list:
        return self.LABELS

    @property
    def n_classes(self) -> int:
        return 6

    def _load(self):
        if self.manifest_csv:
            df = pd.read_csv(self.manifest_csv, dtype={"id": str, "image_name": str})
            if self.split != "all":
                df = df[df["split"] == self.split].copy()
            for row in df.to_dict("records"):
                label_idx = int(row["label"])
                image_path = str(row["image_path"])
                if not Path(image_path).is_absolute():
                    image_path = str(Path(self.data_root) / image_path)
                self.samples.append(ClassificationSample(
                    id=str(row["id"]),
                    image_path=image_path,
                    split=str(row["split"]),
                    label=label_idx,
                    label_name=self.LABELS[label_idx],
                    metadata={
                        "image_name": str(row.get("image_name", "")),
                        "official_split": str(row.get("official_split", row.get("split", ""))),
                        "source_manifest": self.manifest_csv,
                    },
                ))
            return

        root = self.data_root / "ddr"
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
                        label=entry["label"],
                        label_name=self.LABELS[entry["label"]],
                        metadata=entry.get("metadata", {}),
                    ))
            return

        for split_dir in ["reference", "test"]:
            split_path = root / split_dir
            if not split_path.exists():
                continue
            if self.split != "all" and split_dir != self.split:
                continue
            for label_idx, label_name in enumerate(self.LABELS):
                label_dir = split_path / label_name
                if not label_dir.exists():
                    continue
                for img_path in sorted(label_dir.glob("*")):
                    if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                        sample_id = f"ddr_{split_dir}_{img_path.stem}"
                        self.samples.append(ClassificationSample(
                            id=sample_id,
                            image_path=str(img_path),
                            split=split_dir,
                            label=label_idx,
                            label_name=label_name,
                        ))

    def get_reference_pool(self):
        return [s for s in self.samples if s.split in ("train", "reference")]

    def get_validation_samples(self):
        return [s for s in self.samples if s.split in ("val", "valid")]
