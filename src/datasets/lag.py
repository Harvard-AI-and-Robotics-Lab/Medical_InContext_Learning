from pathlib import Path
import json
from .base import ClassificationDataset, ClassificationSample
import random


class LAGDataset(ClassificationDataset):
    LABELS = ["non_glaucoma", "glaucoma"]

    @property
    def name(self) -> str:
        return "lag"

    @property
    def label_names(self) -> list:
        return self.LABELS

    @property
    def n_classes(self) -> int:
        return 2

    def _load(self):
        root = self.data_root / "lag"
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
                        sample_id = f"lag_{split_dir}_{img_path.stem}"
                        self.samples.append(ClassificationSample(
                            id=sample_id,
                            image_path=str(img_path),
                            split=split_dir,
                            label=label_idx,
                            label_name=label_name,
                        ))
