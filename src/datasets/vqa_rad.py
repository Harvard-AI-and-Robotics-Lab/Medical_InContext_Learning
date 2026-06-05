from pathlib import Path
import json
import pandas as pd
from .base import VQADataset, VQASample


class VQARADDataset(VQADataset):
    def __init__(
        self,
        data_root: str,
        split: str = "all",
        manifest_csv: str = "",
        manifest_json: str = "",
    ):
        self.manifest_csv = manifest_csv
        self.manifest_json = manifest_json
        super().__init__(data_root=data_root, split=split)

    @property
    def name(self) -> str:
        return "vqa_rad"

    def _load(self):
        if self.manifest_csv:
            df = pd.read_csv(self.manifest_csv, dtype={"id": str})
            if self.split != "all":
                df = df[df["split"] == self.split].copy()
            for row in df.to_dict("records"):
                image_path = str(row["image_path"])
                if not Path(image_path).is_absolute():
                    image_path = str(Path(self.data_root) / image_path)
                self.samples.append(VQASample(
                    id=str(row["id"]),
                    image_path=image_path,
                    split=str(row["split"]),
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    question_type=str(row.get("question_type", "")),
                    metadata={
                        "source_image_path": str(row.get("source_image_path", "")),
                        "source_manifest": self.manifest_csv,
                    },
                ))
            return

        root = self.data_root / "vqa_rad"
        manifest_path = Path(self.manifest_json) if self.manifest_json else root / "manifest.json"
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path

        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            for entry in manifest["samples"]:
                if self.split == "all" or entry["split"] == self.split:
                    image_path = Path(entry["image_path"])
                    if not image_path.is_absolute():
                        image_path = root / image_path
                    self.samples.append(VQASample(
                        id=entry["id"],
                        image_path=str(image_path),
                        split=entry["split"],
                        question=entry["question"],
                        answer=entry["answer"],
                        question_type=entry.get("answer_type", entry.get("question_type", "")),
                        metadata=entry.get("metadata", {}),
                    ))
            return

        for split_dir in ["reference", "test"]:
            split_path = root / split_dir
            if not split_path.exists():
                continue
            if self.split != "all" and split_dir != self.split:
                continue

            qa_path = split_path / "qa.json"
            if not qa_path.exists():
                continue

            with open(qa_path, "r") as f:
                qa_data = json.load(f)

            for item in qa_data:
                img_path = split_path / "images" / item["image"]
                if not img_path.exists():
                    continue
                sample_id = f"vqa_rad_{split_dir}_{item.get('id', item['image'].split('.')[0])}"
                self.samples.append(VQASample(
                    id=sample_id,
                    image_path=str(img_path),
                    split=split_dir,
                    question=item["question"],
                    answer=item["answer"],
                    question_type=item.get("question_type", ""),
                ))

    def get_reference_pool(self):
        return [s for s in self.samples if s.split in ("train", "reference")]

    def get_validation_samples(self):
        return [s for s in self.samples if s.split == "val"]
