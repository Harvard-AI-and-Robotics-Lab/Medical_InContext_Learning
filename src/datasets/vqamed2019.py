from pathlib import Path
import json

from .base import VQADataset, VQASample


class VQAMed2019Dataset(VQADataset):
    @property
    def name(self) -> str:
        return "vqamed2019"

    def __init__(self, data_root: str, split: str = "test", manifest_json: str = ""):
        self.manifest_json = manifest_json
        super().__init__(data_root=data_root, split=split)

    def _load(self):
        root = self.data_root / "vqamed2019"
        manifest_path = Path(self.manifest_json) if self.manifest_json else root / "manifest.json"
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"VQA-Med 2019 manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        for entry in manifest["samples"]:
            if self.split != "all" and entry["split"] != self.split:
                continue
            image_path = Path(entry["image_path"])
            if not image_path.is_absolute():
                image_path = root / image_path
            self.samples.append(
                VQASample(
                    id=entry["id"],
                    image_path=str(image_path),
                    split=entry["split"],
                    question=entry["question"],
                    answer=entry["answer"],
                    question_type=entry.get("answer_type", ""),
                    metadata=entry.get("metadata", {}),
                )
            )
