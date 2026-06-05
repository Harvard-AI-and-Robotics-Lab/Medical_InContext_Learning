from pathlib import Path
import json
from .base import VQADataset, VQASample


class PathVQADataset(VQADataset):
    def __init__(self, data_root: str, split: str = "test", manifest_json: str = ""):
        self.manifest_json = manifest_json
        super().__init__(data_root=data_root, split=split)

    @property
    def name(self) -> str:
        return "pathvqa"

    def _manifest_path(self, root: Path) -> Path:
        if self.manifest_json:
            path = Path(self.manifest_json)
            return path if path.is_absolute() else Path.cwd() / path
        return root / "manifest.json"

    def _load(self):
        root = self.data_root / "pathvqa"
        manifest_path = self._manifest_path(root)

        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            for entry in manifest["samples"]:
                if self.split == "all" or entry["split"] == self.split:
                    self.samples.append(VQASample(
                        id=entry["id"],
                        image_path=str(root / entry["image_path"]),
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
                sample_id = f"pathvqa_{split_dir}_{item.get('id', item['image'].split('.')[0])}"
                self.samples.append(VQASample(
                    id=sample_id,
                    image_path=str(img_path),
                    split=split_dir,
                    question=item["question"],
                    answer=item["answer"],
                    question_type=item.get("question_type", ""),
                ))
