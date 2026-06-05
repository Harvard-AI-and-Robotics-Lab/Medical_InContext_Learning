from pathlib import Path
import json

from .base import VQADataset, VQASample


class SLAKEDataset(VQADataset):
    def __init__(self, data_root: str, split: str = "test", manifest_json: str = "", lang: str = "en"):
        self.manifest_json = manifest_json
        self.lang = lang
        super().__init__(data_root=data_root, split=split)

    @property
    def name(self) -> str:
        return "slake"

    def _resolve_root(self) -> Path:
        root = self.data_root
        if root.name.lower() == "slake":
            return root
        return root / "slake"

    def _manifest_path(self, root: Path) -> Path:
        if self.manifest_json:
            path = Path(self.manifest_json)
            return path if path.is_absolute() else Path.cwd() / path
        lang_suffix = self.lang if self.lang else "en"
        return root / f"manifest_{lang_suffix}.json"

    def _load(self):
        root = self._resolve_root()
        manifest_path = self._manifest_path(root)
        if not manifest_path.exists():
            raise FileNotFoundError(f"SLAKE manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        for entry in manifest.get("samples", []):
            entry_split = entry.get("split", "")
            if self.split != "all" and entry_split != self.split:
                continue
            image_path = Path(entry["image_path"])
            if not image_path.is_absolute():
                image_path = root / image_path
            self.samples.append(
                VQASample(
                    id=str(entry["id"]),
                    image_path=str(image_path),
                    split=entry_split,
                    question=entry.get("question", ""),
                    answer=str(entry.get("answer", "")),
                    question_type=entry.get("answer_type", entry.get("question_type", "")),
                    metadata=entry.get("metadata", {}),
                )
            )

    def get_validation_samples(self):
        return [s for s in self.samples if s.split == "validation"]
