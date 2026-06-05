from pathlib import Path
import json

from .base import VQADataset, VQASample


class PathMMUDataset(VQADataset):
    def __init__(self, data_root: str, split: str = "test", manifest_json: str = ""):
        self.manifest_json = manifest_json
        super().__init__(data_root=data_root, split=split)

    @property
    def name(self) -> str:
        return "pathmmu"

    def _manifest_path(self, root: Path) -> Path:
        if self.manifest_json:
            path = Path(self.manifest_json)
            return path if path.is_absolute() else Path.cwd() / path
        return root / "manifest_available_samples.json"

    @staticmethod
    def _format_multichoice_question(question: str, options: list[str]) -> str:
        if not options:
            return question
        return (
            f"{question}\n"
            "Options:\n"
            + "\n".join(str(option) for option in options)
            + "\nChoose the single best option and answer with the exact option text."
        )

    def _load(self):
        root = self.data_root / "pathmmu"
        manifest_path = self._manifest_path(root)
        if not manifest_path.exists():
            raise FileNotFoundError(f"PathMMU manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        samples = manifest.get("samples", manifest if isinstance(manifest, list) else [])
        for entry in samples:
            entry_split = entry.get("split", "")
            if self.split != "all" and entry_split != self.split:
                continue

            image_path = Path(entry["image_path"])
            if not image_path.is_absolute():
                image_path = root / image_path
            if not image_path.exists():
                continue

            metadata = entry.get("metadata", {}) or {}
            options = entry.get("options", metadata.get("options", [])) or []
            metadata = dict(metadata)
            metadata["options"] = options
            metadata["raw_question"] = entry.get("question", "")

            self.samples.append(
                VQASample(
                    id=str(entry["id"]),
                    image_path=str(image_path),
                    split=entry_split,
                    question=self._format_multichoice_question(entry.get("question", ""), options),
                    answer=str(entry.get("answer", "")),
                    question_type=entry.get("answer_type", "MULTICHOICE"),
                    metadata=metadata,
                )
            )
