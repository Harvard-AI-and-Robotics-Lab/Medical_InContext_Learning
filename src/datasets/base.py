from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from PIL import Image
import json
import hashlib


@dataclass
class Sample:
    id: str
    image_path: str
    split: str
    metadata: dict = None

    def load_image(self):
        return Image.open(self.image_path).convert("RGB")

    def image_hash(self):
        with open(self.image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()


@dataclass
class ClassificationSample(Sample):
    label: int = -1
    label_name: str = ""
    multi_label: list = None


@dataclass
class VQASample(Sample):
    question: str = ""
    answer: str = ""
    question_type: str = ""


class BaseDataset(ABC):
    def __init__(self, data_root: str, split: str = "test"):
        self.data_root = Path(data_root)
        self.split = split
        self.samples = []
        self._load()

    @abstractmethod
    def _load(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def task_type(self) -> str:
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_reference_pool(self):
        return [s for s in self.samples if s.split == "reference"]

    def get_test_samples(self):
        return [s for s in self.samples if s.split == "test"]

    def get_ids(self):
        return [s.id for s in self.samples]

    def summary(self):
        splits = {}
        for s in self.samples:
            splits[s.split] = splits.get(s.split, 0) + 1
        return {"name": self.name, "task_type": self.task_type, "total": len(self.samples), "splits": splits}

    def save_manifest(self, path: str):
        manifest = {
            "name": self.name,
            "task_type": self.task_type,
            "n_samples": len(self.samples),
            "samples": [{"id": s.id, "split": s.split, "image_path": s.image_path} for s in self.samples],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)


class ClassificationDataset(BaseDataset):
    @property
    def task_type(self) -> str:
        return "classification"

    @property
    @abstractmethod
    def label_names(self) -> list:
        pass

    @property
    @abstractmethod
    def n_classes(self) -> int:
        pass

    @property
    def is_multi_label(self) -> bool:
        return False

    def label_distribution(self):
        dist = {}
        for s in self.samples:
            if isinstance(s, ClassificationSample):
                key = s.label_name if s.label_name else str(s.label)
                dist[key] = dist.get(key, 0) + 1
        return dist


class VQADataset(BaseDataset):
    @property
    def task_type(self) -> str:
        return "vqa"

    def get_questions(self):
        return [s.question for s in self.samples if isinstance(s, VQASample)]
