from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
from PIL import Image
import hashlib
import json


@dataclass
class EncoderOutput:
    global_embedding: np.ndarray
    spatial_features: Optional[np.ndarray] = None
    encoder_name: str = ""
    encoder_version: str = ""
    preprocessing_hash: str = ""

    def to_dict(self):
        result = {
            "global_embedding_shape": list(self.global_embedding.shape),
            "encoder_name": self.encoder_name,
            "encoder_version": self.encoder_version,
            "preprocessing_hash": self.preprocessing_hash,
        }
        if self.spatial_features is not None:
            result["spatial_features_shape"] = list(self.spatial_features.shape)
        return result


class BaseEncoder(ABC):
    def __init__(self, model_id: str, device: str = "cuda", image_size: int = 224):
        self.model_id = model_id
        self.device = device
        self.image_size = image_size
        self.model = None
        self.transform = None
        self._build()

    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def _extract(self, pixel_values: torch.Tensor) -> EncoderOutput:
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def spatial_token_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def encoder_version(self) -> str:
        pass

    def preprocessing_hash(self):
        desc = json.dumps({
            "model_id": self.model_id,
            "image_size": self.image_size,
            "transform": str(self.transform),
        }, sort_keys=True)
        return hashlib.sha256(desc.encode()).hexdigest()[:16]

    def encode_image(self, image: Image.Image) -> EncoderOutput:
        if self.transform is None:
            raise RuntimeError("Encoder not initialized")
        pixel_values = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self._extract(pixel_values)
        output.encoder_name = self.__class__.__name__
        output.encoder_version = self.encoder_version
        output.preprocessing_hash = self.preprocessing_hash()
        return output

    def encode_batch(self, images: list, batch_size: int = 32) -> list:
        results = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            tensors = torch.stack([self.transform(img) for img in batch_imgs]).to(self.device)
            with torch.no_grad():
                for j in range(tensors.shape[0]):
                    output = self._extract(tensors[j:j+1])
                    output.encoder_name = self.__class__.__name__
                    output.encoder_version = self.encoder_version
                    output.preprocessing_hash = self.preprocessing_hash()
                    results.append(output)
        return results

    def encode_paths(self, image_paths: list, batch_size: int = 32) -> list:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return self.encode_batch(images, batch_size)
