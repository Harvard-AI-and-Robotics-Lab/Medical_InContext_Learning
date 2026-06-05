import numpy as np
import torch

from .base import BaseEncoder, EncoderOutput


class SigLIP2Encoder(BaseEncoder):
    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch16-384",
        device: str = "cuda",
        image_size: int = 384,
        torch_dtype: str = "float16",
    ):
        self.torch_dtype_name = torch_dtype
        super().__init__(model_id=model_id, device=device, image_size=image_size)

    @property
    def embedding_dim(self) -> int:
        return int(getattr(self, "_embedding_dim", 1152))

    @property
    def spatial_token_dim(self) -> int:
        return 0

    @property
    def encoder_version(self) -> str:
        return f"siglip2-{self.model_id}-{self.torch_dtype_name}"

    def _resolve_dtype(self):
        if self.torch_dtype_name in ("float16", "fp16"):
            return torch.float16
        if self.torch_dtype_name in ("bfloat16", "bf16"):
            return torch.bfloat16
        return torch.float32

    def _build(self):
        from transformers import AutoModel, AutoProcessor

        dtype = self._resolve_dtype()
        self.model = AutoModel.from_pretrained(self.model_id, dtype=dtype).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.transform = self.processor
        vision_config = getattr(self.model.config, "vision_config", None)
        self._embedding_dim = int(getattr(vision_config, "projection_size", 0) or getattr(self.model.config, "projection_dim", 1152))

    def _features_from_output(self, output):
        features = output.pooler_output if hasattr(output, "pooler_output") else output
        features = torch.nn.functional.normalize(features, dim=-1)
        return features

    def _extract(self, pixel_values: torch.Tensor) -> EncoderOutput:
        output = self.model.get_image_features(pixel_values=pixel_values)
        features = self._features_from_output(output)
        global_embedding = features.detach().cpu().numpy().squeeze(0).astype(np.float32, copy=False)
        return EncoderOutput(global_embedding=global_embedding, spatial_features=None)

    def encode_image(self, image) -> EncoderOutput:
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self._extract(inputs["pixel_values"])
        output.encoder_name = self.__class__.__name__
        output.encoder_version = self.encoder_version
        output.preprocessing_hash = self.preprocessing_hash()
        return output

    def encode_batch(self, images: list, batch_size: int = 32) -> list:
        results = []
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start:start + batch_size]
            inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.get_image_features(**inputs)
                features = self._features_from_output(output)
            for feature in features.detach().cpu().numpy().astype(np.float32, copy=False):
                output = EncoderOutput(global_embedding=feature, spatial_features=None)
                output.encoder_name = self.__class__.__name__
                output.encoder_version = self.encoder_version
                output.preprocessing_hash = self.preprocessing_hash()
                results.append(output)
        return results
