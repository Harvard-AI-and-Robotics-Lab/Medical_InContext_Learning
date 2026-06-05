import numpy as np
import torch

from .base import BaseEncoder, EncoderOutput


class BiomedCLIPEncoder(BaseEncoder):
    def __init__(
        self,
        model_id: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device: str = "cuda",
        image_size: int = 224,
    ):
        super().__init__(model_id=model_id, device=device, image_size=image_size)

    @property
    def embedding_dim(self) -> int:
        return int(getattr(self, "_embedding_dim", 512))

    @property
    def spatial_token_dim(self) -> int:
        return 0

    @property
    def encoder_version(self) -> str:
        return f"biomedclip-openclip-{self.model_id}"

    def _build(self):
        from open_clip import create_model_from_pretrained

        model_name = f"hf-hub:{self.model_id}"
        self.model, self.transform = create_model_from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self._embedding_dim = int(getattr(self.model, "embed_dim", 512))

    def _extract(self, pixel_values: torch.Tensor) -> EncoderOutput:
        features = self.model.encode_image(pixel_values)
        features = torch.nn.functional.normalize(features, dim=-1)
        global_embedding = features.detach().cpu().numpy().squeeze(0).astype(np.float32, copy=False)
        return EncoderOutput(global_embedding=global_embedding, spatial_features=None)

    def encode_batch(self, images: list, batch_size: int = 32) -> list:
        results = []
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start:start + batch_size]
            tensors = torch.stack([self.transform(img) for img in batch_imgs]).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(tensors)
                features = torch.nn.functional.normalize(features, dim=-1)
            for feature in features.detach().cpu().numpy().astype(np.float32, copy=False):
                output = EncoderOutput(global_embedding=feature, spatial_features=None)
                output.encoder_name = self.__class__.__name__
                output.encoder_version = self.encoder_version
                output.preprocessing_hash = self.preprocessing_hash()
                results.append(output)
        return results
