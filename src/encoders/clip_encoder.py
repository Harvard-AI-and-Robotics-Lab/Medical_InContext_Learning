import torch
import numpy as np
from torchvision import transforms
from .base import BaseEncoder, EncoderOutput


class CLIPEncoder(BaseEncoder):
    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        image_size: int = 224,
        return_spatial: bool = True,
    ):
        self.return_spatial = return_spatial
        super().__init__(model_id=model_id, device=device, image_size=image_size)

    @property
    def embedding_dim(self) -> int:
        return 1024

    @property
    def spatial_token_dim(self) -> int:
        return 1024

    @property
    def encoder_version(self) -> str:
        return "clip-vit-large-patch14-v1"

    def _build(self):
        from transformers import CLIPModel, CLIPProcessor

        self.clip_model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.processor = CLIPProcessor.from_pretrained(self.model_id)

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def _extract(self, pixel_values: torch.Tensor) -> EncoderOutput:
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values, output_hidden_states=self.return_spatial)
        cls_token = vision_outputs.last_hidden_state[:, 0, :]
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :] if self.return_spatial else None

        global_embedding = cls_token.cpu().numpy().squeeze(0)
        spatial_features = patch_tokens.cpu().numpy().squeeze(0) if patch_tokens is not None else None

        norm = np.linalg.norm(global_embedding)
        if norm > 0:
            global_embedding = global_embedding / norm

        return EncoderOutput(
            global_embedding=global_embedding,
            spatial_features=spatial_features,
        )


    def encode_batch(self, images: list, batch_size: int = 32) -> list:
        results = []
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start:start + batch_size]
            tensors = torch.stack([self.transform(img) for img in batch_imgs]).to(self.device)
            with torch.no_grad():
                vision_outputs = self.clip_model.vision_model(pixel_values=tensors, output_hidden_states=self.return_spatial)
                cls_tokens = vision_outputs.last_hidden_state[:, 0, :]
                patch_tokens = vision_outputs.last_hidden_state[:, 1:, :] if self.return_spatial else None
                cls_np = cls_tokens.detach().cpu().numpy().astype(np.float32, copy=False)
                norms = np.linalg.norm(cls_np, axis=1, keepdims=True)
                cls_np = cls_np / np.maximum(norms, 1e-12)
                patch_np = patch_tokens.detach().cpu().numpy().astype(np.float32, copy=False) if patch_tokens is not None else None
            for idx, global_embedding in enumerate(cls_np):
                output = EncoderOutput(
                    global_embedding=global_embedding,
                    spatial_features=patch_np[idx] if patch_np is not None else None,
                    encoder_name=self.__class__.__name__,
                    encoder_version=self.encoder_version,
                    preprocessing_hash=self.preprocessing_hash(),
                )
                results.append(output)
        return results
