import torch
import numpy as np
from torchvision import transforms
from .base import BaseEncoder, EncoderOutput


class MAEEncoder(BaseEncoder):
    def __init__(self, model_id: str = "facebook/vit-mae-large", device: str = "cuda", image_size: int = 224):
        super().__init__(model_id=model_id, device=device, image_size=image_size)

    @property
    def embedding_dim(self) -> int:
        return 1024

    @property
    def spatial_token_dim(self) -> int:
        return 1024

    @property
    def encoder_version(self) -> str:
        return "vit-mae-large-v1"

    def _build(self):
        from transformers import ViTMAEModel

        self.model = ViTMAEModel.from_pretrained(self.model_id).to(self.device)
        # Retrieval features must be deterministic and include all image patches.
        self.model.config.mask_ratio = 0.0
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _extract(self, pixel_values: torch.Tensor) -> EncoderOutput:
        outputs = self.model(pixel_values)
        all_tokens = outputs.last_hidden_state
        cls_token = all_tokens[:, 0, :]
        patch_tokens = all_tokens[:, 1:, :]

        global_embedding = cls_token.cpu().numpy().squeeze(0)
        spatial_features = patch_tokens.cpu().numpy().squeeze(0)

        norm = np.linalg.norm(global_embedding)
        if norm > 0:
            global_embedding = global_embedding / norm

        return EncoderOutput(
            global_embedding=global_embedding,
            spatial_features=spatial_features,
        )
