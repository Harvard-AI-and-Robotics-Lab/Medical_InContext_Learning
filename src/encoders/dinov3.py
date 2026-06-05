import json
import hashlib

import numpy as np
import torch

from .base import BaseEncoder, EncoderOutput


class _DINOv3FallbackImageProcessor:
    """Minimal ImageNet processor for DINOv3 repos without preprocessor_config.json."""

    def __init__(self):
        self.size = None
        self.crop_size = None
        self.do_resize = True
        self.do_rescale = True
        self.do_normalize = True
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

    def __call__(self, images, return_tensors="pt", do_resize=True, size=None):
        from PIL import Image

        if not isinstance(images, (list, tuple)):
            images = [images]
        height = int((size or {}).get("height") or (size or {}).get("shortest_edge") or 518)
        width = int((size or {}).get("width") or height)
        mean = torch.tensor(self.image_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.image_std, dtype=torch.float32).view(3, 1, 1)
        tensors = []
        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image))
            image = image.convert("RGB")
            if do_resize:
                image = image.resize((width, height), Image.BICUBIC)
            arr = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensor = (tensor - mean) / std
            tensors.append(tensor)
        return {"pixel_values": torch.stack(tensors, dim=0)}



class DINOv3Encoder(BaseEncoder):
    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        device: str = "cuda",
        image_size: int = 512,
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
        return "dinov3-vitl16-lvd1689m-v1"

    def _coerce_image_size(self) -> int:
        if self.image_size % 16 == 0:
            return self.image_size
        return max(16, self.image_size - (self.image_size % 16))

    def _build(self):
        from transformers import AutoImageProcessor, AutoModel

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        except OSError as exc:
            if "preprocessor_config.json" not in str(exc) and "image processor" not in str(exc):
                raise
            self.processor = _DINOv3FallbackImageProcessor()
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # BaseEncoder expects `transform` to exist. DINOv3 uses the model's
        # image processor instead of a torchvision transform pipeline.
        self.transform = f"dino3_auto_image_processor(size={self._coerce_image_size()})"

    def preprocessing_hash(self):
        desc = json.dumps(
            {
                "model_id": self.model_id,
                "image_size": self._coerce_image_size(),
                "processor_class": getattr(self.processor, "__class__", type(self.processor)).__name__,
                "processor_size": getattr(self.processor, "size", None),
                "processor_crop_size": getattr(self.processor, "crop_size", None),
                "processor_do_resize": getattr(self.processor, "do_resize", None),
                "processor_do_rescale": getattr(self.processor, "do_rescale", None),
                "processor_do_normalize": getattr(self.processor, "do_normalize", None),
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(desc.encode()).hexdigest()[:16]

    def _prepare_inputs(self, images):
        size = self._coerce_image_size()
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            do_resize=True,
            size={"height": size, "width": size},
        )
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _extract(self, model_inputs: dict) -> EncoderOutput:
        outputs = self.model(**model_inputs)

        global_tensor = getattr(outputs, "pooler_output", None)
        if global_tensor is None:
            global_tensor = outputs.last_hidden_state[:, 0, :]

        global_embedding = global_tensor[0].detach().cpu().numpy()
        norm = np.linalg.norm(global_embedding)
        if norm > 0:
            global_embedding = global_embedding / norm

        spatial_features = None
        hidden = getattr(outputs, "last_hidden_state", None)
        if self.return_spatial and hidden is not None and hidden.ndim == 3:
            num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
            patch_start = 1 + num_register_tokens
            if hidden.shape[1] > patch_start:
                spatial_features = hidden[:, patch_start:, :][0].detach().cpu().numpy()

        return EncoderOutput(global_embedding=global_embedding, spatial_features=spatial_features)

    def encode_image(self, image) -> EncoderOutput:
        with torch.no_grad():
            output = self._extract(self._prepare_inputs(image))
        output.encoder_name = self.__class__.__name__
        output.encoder_version = self.encoder_version
        output.preprocessing_hash = self.preprocessing_hash()
        return output

    def encode_batch(self, images: list, batch_size: int = 16) -> list:
        results = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            inputs = self._prepare_inputs(batch_imgs)
            with torch.no_grad():
                outputs = self.model(**inputs)

            global_tensor = getattr(outputs, "pooler_output", None)
            if global_tensor is None:
                global_tensor = outputs.last_hidden_state[:, 0, :]

            hidden = getattr(outputs, "last_hidden_state", None) if self.return_spatial else None
            num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
            patch_start = 1 + num_register_tokens

            for batch_idx in range(global_tensor.shape[0]):
                global_embedding = global_tensor[batch_idx].detach().cpu().numpy()
                norm = np.linalg.norm(global_embedding)
                if norm > 0:
                    global_embedding = global_embedding / norm

                spatial_features = None
                if hidden is not None and hidden.ndim == 3 and hidden.shape[1] > patch_start:
                    spatial_features = hidden[batch_idx, patch_start:, :].detach().cpu().numpy()

                output = EncoderOutput(
                    global_embedding=global_embedding,
                    spatial_features=spatial_features,
                    encoder_name=self.__class__.__name__,
                    encoder_version=self.encoder_version,
                    preprocessing_hash=self.preprocessing_hash(),
                )
                results.append(output)
        return results
