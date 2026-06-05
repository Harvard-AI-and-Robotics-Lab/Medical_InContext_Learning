from .base import BaseEncoder, EncoderOutput
from .dinov3 import DINOv3Encoder
from .clip_encoder import CLIPEncoder
from .mae import MAEEncoder
from .biomedclip import BiomedCLIPEncoder
from .openclip_encoder import OpenCLIPEncoder
from .siglip2_encoder import SigLIP2Encoder

ENCODERS = {
    "dinov3": DINOv3Encoder,
    "clip": CLIPEncoder,
    "mae": MAEEncoder,
    "biomedclip": BiomedCLIPEncoder,
    "openclip": OpenCLIPEncoder,
    "siglip2": SigLIP2Encoder,
}


def get_encoder(name: str, **kwargs) -> BaseEncoder:
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(ENCODERS.keys())}")
    return ENCODERS[name](**kwargs)
