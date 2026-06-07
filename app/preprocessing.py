"""
Preprocessing — Replique exacte du val_transform du notebook d'entrainement.

  Resize(32x32) -> Normalize(mean=[0.1307]*3, std=[0.3081]*3) -> ToTensorV2

Aucun data-augmentation en inference.
"""

import io
import base64

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from app.config import get_settings

settings = get_settings()

# Transform identique au val_transform du notebook
_TRANSFORM = A.Compose(
    [
        A.Resize(height=settings.input_size, width=settings.input_size),
        A.Normalize(mean=settings.normalize_mean, std=settings.normalize_std),
        ToTensorV2(),
    ]
)


def _pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convertit une PIL Image en tensor normalise pret pour le modele."""
    # Le modele a ete entraine sur des images Grayscale
    gray = pil_image.convert("L")
    np_img = np.array(gray)                          # (H, W) uint8
    transformed = _TRANSFORM(image=np_img)["image"]  # (1, 32, 32) float32
    return transformed.unsqueeze(0)                  # (1, 1, 32, 32)


def preprocess_bytes(image_bytes: bytes) -> torch.Tensor:
    """Preproce les bytes image (multipart upload)."""
    pil = Image.open(io.BytesIO(image_bytes))
    return _pil_to_tensor(pil)


def preprocess_base64(b64_string: str) -> torch.Tensor:
    """Preproce une image encodee en Base64."""
    # Supporte le prefixe data:image/...;base64, ou raw base64
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    raw = base64.b64decode(b64_string)
    return preprocess_bytes(raw)
