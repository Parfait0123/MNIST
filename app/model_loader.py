"""
Chargement du modèle MNIST depuis Kaggle au démarrage de l'API.
"""

import os
import logging
from typing import Optional

import torch
import kagglehub

from app.config import get_settings
from app.model_arch import MNISTClassifier

logger = logging.getLogger(__name__)
settings = get_settings()

_MODEL: Optional[MNISTClassifier] = None

DEVICE = torch.device("cpu")


def _download_and_load() -> MNISTClassifier:
    logger.info(f"Telechargement du modele depuis Kaggle : {settings.kaggle_model_handle}")

    model_dir = kagglehub.model_download(settings.kaggle_model_handle)
    logger.info(f"Contenu du dossier telecharge : {os.listdir(model_dir)}")

    local_path = os.path.join(model_dir, settings.model_filename)

    if not os.path.exists(local_path):
        pth_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
        if not pth_files:
            raise FileNotFoundError(f"Aucun fichier .pth trouve dans {model_dir}")
        local_path = os.path.join(model_dir, pth_files[0])
        logger.info(f"Fichier .pth detecte : {pth_files[0]}")

    logger.info(f"Chargement depuis : {local_path}")

    model = MNISTClassifier(
        num_classes=len(settings.classes),
    )

    state_dict = torch.load(local_path, map_location=DEVICE, weights_only=True)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    logger.info("Modele MNIST charge et en mode eval.")
    return model


def load_model() -> None:
    global _MODEL
    _MODEL = _download_and_load()


def get_model() -> MNISTClassifier:
    if _MODEL is None:
        raise RuntimeError("Le modele MNIST n'est pas charge.")
    return _MODEL