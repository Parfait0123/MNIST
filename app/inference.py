"""
Moteur d'inference — logique partagee pour l'endpoint de prediction MNIST.
"""

import torch
import torch.nn.functional as F

from app.model_arch import MNISTClassifier
from app.model_loader import DEVICE


@torch.inference_mode()
def run_inference(model: MNISTClassifier, tensor: torch.Tensor, class_labels: list[str]) -> dict:
    """
    Execute une inference et retourne le chiffre predit + probabilites.

    Args:
        model:        Modele charge en mode eval
        tensor:       Tenseur (1, 3, H, W) deja normalise
        class_labels: Liste des noms de classes dans l'ordre des logits

    Returns:
        dict avec prediction (str), confidence (float), probabilities (dict),
        top3 (list des 3 classes les plus probables)
    """
    tensor = tensor.to(DEVICE)
    logits = model(tensor)                              # (1, 10)
    probs = F.softmax(logits, dim=1).squeeze(0)         # (10,)

    probs_list = probs.cpu().tolist()
    probs_dict = {label: round(p, 6) for label, p in zip(class_labels, probs_list)}

    best_idx = int(probs.argmax())
    prediction = class_labels[best_idx]
    confidence = round(probs_list[best_idx], 6)

    sorted_pairs = sorted(zip(class_labels, probs_list), key=lambda x: x[1], reverse=True)
    top3 = [
        {"label": lbl, "confidence": round(score, 6)}
        for lbl, score in sorted_pairs[:3]
    ]

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probs_dict,
        "top3": top3,
    }
