"""
Routes FastAPI — 2 endpoints de prediction (upload + base64) + health check.
"""

import logging
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status

from app.config import get_settings
from app.security import require_api_key
from app.model_loader import get_model, _MODEL
from app.preprocessing import preprocess_bytes, preprocess_base64
from app.inference import run_inference
from app.schemas import (
    PredictionResponse,
    Base64ImageRequest,
    HealthResponse,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


# HEALTH CHECK — public, pas d'auth

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Sante de l'API",
    tags=["System"],
)
async def health_check():
    from app.model_loader import _MODEL
    return {
        "status": "ok",
        "model_loaded": _MODEL is not None,
        "version": "1.0.0",
    }


# HELPERS

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


async def _read_and_validate_upload(file: UploadFile) -> bytes:
    """Lit et valide le fichier uploade."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Type de fichier non supporte : {file.content_type}. "
                "Formats acceptes : JPEG, PNG, BMP, TIFF, WebP."
            ),
        )
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image trop volumineuse. Limite : 10 MB.",
        )
    return image_bytes


# PREDICT — UPLOAD fichier

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Reconnaissance de chiffre manuscrit — upload fichier",
    tags=["Prediction"],
)
async def predict_upload(
    file: UploadFile = File(..., description="Image du chiffre manuscrit (JPEG/PNG/BMP/TIFF/WebP)"),
    _: str = Depends(require_api_key),
):
    """
    Predit le chiffre manuscrit (0-9) contenu dans l'image uploadee.

    L'image peut etre en niveaux de gris ou RGB — elle sera convertie
    automatiquement en RGB avant le preprocessing.
    """
    image_bytes = await _read_and_validate_upload(file)
    try:
        tensor = preprocess_bytes(image_bytes)
    except Exception as e:
        logger.error(f"Erreur preprocessing upload : {e}")
        raise HTTPException(status_code=422, detail=f"Impossible de decoder l'image : {e}")

    result = run_inference(get_model(), tensor, settings.classes)
    return result


# PREDICT — BASE64

@router.post(
    "/predict/base64",
    response_model=PredictionResponse,
    summary="Reconnaissance de chiffre manuscrit — image Base64",
    tags=["Prediction"],
)
async def predict_base64(
    body: Base64ImageRequest,
    _: str = Depends(require_api_key),
):
    """
    Predit le chiffre manuscrit (0-9) depuis une image encodee en Base64.

    Accepte le raw Base64 ou le format data URI : `data:image/png;base64,...`
    """
    try:
        tensor = preprocess_base64(body.image)
    except Exception as e:
        logger.error(f"Erreur preprocessing base64 : {e}")
        raise HTTPException(status_code=422, detail=f"Impossible de decoder le Base64 : {e}")

    result = run_inference(get_model(), tensor, settings.classes)
    return result
