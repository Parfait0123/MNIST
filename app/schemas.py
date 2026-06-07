from pydantic import BaseModel, Field
from typing import Optional


# Requetes

class Base64ImageRequest(BaseModel):
    image: str = Field(
        ...,
        description="Image encodee en Base64 (raw ou avec prefixe data:image/...;base64,)",
        examples=["data:image/png;base64,iVBORw0KGgo..."],
    )


# Reponses

class PredictionClass(BaseModel):
    label: str = Field(..., description="Chiffre predit (0 a 9)")
    confidence: float = Field(..., description="Score de confiance (0.0 a 1.0)", ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Chiffre predit (0 a 9)")
    confidence: float = Field(..., description="Confiance sur la classe predite (0.0 a 1.0)", ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(
        ..., description="Probabilites pour chacun des 10 chiffres (0 a 9)"
    )
    top3: list[PredictionClass] = Field(..., description="Top 3 des chiffres les plus probables")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
