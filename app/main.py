"""
MNIST Digit Recognition API
----------------------------
API FastAPI deployee sur HuggingFace Spaces (Docker).

Un seul modele EfficientNetV2S fine-tune sur MNIST :
  - 10 classes : chiffres 0 a 9
  - Preprocessing : Resize 32x32 + Normalize MNIST (mean=0.1307, std=0.3081)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.model_loader import load_model
from app.routes import router

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

# Rate Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])


# Lifespan — chargement du modele au demarrage
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Demarrage de l'API MNIST — chargement du modele...")
    load_model()
    logger.info("Modele pret. API operationnelle.")
    yield
    logger.info("Arret de l'API MNIST.")


# Application
app = FastAPI(
    title="MNIST Digit Recognition API",
    description="""
## API de Reconnaissance de Chiffres Manuscrits

Utilise un modele **EfficientNetV2S** fine-tune sur le dataset **MNIST** (60 000 images d'entrainement).

| Endpoint | Description | Auth |
|---|---|---|
| `GET /api/v1/health` | Sante de l'API | Non |
| `POST /api/v1/predict` | Prediction via upload fichier | X-API-Key |
| `POST /api/v1/predict/base64` | Prediction via image Base64 | X-API-Key |

### Authentification
Toutes les requetes de prediction necessitent le header **`X-API-Key`**.

### Format d'image accepte
- Upload fichier (`multipart/form-data`) : JPEG, PNG, BMP, TIFF, WebP — max 10 MB
- Base64 JSON : raw Base64 ou avec prefixe `data:image/...;base64,`
- Images en niveaux de gris ou RGB acceptees

### Preprocessing applique
- Conversion en RGB
- Resize 32x32
- Normalisation MNIST : mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081]

### Reponse
```json
{
  "prediction": "7",
  "confidence": 0.9987,
  "probabilities": {"0": 0.0001, "1": 0.0002, ..., "9": 0.0003},
  "top3": [
    {"label": "7", "confidence": 0.9987},
    {"label": "1", "confidence": 0.0008},
    {"label": "9", "confidence": 0.0003}
  ]
}
```
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Routes
app.include_router(router, prefix="/api/v1")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Erreur non geree : {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erreur interne du serveur.", "code": "INTERNAL_ERROR"},
    )


# Root
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "MNIST Digit Recognition API v1.0.0 — voir /docs pour la documentation."
    }
