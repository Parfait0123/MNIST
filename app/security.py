"""
Securite — Verification de l'API Key via header X-API-Key.
"""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import get_settings

settings = get_settings()

api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


async def require_api_key(api_key: str = Security(api_key_header)) -> str:
    """Dependance FastAPI — leve 401 si l'API Key est absente ou incorrecte."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key manquante. Ajoutez le header 'X-API-Key'.",
        )
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key invalide.",
        )
    return api_key
