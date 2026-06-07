from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # API Security
    api_key: str = os.getenv("API_KEY", "change-me-in-production")
    api_key_header: str = "X-API-Key"

    # Model config — CNN Custom sur MNIST (notebook)
    input_size: int = 32

    # Preprocessing stats MNIST (mean/std sur images grayscale)
    normalize_mean: list[float] = [0.1307]
    normalize_std: list[float] = [0.3081]

    # Kaggle model repo
    kaggle_model_handle: str = "parfaitbotchi1/mnist/pyTorch/default"
    model_filename: str = "best_model_1.pth"

    # Classes MNIST
    classes: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # Rate limiting
    rate_limit: str = "60/minute"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
