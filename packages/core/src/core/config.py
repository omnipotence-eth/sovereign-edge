"""Application configuration loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global settings. Load from environment or .env file."""

    # Paths
    project_root: Path = Path("/home/jetson/sovereign-edge")
    ssd_root: Path = Path("/ssd/sovereign-edge")
    lancedb_path: Path = Path("/ssd/sovereign-edge/lancedb")
    logs_path: Path = Path("/ssd/sovereign-edge/logs")
    models_path: Path = Path("/ssd/sovereign-edge/models")

    # Ollama
    ollama_host: str = "http://127.0.0.1:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    local_llm_model: str = "qwen3:0.6b"

    # Cloud API Keys (loaded from SOPS-decrypted env vars)
    groq_api_key: str = ""
    google_api_key: str = ""
    cerebras_api_key: str = ""
    mistral_api_key: str = ""
    telegram_bot_token: str = ""
    alpha_vantage_key: str = ""
    jina_api_key: str = ""

    # Cloud API Rate Limits (requests per minute)
    groq_rpm: int = 30
    gemini_rpm: int = 15
    cerebras_rpm: int = 30
    mistral_rpm: int = 2

    # Feature Flags
    voice_enabled: bool = False
    creative_enabled: bool = True
    debug_mode: bool = False

    # Scheduling
    morning_wake_hour: int = 5
    morning_wake_minute: int = 0
    timezone: str = "US/Central"

    model_config = {"env_prefix": "SE_", "env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Singleton settings instance."""
    return Settings()
