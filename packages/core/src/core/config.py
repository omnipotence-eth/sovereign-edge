from __future__ import annotations

import functools
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM providers (priority order: Groq → Gemini → Cerebras → Mistral) ──
    groq_api_key: str = ""
    gemini_api_key: str = ""
    cerebras_api_key: str = ""
    mistral_api_key: str = ""

    # ── Telegram ─────────────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_allowed_user_id: int = 0  # John's Telegram user ID

    # ── Intelligence squad ───────────────────────────────────────────────────
    alpha_vantage_api_key: str = ""
    fmp_api_key: str = ""  # Financial Modeling Prep — free tier 250 req/day for transcripts

    # ── Voice service ─────────────────────────────────────────────────────────
    stt_model: str = "base.en"  # faster-whisper: tiny.en, base.en, small.en, medium.en
    watchlist: list[str] = Field(default_factory=lambda: ["NVDA", "MSFT", "GOOGL", "META"])
    market_alert_threshold: float = 0.02  # 2% move triggers alert

    # ── Career squad ─────────────────────────────────────────────────────────
    job_target_roles: list[str] = Field(
        default_factory=lambda: ["ML Engineer", "AI Engineer", "LLM Engineer"]
    )
    job_target_location: str = "Dallas Fort Worth TX"
    job_target_cities: list[str] = Field(
        default_factory=lambda: [
            "Dallas",
            "Fort Worth",
            "Plano",
            "Irving",
            "Arlington",
            "Frisco",
            "McKinney",
            "Allen",
            "Richardson",
            "Carrollton",
            "Southlake",
            "Grapevine",
            "Addison",
        ]
    )
    resume_path: Path = Path.home() / "Documents" / "Job Search"

    # ── Router / model paths ─────────────────────────────────────────────────
    router_model_path: Path = Path("data/models/router.onnx")
    router_tokenizer_name: str = "distilbert-base-uncased"
    router_confidence_threshold: float = 0.7

    # ── Memory / vector store ────────────────────────────────────────────────
    lancedb_path: Path = Path("data/lancedb")
    skill_db_path: Path = Path("data/skills.db")
    mem0_user_id: str = "john"

    # ── Observability ────────────────────────────────────────────────────────
    log_json: bool = False
    log_level: str = "INFO"
    otel_endpoint: str = ""  # empty = no OTEL export

    # ── Scheduling ───────────────────────────────────────────────────────────
    morning_brief_hour: Annotated[int, Field(ge=0, le=23)] = 6
    job_scan_hour: Annotated[int, Field(ge=0, le=23)] = 9
    market_summary_hour: Annotated[int, Field(ge=0, le=23)] = 18

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    def active_llm_providers(self) -> list[str]:
        """Return LiteLLM model strings for configured providers, in priority order."""
        providers = []
        if self.groq_api_key:
            providers.append("groq/llama-3.3-70b-versatile")
        if self.gemini_api_key:
            providers.append("gemini/gemini-1.5-flash")
        if self.cerebras_api_key:
            providers.append("cerebras/llama3.1-70b")
        if self.mistral_api_key:
            providers.append("mistral/mistral-large-latest")
        return providers

    def has_router_model(self) -> bool:
        return self.router_model_path.exists()


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
