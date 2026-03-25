"""
Multi-provider LLM gateway using LiteLLM as a library (NOT proxy).

Automatic fallback chain: Groq → Gemini → Cerebras → Mistral → Local
Per-provider token-bucket RPM rate limiting with persistent state.
Daily token usage tracking.
Exponential backoff on transient errors.
Real cost tracking via litellm.completion_cost().
~75MB RAM overhead.

USE get_gateway() — do NOT instantiate LLMGateway() directly.
The singleton preserves TokenBucket and UsageTracker state across calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date

import litellm
from core.config import get_settings
from core.types import RoutingDecision

litellm.set_verbose = False  # type: ignore[attr-defined]
litellm.suppress_debug_info = True  # type: ignore[attr-defined]

# LiteLLM reads API keys from standard env vars (GROQ_API_KEY etc.), but this
# service loads secrets with the SE_ prefix. Bridge them once at module load.
_s = get_settings()
_KEY_MAP: dict[str, str] = {
    "GROQ_API_KEY": _s.groq_api_key,
    "GOOGLE_API_KEY": _s.google_api_key,
    "CEREBRAS_API_KEY": _s.cerebras_api_key,
    "MISTRAL_API_KEY": _s.mistral_api_key,
}
for _env_var, _key_val in _KEY_MAP.items():
    if _key_val:
        os.environ[_env_var] = _key_val

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    model: str
    rpm: int  # Requests per minute
    tpd: int  # Tokens per day (approximate limit)
    priority: int  # Lower = tried first
    env_key: str  # Environment variable name for API key


def _build_providers() -> list[ProviderConfig]:
    """Build provider list using RPM values from Settings (allows per-device tuning)."""
    s = get_settings()
    return [
        ProviderConfig(
            model="groq/llama-3.3-70b-versatile",
            rpm=s.groq_rpm,
            tpd=500_000,
            priority=1,
            env_key="GROQ_API_KEY",
        ),
        ProviderConfig(
            model="gemini/gemini-2.5-flash-preview-04-17",
            rpm=s.gemini_rpm,
            tpd=250_000,
            priority=2,
            env_key="GOOGLE_API_KEY",
        ),
        ProviderConfig(
            model="cerebras/llama-3.3-70b",
            rpm=s.cerebras_rpm,
            tpd=1_000_000,
            priority=3,
            env_key="CEREBRAS_API_KEY",
        ),
        ProviderConfig(
            model="mistral/mistral-small-latest",
            rpm=s.mistral_rpm,
            tpd=33_000_000,
            priority=4,
            env_key="MISTRAL_API_KEY",
        ),
    ]


LOCAL_FALLBACK_MODEL = "ollama/qwen3:0.6b"


@dataclass
class TokenBucket:
    """Token bucket for per-provider RPM enforcement.

    Refills at rate = rpm / 60 tokens per second.
    Max capacity = rpm tokens (1 minute burst).
    """

    rpm: int
    _tokens: float = field(init=False)
    _last_refill: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.rpm)

    def acquire(self) -> bool:
        """Return True if a request slot is available, False if rate-limited."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(float(self.rpm), self._tokens + elapsed * (self.rpm / 60.0))
        self._last_refill = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


@dataclass
class UsageTracker:
    """Tracks daily token usage per provider. Resets at midnight."""

    _date: date = field(default_factory=date.today)
    _usage: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, tokens: int) -> None:
        self._reset_if_new_day()
        self._usage[model] = self._usage.get(model, 0) + tokens

    def get(self, model: str) -> int:
        self._reset_if_new_day()
        return self._usage.get(model, 0)

    def total_today(self) -> int:
        self._reset_if_new_day()
        return sum(self._usage.values())

    def _reset_if_new_day(self) -> None:
        today = date.today()
        if self._date != today:
            self._usage.clear()
            self._date = today


class LLMGateway:
    """
    Multi-provider LLM gateway with RPM rate limiting, daily token tracking, and retry.

    Do NOT instantiate directly — use get_gateway() to access the module singleton.
    Direct instantiation creates a fresh TokenBucket and UsageTracker, which
    means per-provider rate limits and daily token caps are never enforced.
    """

    def __init__(self) -> None:
        self.tracker = UsageTracker()
        self.settings = get_settings()
        self._providers = _build_providers()
        self._buckets: dict[str, TokenBucket] = {
            p.model: TokenBucket(rpm=p.rpm) for p in self._providers
        }

    async def complete(
        self,
        messages: list[dict[str, str]],
        routing: RoutingDecision = RoutingDecision.CLOUD,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        squad: str = "general",
    ) -> dict[str, object]:
        """
        Send a completion request through the fallback chain.

        Returns dict with keys: content, model, tokens_in, tokens_out, latency_ms, cost_usd
        """
        if routing == RoutingDecision.LOCAL:
            return await self._call_local(messages, max_tokens, temperature, squad)

        for provider in sorted(self._providers, key=lambda p: p.priority):
            if self.tracker.get(provider.model) >= provider.tpd:
                logger.debug("provider_daily_limit_reached model=%s", provider.model)
                continue

            if not self._buckets[provider.model].acquire():
                logger.debug("provider_rpm_limited model=%s", provider.model)
                continue

            result = await self._call_with_retry(provider, messages, max_tokens, temperature, squad)
            if result is not None:
                return result

        logger.warning("all_cloud_providers_failed falling_back=local")
        return await self._call_local(messages, max_tokens, temperature, squad)

    async def _call_with_retry(
        self,
        provider: ProviderConfig,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        squad: str,
    ) -> dict[str, object] | None:
        """Attempt provider with exponential backoff. Returns None to try next provider."""
        for attempt in range(_MAX_RETRIES):
            try:
                start = time.monotonic()
                response = await litellm.acompletion(  # type: ignore[attr-defined]
                    model=provider.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=30,
                )
                elapsed = (time.monotonic() - start) * 1000

                tokens_in = response.usage.prompt_tokens if response.usage else 0
                tokens_out = response.usage.completion_tokens if response.usage else 0
                self.tracker.add(provider.model, tokens_in + tokens_out)

                content = response.choices[0].message.content or ""
                try:
                    cost_usd = float(litellm.completion_cost(completion_response=response))  # type: ignore[attr-defined]
                except Exception:
                    cost_usd = 0.0

                logger.info(
                    "llm_response model=%s squad=%s tokens_in=%d tokens_out=%d "
                    "latency_ms=%.1f cost_usd=%.6f",
                    provider.model,
                    squad,
                    tokens_in,
                    tokens_out,
                    elapsed,
                    cost_usd,
                )

                return {
                    "content": content,
                    "model": provider.model,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "latency_ms": round(elapsed, 1),
                    "cost_usd": cost_usd,
                }

            except litellm.RateLimitError:  # type: ignore[attr-defined]
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "provider_rate_limited model=%s attempt=%d retry_in=%.1fs",
                    provider.model,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)

            except litellm.ServiceUnavailableError:  # type: ignore[attr-defined]
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "provider_unavailable model=%s attempt=%d retry_in=%.1fs",
                    provider.model,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)

            except litellm.AuthenticationError:  # type: ignore[attr-defined]
                logger.warning("provider_auth_failed model=%s — check API key", provider.model)
                return None  # Bad key — skip this provider entirely

            except litellm.BadRequestError:  # type: ignore[attr-defined]
                logger.warning("provider_bad_request model=%s", provider.model)
                return None  # Bad request — skip provider, try next

            except (TimeoutError, litellm.Timeout):  # type: ignore[attr-defined]
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "provider_timeout model=%s attempt=%d retry_in=%.1fs",
                    provider.model,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)

            except Exception:
                logger.warning(
                    "provider_unexpected_error model=%s attempt=%d",
                    provider.model,
                    attempt + 1,
                    exc_info=True,
                )
                return None

        logger.warning(
            "provider_exhausted_retries model=%s max_retries=%d",
            provider.model,
            _MAX_RETRIES,
        )
        return None

    async def _call_local(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        squad: str,
    ) -> dict[str, object]:
        """Call local Ollama model as last resort."""
        start = time.monotonic()
        try:
            response = await litellm.acompletion(  # type: ignore[attr-defined]
                model=LOCAL_FALLBACK_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=120,
                api_base=self.settings.ollama_host,
            )
            elapsed = (time.monotonic() - start) * 1000
            content = response.choices[0].message.content or ""
            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0
            try:
                cost_usd = float(litellm.completion_cost(completion_response=response))  # type: ignore[attr-defined]
            except Exception:
                cost_usd = 0.0

            logger.info(
                "local_model_response squad=%s latency_ms=%.1f cost_usd=%.6f",
                squad,
                elapsed,
                cost_usd,
            )
            return {
                "content": content,
                "model": LOCAL_FALLBACK_MODEL,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": round(elapsed, 1),
                "cost_usd": cost_usd,
            }

        except Exception:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error("local_model_failed squad=%s", squad, exc_info=True)
            return {
                "content": (
                    "⚠️ All inference providers are currently unavailable. Please try again shortly."
                ),
                "model": "none",
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": round(elapsed_ms, 1),
                "cost_usd": 0.0,
            }


# ── Module-level singleton ────────────────────────────────────────────────────
# IMPORTANT: Always use get_gateway() instead of LLMGateway().
# Direct instantiation creates a fresh TokenBucket and UsageTracker, which
# means per-provider rate limits and daily token caps are never enforced.

_instance: LLMGateway | None = None


def get_gateway() -> LLMGateway:
    """Return the module-level LLMGateway singleton."""
    global _instance
    if _instance is None:
        _instance = LLMGateway()
    return _instance
