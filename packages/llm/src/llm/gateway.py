from __future__ import annotations

import logging
from typing import Any, Literal, TypeAlias

import litellm
import structlog
from core.config import Settings, get_settings
from core.exceptions import LLMError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

# litellm is noisy by default
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

Role: TypeAlias = Literal["system", "user", "assistant"]


class Message:
    __slots__ = ("role", "content")

    def __init__(self, role: Role, content: str) -> None:
        self.role: Role = role
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def system(cls, content: str) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls("assistant", content)


class LLMGateway:
    """Unified LLM gateway with provider failover and retry.

    Provider priority: Groq → Gemini Flash → Cerebras → Mistral.
    Falls back to the next provider on RateLimitError or APIError.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        providers = self._settings.active_llm_providers()
        if not providers:
            raise LLMError("No LLM providers configured. Set at least one API key.")

        full_messages: list[dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(m.to_dict() for m in messages)

        last_exc: Exception | None = None
        for model in providers:
            try:
                response = await self._call_with_retry(
                    model=model,
                    messages=full_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                logger.info("llm.complete", model=model, tokens=response.usage.total_tokens)
                content = response.choices[0].message.content
                if not content:
                    raise LLMError(f"Empty response from {model}")
                return content
            except LLMError:
                raise
            except Exception as exc:
                logger.warning("llm.provider_failed", model=model, error=str(exc))
                last_exc = exc
                continue

        raise LLMError(f"All LLM providers failed. Last error: {last_exc}") from last_exc

    @retry(
        retry=retry_if_exception_type((litellm.RateLimitError, litellm.Timeout)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_with_retry(self, **kwargs: Any) -> Any:  # noqa: ANN401
        return await litellm.acompletion(**kwargs)

    async def complete_structured(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Complete with JSON output enforcement via response_format."""
        return await self.complete(
            messages,
            system=system,
            max_tokens=max_tokens,
            temperature=0.2,
            response_format=response_format or {"type": "json_object"},
        )
