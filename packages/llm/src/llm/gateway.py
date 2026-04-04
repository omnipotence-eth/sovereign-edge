from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any, Literal, TypeAlias, TypeVar

import litellm
import structlog
from core.config import Settings, get_settings
from core.exceptions import LLMError
from pydantic import BaseModel, ValidationError
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
_M = TypeVar("_M", bound=BaseModel)

# Hard ceiling on provider response wait — prevents indefinite hangs on quota resets
_LLM_TIMEOUT_SECONDS = 30.0


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
    """Unified LLM gateway with provider failover, retry, streaming, and timeouts.

    Provider priority: Groq → Gemini Flash → Cerebras → Mistral.
    Falls back to the next provider on RateLimitError or APIError.
    All calls include a 30-second hard timeout to prevent event-loop blocking.
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
                logger.warning("llm.provider_failed", model=model, error_type=type(exc).__name__)
                last_exc = exc
                continue

        msg = f"All LLM providers failed. Last error type: {type(last_exc).__name__}"
        raise LLMError(msg) from last_exc

    async def stream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens as an async generator with provider fallback.

        Usage::

            async for chunk in gateway.stream([Message.user("hello")]):
                print(chunk, end="", flush=True)
        """
        providers = self._settings.active_llm_providers()
        if not providers:
            raise LLMError("No LLM providers configured. Set at least one API key.")

        full_messages: list[dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(m.to_dict() for m in messages)

        return self._stream_with_fallback(
            full_messages, providers, max_tokens=max_tokens, temperature=temperature
        )

    async def _stream_with_fallback(
        self,
        full_messages: list[dict[str, str]],
        providers: list[str],
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        last_exc: Exception | None = None
        for model in providers:
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=full_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    request_timeout=_LLM_TIMEOUT_SECONDS,
                )
                logger.info("llm.stream_start", model=model)
                async for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                return
            except Exception as exc:
                logger.warning(
                    "llm.stream_provider_failed",
                    model=model,
                    error_type=type(exc).__name__,
                )
                last_exc = exc
                continue

        err_type = type(last_exc).__name__
        raise LLMError(f"All LLM providers failed during streaming. Last: {err_type}") from last_exc

    @retry(
        retry=retry_if_exception_type((litellm.RateLimitError, litellm.Timeout)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_with_retry(self, **kwargs: Any) -> Any:  # noqa: ANN401
        # Ensure every call has an explicit timeout regardless of caller kwargs
        kwargs.setdefault("request_timeout", _LLM_TIMEOUT_SECONDS)
        return await litellm.acompletion(**kwargs)

    async def complete_structured(
        self,
        messages: list[Message],
        *,
        schema: type[_M],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> _M:
        """Complete with JSON output and Pydantic schema validation.

        Always uses temperature=0.2 and response_format=json_object.
        Raises LLMError if the response fails schema validation.
        """
        raw = await self.complete(
            messages,
            system=system,
            max_tokens=max_tokens,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            raise LLMError(
                f"LLM response failed {schema.__name__} validation: {type(exc).__name__}"
            ) from exc
