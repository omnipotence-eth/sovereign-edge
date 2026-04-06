from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from core.config import Settings
from llm.gateway import LLMGateway, Message


@pytest.fixture()
def settings_no_keys() -> Settings:
    return Settings(groq_api_key="", gemini_api_key="", cerebras_api_key="", mistral_api_key="")


@pytest.fixture()
def settings_groq() -> Settings:
    return Settings(groq_api_key="gsk_test_key")


def _mock_response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage.total_tokens = 42
    return resp


# ── Message helpers ───────────────────────────────────────────────────────────


def test_message_to_dict() -> None:
    m = Message.user("hello")
    assert m.to_dict() == {"role": "user", "content": "hello"}


def test_message_system_factory() -> None:
    m = Message.system("you are helpful")
    assert m.role == "system"


def test_message_assistant_factory() -> None:
    m = Message.assistant("I can help")
    assert m.role == "assistant"


# ── No providers configured ───────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_complete_falls_back_to_local_when_no_providers(settings_no_keys: Settings) -> None:
    """When no cloud keys are configured, complete() falls back to local Ollama."""
    gw = LLMGateway(settings=settings_no_keys)
    with patch.object(gw, "_call_local", new=AsyncMock(return_value={"content": "local answer"})):
        result = await gw.complete([Message.user("hi")])
    assert result == "local answer"


# ── Successful completion ─────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_complete_returns_content(settings_groq: Settings) -> None:
    gw = LLMGateway(settings=settings_groq)
    mock_resp = _mock_response("test answer")

    with patch.object(gw, "_call_with_retry", new=AsyncMock(return_value=mock_resp)):
        result = await gw.complete([Message.user("question")])

    assert result == "test answer"


@pytest.mark.asyncio()
async def test_complete_injects_system_prompt(settings_groq: Settings) -> None:
    gw = LLMGateway(settings=settings_groq)
    mock_resp = _mock_response("answer")
    captured: list[dict] = []

    async def capture(**kwargs: object) -> MagicMock:
        captured.append(dict(kwargs))
        return mock_resp

    with patch.object(gw, "_call_with_retry", new=capture):
        await gw.complete([Message.user("q")], system="be helpful")

    messages = captured[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "be helpful"


# ── Provider failover ─────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_complete_falls_over_to_next_provider() -> None:
    import litellm

    settings = Settings(groq_api_key="gsk_test", gemini_api_key="gem_test")
    gw = LLMGateway(settings=settings)
    mock_resp = _mock_response("fallback answer")
    call_count = 0

    async def failing_first(**kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise litellm.RateLimitError("rate limited", llm_provider="groq", model="llama")
        return mock_resp

    with patch.object(gw, "_call_with_retry", new=failing_first):
        result = await gw.complete([Message.user("q")])

    assert result == "fallback answer"
    assert call_count == 2


@pytest.mark.asyncio()
async def test_complete_falls_back_to_local_when_all_fail() -> None:
    """When all cloud providers raise, complete() falls back to local Ollama."""
    import litellm

    settings = Settings(groq_api_key="gsk_test")
    gw = LLMGateway(settings=settings)

    async def always_fail(**kwargs: object) -> None:
        raise litellm.APIConnectionError("down", llm_provider="groq", model="llama")

    local_mock = AsyncMock(return_value={"content": "local fallback"})
    with patch.object(gw, "_call_with_retry", new=always_fail):
        with patch.object(gw, "_call_local", new=local_mock):
            result = await gw.complete([Message.user("q")])
    assert result == "local fallback"


# ── Empty response guard ──────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_complete_falls_back_on_empty_content(settings_groq: Settings) -> None:
    """Empty provider response skips that provider; falls back to local Ollama."""
    gw = LLMGateway(settings=settings_groq)
    mock_resp = _mock_response("")

    local_mock = AsyncMock(return_value={"content": "local answer"})
    with patch.object(gw, "_call_with_retry", new=AsyncMock(return_value=mock_resp)):
        with patch.object(gw, "_call_local", new=local_mock):
            result = await gw.complete([Message.user("q")])
    assert result == "local answer"
