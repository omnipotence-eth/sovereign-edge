"""Tests for core.reflection — ReflectionResult model validation.

Covers:
- score field: valid range 1-5, rejects 0 and 6
- improved flag semantics: True only when response was rewritten
- ReflectionResult is a valid Pydantic model (serialisable / parseable)
- reflect_response() falls back gracefully when LLM fails
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from core.reflection import ReflectionResult, reflect_response
from pydantic import ValidationError

# ── ReflectionResult model validation ─────────────────────────────────────────


def test_reflection_result_valid_min_score() -> None:
    r = ReflectionResult(score=1, critique="poor", improved_response="better", improved=True)
    assert r.score == 1


def test_reflection_result_valid_max_score() -> None:
    r = ReflectionResult(score=5, critique="perfect", improved_response="same", improved=False)
    assert r.score == 5


def test_reflection_result_score_below_min_raises() -> None:
    with pytest.raises(ValidationError):
        ReflectionResult(score=0, critique="x", improved_response="x", improved=False)


def test_reflection_result_score_above_max_raises() -> None:
    with pytest.raises(ValidationError):
        ReflectionResult(score=6, critique="x", improved_response="x", improved=False)


def test_reflection_result_is_serialisable() -> None:
    r = ReflectionResult(
        score=4,
        critique="Good response.",
        improved_response="The original response.",
        improved=False,
    )
    data = r.model_dump()
    assert data["score"] == 4
    assert data["improved"] is False


# ── reflect_response() fallback ────────────────────────────────────────────────


async def test_reflect_response_returns_none_on_gateway_failure() -> None:
    """If the LLM gateway raises, reflect_response must return None (not re-raise)."""
    mock_gateway = AsyncMock()
    mock_gateway.complete_structured.side_effect = RuntimeError("network error")

    with patch("llm.gateway.get_gateway", return_value=mock_gateway):
        result = await reflect_response(
            query="tell me about LangGraph",
            response="LangGraph is a framework for building stateful agents.",
            intent="intelligence",
            expert="intelligence",
        )

    assert result is None


async def test_reflect_response_returns_result_on_success() -> None:
    """When the gateway succeeds, reflect_response returns the ReflectionResult."""
    expected = ReflectionResult(
        score=5,
        critique="Excellent response.",
        improved_response="LangGraph is a framework for building stateful agents.",
        improved=False,
    )
    mock_gateway = AsyncMock()
    mock_gateway.complete_structured = AsyncMock(return_value=expected)

    with patch("llm.gateway.get_gateway", return_value=mock_gateway):
        result = await reflect_response(
            query="what is LangGraph",
            response="LangGraph is a framework for building stateful agents.",
            intent="intelligence",
            expert="intelligence",
        )

    assert result is not None
    assert result.score == 5
    assert result.improved is False
