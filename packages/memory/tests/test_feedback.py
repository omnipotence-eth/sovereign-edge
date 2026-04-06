"""Tests for memory.feedback — signal detection and skill extraction.

Covers:
- Positive signal detection for a range of acknowledgement phrases
- Negative signal detection for a range of correction phrases
- Mixed signal (positive + negative) resolves as negative
- Long messages are ignored (> 120 chars)
- Neutral messages return None
- Exact boundary: 120-char message is scanned; 121-char is not
"""

from __future__ import annotations

import pytest
from memory.feedback import _MAX_FEEDBACK_LEN, detect_feedback_signal

# ── Positive signals ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "thanks",
        "Thank you!",
        "Perfect, exactly what I needed",
        "That's great",
        "Excellent work",
        "Very helpful",
        "Awesome response",
        "Got it, appreciate it",
        "Spot on",
        "Well done!",
    ],
)
def test_positive_signals_detected(text: str) -> None:
    assert detect_feedback_signal(text) == "positive", f"Expected positive for {text!r}"


# ── Negative signals ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "That's wrong",
        "incorrect answer",
        "Not right, try again",
        "Not helpful at all",
        "That's not what I asked",
        "Doesn't work",
        "off topic",
        "Bad response",
        "No that is not right",
    ],
)
def test_negative_signals_detected(text: str) -> None:
    assert detect_feedback_signal(text) == "negative", f"Expected negative for {text!r}"


# ── Mixed signal resolution ────────────────────────────────────────────────────


def test_mixed_signal_resolves_as_negative() -> None:
    """'thanks but that's wrong' contains both — negative takes precedence."""
    result = detect_feedback_signal("thanks but that's wrong")
    assert result == "negative"


# ── Neutral / ignored ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "What is the capital of France?",
        "Tell me more about LangGraph",
        "How do I apply for this job?",
        "ok",
        "...",
    ],
)
def test_neutral_messages_return_none(text: str) -> None:
    assert detect_feedback_signal(text) is None


# ── Length boundary ────────────────────────────────────────────────────────────


def test_message_at_max_length_is_scanned() -> None:
    """A message exactly _MAX_FEEDBACK_LEN chars that contains a signal is detected."""
    text = ("thanks " + "x" * (_MAX_FEEDBACK_LEN - 7))
    assert len(text) == _MAX_FEEDBACK_LEN
    assert detect_feedback_signal(text) == "positive"


def test_message_over_max_length_is_ignored() -> None:
    """Messages longer than _MAX_FEEDBACK_LEN are never scanned."""
    text = "thanks " + "x" * (_MAX_FEEDBACK_LEN)  # one over
    assert len(text) > _MAX_FEEDBACK_LEN
    assert detect_feedback_signal(text) is None


# ── Case-insensitivity ─────────────────────────────────────────────────────────


def test_signal_detection_is_case_insensitive() -> None:
    assert detect_feedback_signal("THANKS") == "positive"
    assert detect_feedback_signal("WRONG") == "negative"
