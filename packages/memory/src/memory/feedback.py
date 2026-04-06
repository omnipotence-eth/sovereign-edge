"""
Implicit feedback detection and skill auto-extraction.

Two capabilities:

detect_feedback_signal
  Keyword-based sentiment detection from user messages.
  Zero cost — no LLM call. Returns "positive", "negative", or None.
  Used by the orchestrator to update SkillLibrary scores in real time.

extract_skill_pattern
  LLM-based extraction of a reusable 1-sentence technique from a high-quality
  response. Called as a background task after reflection scores ≥ 4.
  Grows the SkillLibrary autonomously so future responses benefit.
"""

from __future__ import annotations

from typing import Literal

from observability.logging import get_logger

logger = get_logger(__name__, component="memory")

__all__ = ["detect_feedback_signal", "extract_skill_pattern"]

# ── Feedback signal keyword sets ───────────────────────────────────────────────
# Ordered sets — checked via substring scan, not exact match.
# Negative checked first so "thanks but that was wrong" → negative.

_NEGATIVE_SIGNALS: frozenset[str] = frozenset(
    {
        "wrong",
        "incorrect",
        "not right",
        "that's not",
        "that isn't",
        "not what i",
        "not helpful",
        "doesn't work",
        "did not work",
        "missed the point",
        "off topic",
        "try again",
        "bad response",
        "that's wrong",
        "no that",
        "not accurate",
    }
)

_POSITIVE_SIGNALS: frozenset[str] = frozenset(
    {
        "thanks",
        "thank you",
        "perfect",
        "exactly",
        "great",
        "excellent",
        "helpful",
        "awesome",
        "love it",
        "nice work",
        "good job",
        "correct",
        "that's it",
        "that works",
        "got it",
        "brilliant",
        "spot on",
        "well done",
        "appreciate",
    }
)

# Only scan short messages — long messages are follow-ups, not feedback.
_MAX_FEEDBACK_LEN = 120


def detect_feedback_signal(text: str) -> Literal["positive", "negative"] | None:
    """Detect implicit positive or negative feedback from a user message.

    Uses keyword matching — zero LLM cost.
    Returns None for neutral / ambiguous / long messages.

    Negative is checked first so mixed signals ("thanks but wrong") → negative.
    """
    if len(text) > _MAX_FEEDBACK_LEN:
        return None

    lower = text.lower().strip()

    if any(sig in lower for sig in _NEGATIVE_SIGNALS):
        return "negative"
    if any(sig in lower for sig in _POSITIVE_SIGNALS):
        return "positive"
    return None


# ── Skill auto-extraction ──────────────────────────────────────────────────────


class SkillPattern:
    """Parsed output of the skill extraction LLM call."""

    __slots__ = ("pattern",)

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern


async def extract_skill_pattern(response: str, intent: str) -> str | None:
    """Extract a reusable 1-sentence technique from a high-quality response.

    Called when reflection scores the response 4 or 5. The extracted pattern
    is stored in SkillLibrary so future responses benefit from the same approach.

    Returns None if extraction fails — callers must treat this as a no-op.
    """
    from llm.gateway import get_gateway
    from pydantic import BaseModel

    class _PatternModel(BaseModel):
        pattern: str

    gateway = get_gateway()
    try:
        result = await gateway.complete_structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract a single reusable skill pattern (≤ 120 chars) from the given "
                        f"response. Describe *how* to approach this type of {intent} task — "
                        "not the specific content of the answer. Write a general technique, "
                        "not a summary."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Response:\n\n{response[:800]}",
                },
            ],
            response_model=_PatternModel,
            max_tokens=60,
            temperature=0.2,
            expert="skill_extract",
        )
        if result and len(result.pattern.strip()) > 10:
            pattern = result.pattern.strip()[:120]
            logger.info("skill_pattern_extracted intent=%s pattern=%r", intent, pattern)
            return pattern
    except Exception:
        logger.debug("skill_extraction_failed intent=%s", intent, exc_info=True)
    return None
