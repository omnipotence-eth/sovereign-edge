"""
Response reflection -- self-critique and quality improvement.

reflect_response() runs a structured LLM self-critique over an expert's response.
The LLM scores quality 1-5 and rewrites the response when the score is below threshold.

Design decisions:
- Only runs for CLOUD-routed, non-cached responses (latency acceptable there).
- Uses complete_structured() -- Pydantic guarantees the score field is a valid int 1-5.
- Failure is silent: callers fall back to the original response.
- Skill auto-extraction is triggered by the orchestrator when score >= 4.

Controlled by SE_REFLECT_ENABLED (default: False).
"""

from __future__ import annotations

from observability.logging import get_logger
from pydantic import BaseModel, Field

logger = get_logger(__name__, component="core")

__all__ = ["ReflectionResult", "reflect_response"]

_SCORE_THRESHOLD = 4  # Rewrite response when score < this

_REFLECT_SYSTEM = """\
You are a quality evaluator for a personal AI assistant.

Score the response 1-5:
  1-2  Incomplete, inaccurate, or unhelpful
  3    Partially useful but missing actionable detail
  4    Complete, accurate, and actionable
  5    Exceptional -- nothing to improve

Rules:
- If score < 4: rewrite improved_response to earn score 4+. Keep the same format.
- If score >= 4: copy the original response unchanged into improved_response.
- Set improved=true only when you actually rewrote the response.
- Keep critique to 1-2 sentences.
"""


class ReflectionResult(BaseModel):
    """Structured output of the self-critique pass."""

    score: int = Field(ge=1, le=5, description="Quality score 1-5")
    critique: str = Field(description="1-2 sentence explanation of the score")
    improved_response: str = Field(
        description="Rewritten response when score < 4; original otherwise"
    )
    improved: bool = Field(description="True only when the response was actually rewritten")


async def reflect_response(
    *,
    query: str,
    response: str,
    intent: str,
    expert: str,
) -> ReflectionResult | None:
    """Run one self-critique pass over *response*.

    Returns None on any failure -- callers must fall back to the original response.

    Args:
        query: The original user query that produced this response.
        response: The expert's generated response to critique.
        intent: The classified intent (e.g. "career", "spiritual").
        expert: Expert name for logging.
    """
    import asyncio

    from llm.gateway import get_gateway

    gateway = get_gateway()
    try:
        # Estimate tokens needed: room to reproduce the full response plus critique
        max_tokens = min(len(response.split()) * 3 + 200, 3000)

        async with asyncio.timeout(15):
            result = await gateway.complete_structured(
                messages=[
                    {"role": "system", "content": _REFLECT_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Intent: {intent}\n"
                            f"User query: {query}\n\n"
                            f"Response to evaluate:\n{response}"
                        ),
                    },
                ],
                response_model=ReflectionResult,
                max_tokens=max_tokens,
                temperature=0.2,
                expert=f"{expert}_reflect",
            )

        if result:
            logger.info(
                "reflection_complete expert=%s intent=%s score=%d improved=%s",
                expert,
                intent,
                result.score,
                result.improved,
            )

        return result

    except Exception:
        logger.debug("reflection_failed expert=%s", expert, exc_info=True)
        return None
