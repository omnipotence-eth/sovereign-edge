"""LangGraph-style routing functions for the orchestrator graph.

These pure functions are used as LangGraph conditional edges and nodes.
They are intentionally dependency-free (no LangGraph import required)
so they can be unit-tested without the full graph being wired up.
"""

from __future__ import annotations

from dataclasses import dataclass

from observability.logging import get_logger

logger = get_logger(__name__, component="orchestrator")

_VALID_INTENTS = frozenset({"spiritual", "career", "intelligence", "creative", "goals"})


@dataclass
class _Message:
    """Minimal message container — satisfies ``messages[n].content`` access pattern."""

    content: str


# ── Edge functions ────────────────────────────────────────────────────────────


def _route_intent(state: dict) -> str:  # type: ignore[type-arg]
    """Return the expert name to route to.

    Falls back to ``"intelligence"`` for any unrecognised intent value.
    """
    intent = state.get("intent", "")
    if intent in _VALID_INTENTS:
        return intent
    logger.debug("route_intent_unknown intent=%s defaulting=intelligence", intent)
    return "intelligence"


def _route_hitl(state: dict) -> str:  # type: ignore[type-arg]
    """Return ``"hitl"`` when human-in-the-loop approval is required, else ``"delivery"``."""
    if state.get("hitl_required"):
        return "hitl"
    return "delivery"


# ── Nodes ─────────────────────────────────────────────────────────────────────


async def delivery_node(state: dict) -> dict:  # type: ignore[type-arg]
    """Assemble the final deliverable message.

    When HITL was required and the human rejected the action
    (``hitl_approved is False``), substitutes a cancellation notice.
    """
    if state.get("hitl_approved") is False:
        content = "Action cancelled."
    else:
        content = state.get("squad_result", "")

    return {"messages": [_Message(content=content)]}
