from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class IntentClass(StrEnum):
    SPIRITUAL = "spiritual"
    CAREER = "career"
    INTELLIGENCE = "intelligence"
    CREATIVE = "creative"


@dataclass(frozen=True)
class RouterResult:
    intent: IntentClass
    confidence: float

    def is_confident(self, threshold: float = 0.7) -> bool:
        return self.confidence >= threshold


@runtime_checkable
class SquadState(Protocol):
    """Minimal protocol that squad run() methods expect from the graph state dict.

    Using a Protocol avoids circular imports between agents and the orchestrator.
    """

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        ...
