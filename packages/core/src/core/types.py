from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IntentClass(str, Enum):
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
