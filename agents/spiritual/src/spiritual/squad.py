"""Spiritual squad — faith formation, prayer, scripture study, devotionals."""
from __future__ import annotations

from core.squad import BaseSquad
from core.types import RoutingDecision, SquadName, TaskRequest, TaskResult
from observability.logging import get_logger

logger = get_logger(__name__, squad="spiritual")

_SYSTEM_PROMPT = """\
You are the Spiritual Intelligence of Sovereign Edge — a contemplative guide
rooted in Christian faith.  You help with scripture study, prayer composition,
theological questions, and daily devotionals.  Respond with depth, warmth, and
scriptural grounding.  Always cite chapter and verse when quoting scripture.\
"""


class SpiritualSquad(BaseSquad):
    """Handles faith-formation tasks and generates morning devotionals."""

    @property
    def name(self) -> str:
        return SquadName.SPIRITUAL

    async def process(self, task: TaskRequest) -> TaskResult:
        import time

        from llm.gateway import LLMGateway

        gateway = LLMGateway()
        t0 = time.monotonic()

        result = await gateway.complete(
            prompt=task.content,
            system=_SYSTEM_PROMPT,
            max_tokens=1024,
        )

        return TaskResult(
            task_id=task.task_id,
            squad=SquadName.SPIRITUAL,
            content=result["content"],
            model_used=result["model"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_usd=result["cost_usd"],
            routing=RoutingDecision.CLOUD,
        )

    async def morning_brief(self) -> str:
        from llm.gateway import LLMGateway

        gateway = LLMGateway()
        result = await gateway.complete(
            prompt=(
                "Generate a brief morning devotional for today: a single scripture verse "
                "with 2–3 sentences of reflection and a one-sentence prayer."
            ),
            system=_SYSTEM_PROMPT,
            max_tokens=300,
        )
        return result["content"]
