"""Creative squad — writing, content strategy, social media, storytelling."""
from __future__ import annotations

from core.squad import BaseSquad
from core.types import RoutingDecision, SquadName, TaskRequest, TaskResult
from observability.logging import get_logger

logger = get_logger(__name__, squad="creative")

_SYSTEM_PROMPT = """\
You are the Creative Engine of Sovereign Edge — a versatile creative director
and writer.  You help with long-form writing, social media content, content
strategy, storytelling, and brand voice.  Your output should be vivid,
purposeful, and tailored to the requested format and audience.  Offer
structural options when the task is open-ended.\
"""

_MORNING_PROMPT = """\
Generate one creative prompt or micro-challenge for today (≤ 100 words):
Choose from: a writing micro-exercise, a content angle worth exploring, or a
storytelling technique to practice.  Make it specific and immediately
actionable — something completable in 15–20 minutes.\
"""


class CreativeSquad(BaseSquad):
    """Handles creative writing tasks and generates daily creative prompts."""

    @property
    def name(self) -> str:
        return SquadName.CREATIVE

    async def process(self, task: TaskRequest) -> TaskResult:
        import time

        from llm.gateway import LLMGateway

        gateway = LLMGateway()
        t0 = time.monotonic()

        result = await gateway.complete(
            prompt=task.content,
            system=_SYSTEM_PROMPT,
            max_tokens=2048,
        )

        return TaskResult(
            task_id=task.task_id,
            squad=SquadName.CREATIVE,
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
            prompt=_MORNING_PROMPT,
            system=_SYSTEM_PROMPT,
            max_tokens=150,
        )
        return result["content"]
