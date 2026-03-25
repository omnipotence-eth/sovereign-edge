"""Intelligence squad — research synthesis, AI/ML trend monitoring, news digest."""
from __future__ import annotations

from core.squad import BaseSquad
from core.types import RoutingDecision, SquadName, TaskRequest, TaskResult
from observability.logging import get_logger

logger = get_logger(__name__, squad="intelligence")

_SYSTEM_PROMPT = """\
You are the Intelligence Core of Sovereign Edge — a research analyst and
knowledge synthesizer.  You track AI/ML breakthroughs, distill research papers,
monitor tech news, and surface actionable insights.  Prioritize information
relevant to: LLM fine-tuning, inference optimization, agentic systems,
Blackwell GPU developments, and DFW tech industry.  Be precise, cite sources
when possible, and flag uncertainty explicitly.\
"""

_MORNING_PROMPT = """\
Generate a concise intelligence briefing for today (≤ 200 words):
1. One significant AI/ML development from the past 24 hours worth knowing.
2. One paper or technique relevant to LLM fine-tuning or inference optimization.
3. One actionable insight based on the above.
Be specific — no generalities.\
"""


class IntelligenceSquad(BaseSquad):
    """Research synthesis and trend monitoring."""

    @property
    def name(self) -> str:
        return SquadName.INTELLIGENCE

    async def process(self, task: TaskRequest) -> TaskResult:
        import time

        from llm.gateway import LLMGateway

        gateway = LLMGateway()
        t0 = time.monotonic()

        result = await gateway.complete(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": task.content},
            ],
            max_tokens=2048,
            routing=task.routing,
            squad=self.name,
        )

        return TaskResult(
            task_id=task.task_id,
            squad=SquadName.INTELLIGENCE,
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
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _MORNING_PROMPT},
            ],
            max_tokens=300,
        )
        return result["content"]
