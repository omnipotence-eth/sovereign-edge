"""Career squad — job search intelligence, resume coaching, interview prep."""
from __future__ import annotations

from core.squad import BaseSquad
from core.types import RoutingDecision, SquadName, TaskRequest, TaskResult
from observability.logging import get_logger

logger = get_logger(__name__, squad="career")

_SYSTEM_PROMPT = """\
You are the Career Intelligence of Sovereign Edge — an expert career strategist
specializing in ML Engineering, AI Engineering, and LLM Engineering roles in
the Dallas-Fort Worth metro.  You help with job search strategy, resume
tailoring, cover letter drafting, technical interview preparation, and market
intelligence.  Emphasize the user's differentiators: GRPO fine-tuning,
LangGraph agents, MCP server development, vLLM/TensorRT-LLM serving,
structured outputs, LLMOps, and Blackwell GPU hands-on experience.  Be
direct, specific, and actionable.\
"""

_MORNING_PROMPT = """\
Give a concise morning career briefing (≤ 150 words):
1. One high-value action to take today on the job search (e.g., reach out to
   a specific company, update a resume section, practice a specific topic).
2. One DFW ML/AI market insight or tip.
Keep it motivating and concrete.\
"""


class CareerSquad(BaseSquad):
    """Handles career tasks and generates morning job-search briefings."""

    @property
    def name(self) -> str:
        return SquadName.CAREER

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
            squad=SquadName.CAREER,
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
            max_tokens=200,
        )
        return result["content"]
