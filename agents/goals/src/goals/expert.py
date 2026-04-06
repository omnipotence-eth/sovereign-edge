"""
Goals expert — personal goal tracking, progress updates, morning brief.

Delegates to ``goals_subgraph`` (LangGraph) for full pipeline:
  goal_router → [goal_writer | goal_reader] → llm_formatter

Falls back to a direct gateway call when LangGraph is unavailable.
"""

from __future__ import annotations

import time

from core.expert import BaseExpert
from core.types import ExpertName, RoutingDecision, TaskRequest, TaskResult
from observability.logging import get_logger

from goals.store import GoalStore
from goals.subgraph import GoalState, goals_subgraph

logger = get_logger(__name__, component="goals")

_MORNING_SYSTEM = """\
You are the Goals Intelligence of Sovereign Edge — a direct, motivating personal coach.
Format your response for Telegram (single asterisks for *bold*, underscores for _italic_).
No ## headers. Keep it under 120 words total.\
"""


class GoalExpert(BaseExpert):
    """Tracks personal goals and surfaces progress in morning briefs."""

    @property
    def name(self) -> str:
        return ExpertName.GOALS

    async def process(self, task: TaskRequest) -> TaskResult:
        t0 = time.monotonic()

        if goals_subgraph is not None:
            return await self._process_via_subgraph(task, t0)
        return await self._process_direct(task, t0)

    async def _process_via_subgraph(self, task: TaskRequest, t0: float) -> TaskResult:
        initial: GoalState = {
            "query": task.content,
            "routing": task.routing,
            "action": "",
            "goal_id": None,
            "title": "",
            "description": "",
            "target_date": None,
            "progress": 0,
            "store_result": "",
            "response": "",
            "model_used": "",
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        }
        try:
            result = await goals_subgraph.ainvoke(initial)
        except Exception:
            logger.warning("goals_subgraph_invoke_failed — falling back to direct", exc_info=True)
            return await self._process_direct(task, t0)

        return TaskResult(
            task_id=task.task_id,
            expert=ExpertName.GOALS,
            content=result["response"],
            model_used=result["model_used"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_usd=result["cost_usd"],
            routing=task.routing,
            metadata={"nodes": "goal_router,goal_writer/reader,llm_formatter"},
        )

    async def _process_direct(self, task: TaskRequest, t0: float) -> TaskResult:
        """Fallback: single LLM call when LangGraph is unavailable."""
        import json

        from llm.gateway import get_gateway

        gateway = get_gateway()
        history: list[dict[str, str]] = []
        if history_json := task.context.get("history"):
            try:
                history = json.loads(history_json)
            except (ValueError, TypeError):
                pass

        content = await gateway.complete(
            messages=[
                {"role": "system", "content": _MORNING_SYSTEM},
                *history,
                {"role": "user", "content": task.content},
            ],
            max_tokens=300,
            routing=task.routing,
            expert=self.name,
        )
        return TaskResult(
            task_id=task.task_id,
            expert=ExpertName.GOALS,
            content=content,
            model_used="",
            tokens_in=0,
            tokens_out=0,
            latency_ms=(time.monotonic() - t0) * 1000,
            cost_usd=0.0,
            routing=task.routing,
        )

    async def morning_brief(self) -> str:
        """Surface up to 3 urgent goals + one motivational action sentence."""

        store = GoalStore()
        urgent = store.get_urgent(limit=3)
        if not urgent:
            return ""

        lines = ["*Goals check-in:*"]
        for i, g in enumerate(urgent, 1):
            due = f", due {g.target_date}" if g.target_date else ""
            lines.append(f"{i}. {g.title} — _{g.progress_pct}% complete{due}_")

        # One motivating nudge via LLM
        try:
            from llm.gateway import get_gateway

            gateway = get_gateway()
            goal_summary = "\n".join(lines[1:])
            action = await gateway.complete(
                messages=[
                    {"role": "system", "content": _MORNING_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Active goals:\n{goal_summary}\n\n"
                            "Give me exactly ONE concrete action I can take today "
                            "to make progress on the most urgent goal. One sentence only."
                        ),
                    },
                ],
                max_tokens=60,
                routing=RoutingDecision.CLOUD,
                expert=self.name,
            )
            lines.append(f"\n*Action:* {action.strip()}")
        except Exception:
            logger.warning("goals_morning_brief_llm_failed", exc_info=True)

        return "\n".join(lines)

    async def health_check(self) -> bool:
        try:
            GoalStore().list_goals()
            return True
        except Exception:
            logger.warning("goals_health_check_failed", exc_info=True)
            return False
