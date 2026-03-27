"""
Director agent — LangGraph-powered multi-squad orchestrator.

Replaces single-shot intent routing with a plan-and-execute graph that can
chain squads when a query spans multiple domains.

Examples of multi-squad queries:
  "Research the latest GRPO papers and write a LinkedIn post about them"
  → intelligence (fetch + synthesise) → creative (draft the post)

  "Find ML engineer jobs at companies working on inference optimisation"
  → intelligence (which companies?) → career (job search those companies)

Single-squad queries pass through with zero overhead — the director plan
resolves to one node and exits immediately.

Graph nodes:
  plan      LLM decides the squad chain for this query
  spiritual / career / intelligence / creative  squad execution nodes
  merge     combines multi-squad outputs into a final response (optional)

Usage:
    from director.graph import DirectorGraph

    graph = DirectorGraph(squads={"intelligence": squad, "creative": squad})
    result = await graph.run(task_request)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.squad import BaseSquad
from core.types import Intent, RoutingDecision, SquadName, TaskRequest, TaskResult
from llm.gateway import get_gateway

try:
    from langgraph.graph import END, StateGraph
    from typing_extensions import TypedDict

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore[assignment,misc]
    END = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ── Squad names the director can route to ─────────────────────────────────────
_ROUTABLE_SQUADS: list[str] = [
    SquadName.SPIRITUAL,
    SquadName.CAREER,
    SquadName.INTELLIGENCE,
    SquadName.CREATIVE,
]

_DIRECTOR_SYSTEM = """\
You are the Director of Sovereign Edge — a multi-squad personal AI system.
Your job is to analyse the user's request and produce a routing plan.

Available squads:
  spiritual    — Bible, faith, prayer, devotionals, theology
  career       — job search, resume, interviews, salary, LinkedIn
  intelligence — AI/ML research, arXiv papers, tech news, trends
  creative     — content writing, social media posts, scripts, blogs

RULES:
1. Return a JSON object ONLY — no prose, no markdown fences.
2. The "squads" list contains 1–3 squad names in execution order.
3. The "rationale" is one sentence explaining why.
4. Use multiple squads ONLY when the query clearly needs it.
   Single-squad queries are the common case — do not over-engineer.

OUTPUT FORMAT (strict JSON):
{
  "squads": ["<squad1>", "<squad2>"],
  "rationale": "<one sentence>",
  "context_pass": true
}

"context_pass": true means pass the first squad's output as context to the next.
Set to false if the squads are independent (parallel is not yet implemented).

EXAMPLES:
  Input: "What does Psalm 23 mean?"
  Output: {"squads": ["spiritual"], "rationale": "Pure scripture question.", "context_pass": false}

  Input: "Research the latest GRPO papers and write a LinkedIn post about them"
  Output: {"squads": ["intelligence", "creative"], "rationale": "Research first, then draft the post using those findings.", "context_pass": true}

  Input: "Find ML engineer jobs at companies building inference chips"
  Output: {"squads": ["intelligence", "career"], "rationale": "Intelligence identifies target companies, career searches those.", "context_pass": true}
"""


# ── LangGraph state ───────────────────────────────────────────────────────────

class DirectorState(TypedDict):
    request: TaskRequest
    plan: list[str]          # ordered squad names
    context_pass: bool       # carry output between squads
    results: list[str]       # accumulated squad outputs
    final_output: str
    error: str


# ── Director graph ────────────────────────────────────────────────────────────

class DirectorGraph:
    """LangGraph-powered director that plans and executes multi-squad chains.

    Falls back to single-squad dispatch when LangGraph is unavailable or
    when the director LLM call fails — preserving backward compatibility
    with the existing Orchestrator.
    """

    def __init__(self, squads: dict[str, BaseSquad]) -> None:
        self._squads = squads
        self._graph = self._build_graph() if _LANGGRAPH_AVAILABLE else None
        if not _LANGGRAPH_AVAILABLE:
            logger.warning(
                "director_langgraph_unavailable — install langgraph>=0.3 for multi-squad chains"
            )

    def _build_graph(self) -> Any:
        """Construct the StateGraph. Called once at init."""
        graph = StateGraph(DirectorState)

        graph.add_node("plan", self._plan_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("merge", self._merge_node)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "execute")
        graph.add_conditional_edges(
            "execute",
            self._should_continue,
            {"continue": "execute", "merge": "merge", END: END},
        )
        graph.add_edge("merge", END)

        return graph.compile()

    # ── Node implementations ──────────────────────────────────────────────────

    async def _plan_node(self, state: DirectorState) -> dict[str, Any]:
        """Ask the LLM to produce a squad execution plan for this query."""
        request = state["request"]
        gateway = get_gateway()

        try:
            result = await gateway.complete(
                messages=[
                    {"role": "system", "content": _DIRECTOR_SYSTEM},
                    {"role": "user", "content": request.content},
                ],
                max_tokens=200,
                temperature=0.1,   # low temp — we want deterministic routing
                routing=RoutingDecision.CLOUD,
                squad="director",
            )
            plan_json = _extract_json(result["content"])
            squads = [s for s in plan_json.get("squads", []) if s in _ROUTABLE_SQUADS]
            context_pass = bool(plan_json.get("context_pass", False))

            if not squads:
                squads = [_intent_to_squad(request.intent)]

            logger.info(
                "director_plan squads=%s context_pass=%s query_len=%d",
                squads, context_pass, len(request.content),
            )
            return {"plan": squads, "context_pass": context_pass, "results": [], "error": ""}

        except Exception:
            logger.warning("director_plan_failed — falling back to intent routing", exc_info=True)
            fallback = _intent_to_squad(request.intent)
            return {"plan": [fallback], "context_pass": False, "results": [], "error": ""}

    async def _execute_node(self, state: DirectorState) -> dict[str, Any]:
        """Execute the next squad in the plan."""
        plan = list(state["plan"])
        results = list(state.get("results", []))

        if not plan:
            return {"plan": plan, "results": results}

        squad_name = plan.pop(0)
        squad = self._squads.get(squad_name)
        if squad is None:
            logger.warning("director_squad_missing squad=%s", squad_name)
            return {"plan": plan, "results": results}

        # Build request — inject prior squad output as context when context_pass=True
        request = state["request"]
        if state.get("context_pass") and results:
            prior_context = "\n\n---\nPrior squad output:\n" + results[-1]
            # Append prior output to the user content so the squad sees it
            enriched_content = request.content + prior_context
            request = request.model_copy(update={"content": enriched_content})

        try:
            result = await squad.process(request)
            results.append(result.content)
            logger.info("director_execute squad=%s chars=%d", squad_name, len(result.content))
        except Exception:
            logger.error("director_execute_failed squad=%s", squad_name, exc_info=True)
            results.append(f"[{squad_name} unavailable]")

        return {"plan": plan, "results": results}

    async def _merge_node(self, state: DirectorState) -> dict[str, Any]:
        """Merge multi-squad outputs into a coherent final response."""
        results = state.get("results", [])
        if not results:
            return {"final_output": "No results from any squad."}

        if len(results) == 1:
            return {"final_output": results[0]}

        # Ask the LLM to weave outputs into a single coherent response
        gateway = get_gateway()
        squad_names = []  # recover from original plan — it's been consumed, use results count
        merge_prompt = (
            "You have received outputs from multiple AI squads for a single user request. "
            "Weave them into ONE coherent, well-structured response. "
            "Do not repeat yourself. Keep the Telegram Markdown format (*bold*, _italic_, links).\n\n"
            + "\n\n---\n".join(f"Squad output {i+1}:\n{r}" for i, r in enumerate(results))
        )
        try:
            merged = await gateway.complete(
                messages=[{"role": "user", "content": merge_prompt}],
                max_tokens=2048,
                routing=state["request"].routing,
                squad="director-merge",
            )
            return {"final_output": merged["content"]}
        except Exception:
            logger.warning("director_merge_failed — concatenating outputs", exc_info=True)
            return {"final_output": "\n\n---\n\n".join(results)}

    # ── Routing condition ─────────────────────────────────────────────────────

    @staticmethod
    def _should_continue(state: DirectorState) -> str:
        """Route: continue executing squads, merge when done, or end on error."""
        if state.get("error"):
            return END
        if state.get("plan"):          # more squads to execute
            return "continue"
        results = state.get("results", [])
        if len(results) > 1:           # multiple outputs need merging
            return "merge"
        return END                      # single output — skip merge node

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, request: TaskRequest) -> TaskResult:
        """Execute the director graph and return a TaskResult.

        Falls back to direct squad dispatch when LangGraph is unavailable.
        """
        if self._graph is None:
            # LangGraph not installed — route directly
            squad_name = _intent_to_squad(request.intent)
            squad = self._squads.get(squad_name) or next(iter(self._squads.values()), None)
            if squad is None:
                return TaskResult(
                    task_id=request.task_id,
                    squad=SquadName.GENERAL,
                    content="No squads registered.",
                    model_used="none",
                    routing=request.routing,
                )
            return await squad.process(request)

        initial_state: DirectorState = {
            "request": request,
            "plan": [],
            "context_pass": False,
            "results": [],
            "final_output": "",
            "error": "",
        }

        try:
            final_state = await self._graph.ainvoke(initial_state)
        except Exception:
            logger.error("director_graph_failed", exc_info=True)
            # Hard fallback — dispatch directly
            squad_name = _intent_to_squad(request.intent)
            squad = self._squads.get(squad_name) or next(iter(self._squads.values()), None)
            if squad:
                return await squad.process(request)
            return TaskResult(
                task_id=request.task_id,
                squad=SquadName.GENERAL,
                content="Director graph failed and no fallback squad available.",
                model_used="none",
                routing=request.routing,
            )

        content = final_state.get("final_output") or (
            final_state.get("results", [""])[-1]
        )
        squads_used = ",".join(
            s for s in _ROUTABLE_SQUADS
            if any(s in r for r in final_state.get("results", []))
        ) or _intent_to_squad(request.intent)

        return TaskResult(
            task_id=request.task_id,
            squad=SquadName(squads_used.split(",")[0]) if squads_used else SquadName.GENERAL,
            content=content,
            model_used="director",
            routing=request.routing,
            metadata={"squads_used": squads_used},
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _intent_to_squad(intent: Intent) -> str:
    """Map Intent enum to squad name string."""
    return {
        Intent.SPIRITUAL: SquadName.SPIRITUAL,
        Intent.CAREER: SquadName.CAREER,
        Intent.INTELLIGENCE: SquadName.INTELLIGENCE,
        Intent.CREATIVE: SquadName.CREATIVE,
    }.get(intent, SquadName.INTELLIGENCE)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response string."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}
