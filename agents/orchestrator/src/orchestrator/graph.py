from __future__ import annotations

import functools

import structlog
from core.config import get_settings
from core.security import sanitize_input
from core.types import IntentClass
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from memory.skill_library import SkillLibrary
from memory.store import MemoryStore
from observability import traced
from router.classifier import IntentRouter

from orchestrator.state import SovereignState

logger = structlog.get_logger(__name__)

_ERROR_RESPONSE = "I encountered an error processing your request. Please try again."

# Guard against memory_context ballooning the prompt when the user has a long history.
# Keeps the most recent ~1 500 tokens worth of episodic context.
_MAX_MEMORY_CONTEXT_CHARS = 6_000


# ── Lazy singletons — one DB connection per process, not per request ──────────


@functools.lru_cache(maxsize=1)
def _get_memory() -> MemoryStore:
    return MemoryStore(get_settings())


@functools.lru_cache(maxsize=1)
def _get_skill_lib() -> SkillLibrary:
    return SkillLibrary(get_settings().skill_db_path)


# ── Node implementations ──────────────────────────────────────────────────────


async def router_node(state: SovereignState) -> dict:
    """Classify the latest user message; populate intent, memory, and skill context."""
    settings = get_settings()
    router = IntentRouter(settings)

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    raw_text = last_human.content if last_human else ""

    # Sanitize before any memory interaction — prevents injection payloads
    # from being stored and later surfaced in future context windows.
    text = sanitize_input(str(raw_text))

    result = router.classify(text)
    memory_context = _get_memory().format_context(text)

    # Guard: trim memory context to prevent context window overflow.
    if len(memory_context) > _MAX_MEMORY_CONTEXT_CHARS:
        memory_context = memory_context[-_MAX_MEMORY_CONTEXT_CHARS:]

    # 3rd-tier memory: load top-ranked skill patterns for this intent
    top_skills = _get_skill_lib().get_top_skills(result.intent.value, limit=2)
    skill_context = "\n".join(f"- {s}" for s in top_skills) if top_skills else ""

    logger.info(
        "orchestrator.routed",
        intent=result.intent.value,
        confidence=result.confidence,
        skill_patterns=len(top_skills),
    )
    return {
        "intent": result.intent.value,
        "intent_confidence": result.confidence,
        "memory_context": memory_context,
        "skill_context": skill_context,
    }


async def spiritual_node(state: SovereignState) -> dict:
    from spiritual import SpiritualSquad

    try:
        squad = SpiritualSquad()
        result = await squad.run(state)  # type: ignore
    except Exception:
        logger.error("orchestrator.spiritual_node_failed", exc_info=True)
        result = _ERROR_RESPONSE
    return {"squad_result": result, "hitl_required": False}


async def career_node(state: SovereignState) -> dict:
    from career import CareerSquad

    try:
        squad = CareerSquad()
        result = await squad.run(state)  # type: ignore
    except Exception:
        logger.error("orchestrator.career_node_failed", exc_info=True)
        result = _ERROR_RESPONSE
    # Career actions (apply, send email) always require HITL
    return {"squad_result": result, "hitl_required": True}


async def intelligence_node(state: SovereignState) -> dict:
    from intelligence import IntelligenceSquad

    try:
        squad = IntelligenceSquad()
        result = await squad.run(state)  # type: ignore
    except Exception:
        logger.error("orchestrator.intelligence_node_failed", exc_info=True)
        result = _ERROR_RESPONSE
    return {"squad_result": result, "hitl_required": False}


async def creative_node(state: SovereignState) -> dict:
    from creative import CreativeSquad

    try:
        squad = CreativeSquad()
        result = await squad.run(state)  # type: ignore
    except Exception:
        logger.error("orchestrator.creative_node_failed", exc_info=True)
        result = _ERROR_RESPONSE
    # Publishing / posting requires HITL
    return {"squad_result": result, "hitl_required": True}


async def memory_node(state: SovereignState) -> dict:
    """Persist the squad result to Mem0 and reinforce skill patterns."""
    result = state.get("squad_result", "")
    intent = state.get("intent", "")

    if result and result != _ERROR_RESPONSE:
        _get_memory().add_memory(f"[{intent}] {result}")
        # Reinforce skill patterns for non-HITL successful completions
        if not state.get("hitl_required"):
            _get_skill_lib().record_outcome(intent, success=True)

    return {}


async def hitl_node(state: SovereignState) -> dict:
    """Pause for human approval if hitl_required=True."""
    if not state.get("hitl_required"):
        return {"hitl_approved": True}

    logger.info("orchestrator.hitl_interrupt", intent=state.get("intent"))
    # LangGraph interrupt() suspends execution here until graph.resume() is called
    approval = interrupt(
        {
            "action": state.get("intent"),
            "preview": state.get("squad_result", "")[:500],
            "message": "Approve this action? Reply /approve or /reject.",
        }
    )
    return {"hitl_approved": approval}


async def delivery_node(state: SovereignState) -> dict:
    """Format and deliver the final response.

    The actual Telegram send is handled by services/telegram which calls
    graph.resume() — delivery_node just packages the final message.
    """
    approved = state.get("hitl_approved")
    result = state.get("squad_result", "")

    if approved is False:
        final = "Action cancelled."
    else:
        final = result

    logger.info("orchestrator.delivering", length=len(final))
    return {"messages": [AIMessage(content=final)]}


# ── Routing logic ─────────────────────────────────────────────────────────────


def _route_intent(state: SovereignState) -> str:
    intent = state.get("intent", IntentClass.INTELLIGENCE.value)
    routes = {
        IntentClass.SPIRITUAL.value: "spiritual",
        IntentClass.CAREER.value: "career",
        IntentClass.INTELLIGENCE.value: "intelligence",
        IntentClass.CREATIVE.value: "creative",
    }
    return routes.get(intent, "intelligence")


def _route_hitl(state: SovereignState) -> str:
    return "hitl" if state.get("hitl_required") else "delivery"


# ── Graph assembly ────────────────────────────────────────────────────────────


def build_graph() -> StateGraph:  # type: ignore
    g = StateGraph(SovereignState)  # type: ignore

    g.add_node("router", router_node)
    g.add_node("spiritual", spiritual_node)
    g.add_node("career", career_node)
    g.add_node("intelligence", intelligence_node)
    g.add_node("creative", creative_node)
    g.add_node("memory", memory_node)
    g.add_node("hitl", hitl_node)
    g.add_node("delivery", delivery_node)

    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        _route_intent,
        {
            "spiritual": "spiritual",
            "career": "career",
            "intelligence": "intelligence",
            "creative": "creative",
        },
    )

    for squad in ("spiritual", "career", "intelligence", "creative"):
        g.add_edge(squad, "memory")

    g.add_conditional_edges("memory", _route_hitl, {"hitl": "hitl", "delivery": "delivery"})
    g.add_edge("hitl", "delivery")
    g.add_edge("delivery", END)

    return g


_checkpointer = MemorySaver()
_compiled = build_graph().compile(checkpointer=_checkpointer, interrupt_before=["hitl"])


@traced("run_turn")
async def run_turn(
    user_text: str,
    *,
    thread_id: str = "default",
    schedule_trigger: str | None = None,
) -> str:
    """Process one user turn through the full graph.

    Wrapped with @traced so every request gets a correlation ID bound to
    all structlog context vars for that request's duration.
    Returns the assistant's final response text.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    initial: SovereignState = {  # type: ignore
        "messages": [HumanMessage(content=user_text)],
        "intent": "",
        "intent_confidence": 0.0,
        "memory_context": "",
        "skill_context": "",
        "squad_result": "",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": schedule_trigger,
    }
    result = await _compiled.ainvoke(initial, config)  # type: ignore
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        return last.content if hasattr(last, "content") else str(last)
    return ""


async def resume_turn(thread_id: str, *, approved: bool) -> str:
    """Resume a graph suspended at the HITL interrupt.

    Records skill outcome based on approval decision.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    result = await _compiled.ainvoke(  # type: ignore
        {"hitl_approved": approved},
        config,
    )

    # Record HITL outcome to skill library
    state = _compiled.get_state(config)  # type: ignore
    if state and state.values:
        intent = state.values.get("intent", "")
        if intent:
            _get_skill_lib().record_outcome(intent, success=approved)

    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        return last.content if hasattr(last, "content") else str(last)
    return "Action cancelled." if not approved else ""
