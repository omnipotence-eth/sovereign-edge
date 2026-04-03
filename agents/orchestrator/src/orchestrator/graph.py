from __future__ import annotations

import structlog
from core.config import get_settings
from core.types import IntentClass
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from memory.store import MemoryStore
from router.classifier import IntentRouter

from orchestrator.state import SovereignState

logger = structlog.get_logger(__name__)

# ── Node implementations ──────────────────────────────────────────────────────


async def router_node(state: SovereignState) -> dict:
    """Classify the latest user message and populate intent + memory context."""
    settings = get_settings()
    router = IntentRouter(settings)
    memory = MemoryStore(settings)

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    text = last_human.content if last_human else ""

    result = router.classify(str(text))
    memory_context = memory.format_context(str(text))

    logger.info(
        "orchestrator.routed",
        intent=result.intent.value,
        confidence=result.confidence,
    )
    return {
        "intent": result.intent.value,
        "intent_confidence": result.confidence,
        "memory_context": memory_context,
    }


async def spiritual_node(state: SovereignState) -> dict:
    from agents.spiritual import SpiritualSquad  # type: ignore[import]

    squad = SpiritualSquad()
    result = await squad.run(state)
    return {"squad_result": result, "hitl_required": False}


async def career_node(state: SovereignState) -> dict:
    from agents.career import CareerSquad  # type: ignore[import]

    squad = CareerSquad()
    result = await squad.run(state)
    # Career actions (apply, send email) always require HITL
    return {"squad_result": result, "hitl_required": True}


async def intelligence_node(state: SovereignState) -> dict:
    from agents.intelligence import IntelligenceSquad  # type: ignore[import]

    squad = IntelligenceSquad()
    result = await squad.run(state)
    return {"squad_result": result, "hitl_required": False}


async def creative_node(state: SovereignState) -> dict:
    from agents.creative import CreativeSquad  # type: ignore[import]

    squad = CreativeSquad()
    result = await squad.run(state)
    # Publishing / posting requires HITL
    return {"squad_result": result, "hitl_required": True}


async def memory_node(state: SovereignState) -> dict:
    """Persist the squad result to Mem0 for future recall."""
    memory = MemoryStore()
    result = state.get("squad_result", "")
    intent = state.get("intent", "")
    if result:
        memory.add_memory(f"[{intent}] {result}")
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


def build_graph() -> StateGraph:
    g = StateGraph(SovereignState)

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


async def run_turn(
    user_text: str,
    *,
    thread_id: str = "default",
    schedule_trigger: str | None = None,
) -> str:
    """Process one user turn through the full graph.

    Returns the assistant's final response text.
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial: SovereignState = {  # type: ignore[typeddict-item]
        "messages": [HumanMessage(content=user_text)],
        "intent": "",
        "intent_confidence": 0.0,
        "memory_context": "",
        "squad_result": "",
        "hitl_required": False,
        "hitl_approved": None,
        "schedule_trigger": schedule_trigger,
    }
    result = await _compiled.ainvoke(initial, config)
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        return last.content if hasattr(last, "content") else str(last)
    return ""


async def resume_turn(thread_id: str, *, approved: bool) -> str:
    """Resume a graph suspended at the HITL interrupt."""
    config = {"configurable": {"thread_id": thread_id}}
    result = await _compiled.ainvoke(
        {"hitl_approved": approved},
        config,
    )
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        return last.content if hasattr(last, "content") else str(last)
    return "Action cancelled." if not approved else ""
