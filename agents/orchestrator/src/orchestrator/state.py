from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SovereignState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str                    # IntentClass.value
    intent_confidence: float
    memory_context: str            # Mem0 retrieved context injected into squad
    squad_result: str              # raw output from the active squad node
    hitl_required: bool            # True = pause for Telegram approval
    hitl_approved: bool | None     # None=pending, True=approved, False=rejected
    schedule_trigger: str | None   # APScheduler job ID if proactive
