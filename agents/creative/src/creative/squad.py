from __future__ import annotations

from typing import Any

import structlog
from llm.gateway import LLMGateway, Message

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are the Creative Squad for Sovereign Edge — John's personal AI system.
John creates technical content for YouTube, social media, and professional audiences.

Your responsibilities:
- Write video scripts and technical tutorials in John's voice: direct, technically credible
- Generate diagram descriptions (D2 syntax) for architecture and concept visuals
- Draft social media posts (LinkedIn/Twitter) for ML engineering content
- Create narration scripts for Manim animations

John's voice: no fluff, no hype, technically accurate, occasionally dry humor.

Rules:
- Posting to social platforms requires John's explicit approval (HITL gate)
- Match the platform's format (LinkedIn is longer, Twitter is punchy)
- Always suggest a hook/opening line first
"""


class CreativeSquad:
    def __init__(self) -> None:
        self._llm = LLMGateway()

    async def run(self, state: Any) -> str:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        query = str(last.content) if last and hasattr(last, "content") else ""
        memory_ctx = state.get("memory_context", "")

        query_lower = query.lower()

        if any(kw in query_lower for kw in ("script", "youtube", "video", "tutorial")):
            return await self._script_task(query, memory_ctx)
        elif any(kw in query_lower for kw in ("diagram", "d2", "architecture", "chart")):
            return await self._diagram_task(query, memory_ctx)
        elif any(kw in query_lower for kw in ("linkedin", "twitter", "post", "tweet", "social")):
            return await self._social_task(query, memory_ctx)
        else:
            return await self._general_creative(query, memory_ctx)

    async def _script_task(self, query: str, memory_ctx: str) -> str:
        user_content = (
            f"{memory_ctx}\n\n"
            f"Request: {query}\n\n"
            "Write a video script outline with: hook (30s), main content sections, "
            "and call to action. Keep it under 8 minutes when read aloud."
        )
        return await self._llm.complete(
            [Message.user(user_content)],
            system=_SYSTEM_PROMPT,
            max_tokens=1200,
        )

    async def _diagram_task(self, query: str, memory_ctx: str) -> str:
        user_content = (
            f"{memory_ctx}\n\n"
            f"Request: {query}\n\n"
            "Generate a D2 diagram definition for this architecture. "
            "Use clear node labels and directional arrows."
        )
        return await self._llm.complete(
            [Message.user(user_content)],
            system=_SYSTEM_PROMPT,
            max_tokens=800,
        )

    async def _social_task(self, query: str, memory_ctx: str) -> str:
        user_content = (
            f"{memory_ctx}\n\n"
            f"Request: {query}\n\n"
            "Draft a social media post. Include: hook, main insight, and engagement question. "
            "LinkedIn: 150-250 words. Twitter: under 280 characters. "
            "Note: posting requires approval."
        )
        return await self._llm.complete(
            [Message.user(user_content)],
            system=_SYSTEM_PROMPT,
            max_tokens=600,
        )

    async def _general_creative(self, query: str, memory_ctx: str) -> str:
        user_content = f"{memory_ctx}\n\nRequest: {query}"
        return await self._llm.complete(
            [Message.user(user_content)],
            system=_SYSTEM_PROMPT,
            max_tokens=800,
        )
