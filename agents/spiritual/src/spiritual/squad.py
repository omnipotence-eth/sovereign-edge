from __future__ import annotations

from typing import Any

import structlog
from llm.gateway import LLMGateway, Message

from spiritual.bible_rag import BibleRAG

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are the Spiritual Squad for Sovereign Edge — John's personal AI system.
John is a Christian who values biblical truth as authoritative.

Your responsibilities:
- Answer biblical questions with scripture-grounded responses
- Retrieve and cross-reference verses from the STEPBible dataset
- Generate devotionals, prayer prompts, and theological explanations
- Support deep Bible study with historical and linguistic context

Rules:
- NEVER fabricate scripture references — if uncertain, say so clearly
- Always cite chapter:verse when quoting
- Distinguish between direct scripture quotes and your own synthesis
- For theological disputes, present the text faithfully without imposing opinion
- Be pastoral in tone: warm, honest, grounded
"""

_DEVOTIONAL_PROMPT = """Write a short (150-200 word) morning devotional for John.
Include: a key verse, a brief reflection, and a practical application point.
Draw from the retrieved scripture context if provided.
"""


class SpiritualSquad:
    def __init__(self) -> None:
        self._llm = LLMGateway()
        self._rag = BibleRAG()

    async def run(self, state: Any) -> str:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        query = str(last.content) if last and hasattr(last, "content") else ""
        memory_ctx = state.get("memory_context", "")

        query_lower = query.lower()

        if any(kw in query_lower for kw in ("devotional", "morning", "daily bread")):
            return await self._devotional_task(memory_ctx)
        elif any(kw in query_lower for kw in ("look up", "find verse", "what does")):
            return await self._verse_lookup_task(query, memory_ctx)
        else:
            return await self._qa_task(query, memory_ctx)

    async def _qa_task(self, query: str, memory_ctx: str) -> str:
        # Retrieve relevant scripture via RAG
        rag_results = self._rag.search(query, limit=5)
        scripture_ctx = self._format_rag_results(rag_results)

        user_content = (
            f"{memory_ctx}\n\n"
            f"**Retrieved scripture context:**\n{scripture_ctx}\n\n"
            f"Question: {query}"
        )
        return await self._llm.complete(
            [Message.user(user_content)],
            system=_SYSTEM_PROMPT,
            max_tokens=800,
        )

    async def _verse_lookup_task(self, query: str, memory_ctx: str) -> str:
        # Try direct lookup first, fall back to semantic search
        direct = self._rag.lookup_verse(query)
        if direct:
            context = f"Verse: {direct}"
        else:
            results = self._rag.search(query, limit=3)
            context = self._format_rag_results(results)

        user_content = f"{memory_ctx}\n\n{context}\n\nRequest: {query}"
        return await self._llm.complete(
            [Message.user(user_content)],
            system=_SYSTEM_PROMPT,
            max_tokens=600,
        )

    async def _devotional_task(self, memory_ctx: str) -> str:
        # Retrieve a few psalms / proverbs for context
        results = self._rag.search("trust in God morning praise", limit=3)
        scripture_ctx = self._format_rag_results(results)

        user_content = f"{memory_ctx}\n\n{scripture_ctx}"
        return await self._llm.complete(
            [Message.user(user_content)],
            system=_SYSTEM_PROMPT + "\n\n" + _DEVOTIONAL_PROMPT,
            max_tokens=400,
        )

    def _format_rag_results(self, results: list[dict]) -> str:
        if not results:
            return "(No matching scripture found in local index)"
        lines = []
        for r in results:
            ref = r.get("ref", "")
            text = r.get("text", "")
            lines.append(f"[{ref}] {text}")
        return "\n".join(lines)

    async def morning_verse(self) -> str:
        """Scheduled: part of 06:00 morning brief."""
        results = self._rag.search("encouragement strength new day", limit=1)
        if results:
            ref = results[0].get("ref", "")
            text = results[0].get("text", "")
            return f"📖 *{ref}*: {text}"
        return ""
