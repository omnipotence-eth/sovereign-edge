from __future__ import annotations

import structlog
from core.config import get_settings
from core.types import SquadState
from llm.gateway import LLMGateway, Message

from intelligence.arxiv import format_digest, get_hf_daily_papers, get_research_digest
from intelligence.market import format_market_summary, get_quotes, get_watchlist_alerts

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are the Intelligence Squad for Sovereign Edge — John's personal AI system.
Your role: deliver clear, accurate market and research intelligence.
Rules:
- Always label speculation vs. confirmed fact
- Include source timestamps on market data
- Keep responses concise — John is busy
- For papers: lead with the practical implication, not just the title
"""


class IntelligenceSquad:
    def __init__(self) -> None:
        self._llm = LLMGateway()
        self._settings = get_settings()

    async def run(self, state: SquadState) -> str:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        query = last.content if last and hasattr(last, "content") else ""
        memory_ctx = state.get("memory_context", "")

        query_lower = str(query).lower()

        # Route to appropriate sub-task
        if any(kw in query_lower for kw in ("paper", "arxiv", "research", "digest", "hf")):
            return await self._research_task(str(query), memory_ctx)
        elif any(kw in query_lower for kw in ("market", "stock", "price", "watchlist", "alert")):
            return await self._market_task(str(query), memory_ctx)
        elif any(kw in query_lower for kw in ("morning", "brief", "summary", "today")):
            return await self._morning_brief(memory_ctx)
        else:
            return await self._general_task(str(query), memory_ctx)

    async def _market_task(self, query: str, memory_ctx: str) -> str:
        quotes = await get_quotes(self._settings.watchlist)
        market_text = format_market_summary(quotes)

        user_content = f"{memory_ctx}\n\n{market_text}\n\nUser asked: {query}"
        response = await self._llm.complete(
            [Message.user(user_content)],
            system=_SYSTEM_PROMPT,
            max_tokens=512,
        )
        return response

    async def _research_task(self, query: str, memory_ctx: str) -> str:
        # Extract search term from query or use defaults
        papers = await get_research_digest()
        digest_text = format_digest(papers[:8])

        user_content = f"{memory_ctx}\n\n{digest_text}\n\nUser asked: {query}"
        response = await self._llm.complete(
            [Message.user(user_content)],
            system=_SYSTEM_PROMPT,
            max_tokens=1024,
        )
        return response

    async def _morning_brief(self, memory_ctx: str) -> str:
        quotes = await get_quotes(self._settings.watchlist)
        market_text = format_market_summary(quotes)
        papers = await get_hf_daily_papers(limit=3)
        paper_lines = [f"• {p.title}" for p in papers]
        paper_text = "\n".join(paper_lines) if paper_lines else "No papers today."

        user_content = (
            f"{memory_ctx}\n\n"
            f"**Markets:**\n{market_text}\n\n"
            f"**Top AI Papers Today:**\n{paper_text}\n\n"
            "Generate a concise morning intelligence brief."
        )
        return await self._llm.complete(
            [Message.user(user_content)],
            system=_SYSTEM_PROMPT,
            max_tokens=600,
        )

    async def _general_task(self, query: str, memory_ctx: str) -> str:
        alerts = await get_watchlist_alerts()
        alert_text = format_market_summary(alerts) if alerts else "No watchlist alerts."

        user_content = f"{memory_ctx}\n\n{alert_text}\n\nUser asked: {query}"
        return await self._llm.complete(
            [Message.user(user_content)],
            system=_SYSTEM_PROMPT,
            max_tokens=512,
        )

    async def daily_market_summary(self) -> str:
        """Scheduled: 18:00 daily — called by APScheduler."""
        alerts = await get_watchlist_alerts()
        if not alerts:
            return ""  # Nothing to report
        market_text = format_market_summary(alerts)
        return await self._llm.complete(
            [Message.user(f"Watchlist alert:\n{market_text}\n\nGenerate a brief alert message.")],
            system=_SYSTEM_PROMPT,
            max_tokens=300,
        )

    async def weekly_digest(self) -> str:
        """Scheduled: weekly — called by APScheduler."""
        papers = await get_research_digest()
        digest_text = format_digest(papers)
        prompt = f"Generate a weekly AI research digest from these papers:\n{digest_text}"
        return await self._llm.complete(
            [Message.user(prompt)],
            system=_SYSTEM_PROMPT,
            max_tokens=1200,
        )
