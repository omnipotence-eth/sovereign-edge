from __future__ import annotations

import structlog
from core.config import get_settings
from core.security import sanitize_input
from core.types import SquadState
from llm.gateway import LLMGateway, Message

from career.scraper import format_listings, scrape_jobs

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are the Career Squad for Sovereign Edge — John's personal AI assistant.
John is a senior ML/AI Engineer based in Dallas-Fort Worth, TX.

Your responsibilities:
- Surface the most relevant DFW job openings for ML Engineer / AI Engineer / LLM Engineer roles
- Tailor resume bullets to specific job descriptions (honest, no fabrication)
- Draft cover letters that match John's voice: direct, technically credible, no fluff
- Prep interview Q&A grounded in John's actual experience

John's key differentiators: LangGraph multi-agent systems, GRPO/fine-tuning (DeepSeek-R1),
vLLM/TensorRT-LLM production serving, OTEL observability, Blackwell GPU (RTX 5070 Ti fp8).

Rules:
- ALWAYS tailor to the specific JD — never generic content
- DFW preferred; remote acceptable; no relocation
- Flag if a role requires clearance or visa sponsorship
- Application submission requires John's explicit approval (HITL gate)
"""

_TAILOR_PROMPT = """Given the job description below, rewrite the provided resume bullets to match
the JD's language and requirements. Only use skills John actually has.
Output as markdown bullet list.

JD:
{jd}

Current bullets (improve but don't fabricate):
{bullets}
"""


class CareerSquad:
    def __init__(self) -> None:
        self._llm = LLMGateway()
        self._settings = get_settings()

    async def run(self, state: SquadState) -> str:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        raw_query = str(last.content) if last and hasattr(last, "content") else ""
        query = sanitize_input(raw_query)
        memory_ctx = state.get("memory_context", "")
        skill_ctx = state.get("skill_context", "")

        query_lower = query.lower()

        if any(kw in query_lower for kw in ("resume", "tailor", "cv", "bullets")):
            return await self._tailor_task(query, memory_ctx, skill_ctx)
        elif any(kw in query_lower for kw in ("cover letter", "application")):
            return await self._cover_letter_task(query, memory_ctx, skill_ctx)
        elif any(kw in query_lower for kw in ("interview", "prep", "question")):
            return await self._interview_prep_task(query, memory_ctx, skill_ctx)
        else:
            return await self._job_search_task(query, memory_ctx, skill_ctx)

    def _build_system(self, skill_ctx: str) -> str:
        if not skill_ctx:
            return _SYSTEM_PROMPT
        return f"{_SYSTEM_PROMPT}\n**Proven approaches:**\n{skill_ctx}"

    async def _job_search_task(self, query: str, memory_ctx: str, skill_ctx: str) -> str:
        listings = await scrape_jobs(
            self._settings.job_target_roles,
            self._settings.job_target_location,
        )
        # Filter DFW
        dfw = [j for j in listings if j.is_dfw(self._settings.job_target_cities)]
        listing_text = format_listings(dfw or listings)

        user_content = (
            f"{memory_ctx}\n\n{listing_text}\n\n"
            f"User asked: {query}\n\n"
            "Highlight the top 3 most relevant roles and why they fit John's background."
        )
        return await self._llm.complete(
            [Message.user(user_content)],
            system=self._build_system(skill_ctx),
            max_tokens=800,
        )

    async def _tailor_task(self, query: str, memory_ctx: str, skill_ctx: str) -> str:
        user_content = (
            f"{memory_ctx}\n\n"
            f"Request: {query}\n\n"
            "Help tailor resume bullets to this role. Ask for the JD if not provided."
        )
        return await self._llm.complete(
            [Message.user(user_content)],
            system=self._build_system(skill_ctx),
            max_tokens=1000,
        )

    async def _cover_letter_task(self, query: str, memory_ctx: str, skill_ctx: str) -> str:
        user_content = (
            f"{memory_ctx}\n\n"
            f"Request: {query}\n\n"
            "Draft a cover letter in John's voice. Keep it under 300 words. "
            "Ask for the specific JD and company if not provided."
        )
        return await self._llm.complete(
            [Message.user(user_content)],
            system=self._build_system(skill_ctx),
            max_tokens=600,
        )

    async def _interview_prep_task(self, query: str, memory_ctx: str, skill_ctx: str) -> str:
        user_content = (
            f"{memory_ctx}\n\n"
            f"Request: {query}\n\n"
            "Generate 5 likely interview questions with strong answer frameworks "
            "tailored to John's background."
        )
        return await self._llm.complete(
            [Message.user(user_content)],
            system=self._build_system(skill_ctx),
            max_tokens=1000,
        )

    async def daily_job_scan(self) -> str:
        """Scheduled: 09:00 Mon-Fri — called by APScheduler."""
        listings = await scrape_jobs(
            self._settings.job_target_roles,
            self._settings.job_target_location,
            results_per_role=15,
        )
        dfw = [j for j in listings if j.is_dfw(self._settings.job_target_cities)]
        if not dfw:
            return ""
        listing_text = format_listings(dfw, limit=5)
        return await self._llm.complete(
            [
                Message.user(
                    f"New DFW job postings from the last 48 hours:\n{listing_text}\n\n"
                    "Give a 2-sentence briefing on the best opportunity."
                )
            ],
            system=_SYSTEM_PROMPT,
            max_tokens=200,
        )
