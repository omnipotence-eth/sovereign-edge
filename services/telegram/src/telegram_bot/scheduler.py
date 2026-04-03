from __future__ import annotations

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from core.config import Settings
from telegram import Bot

logger = structlog.get_logger(__name__)


def build_scheduler(bot: Bot, settings: Settings, user_id: int) -> AsyncIOScheduler:
    """Wire APScheduler jobs to proactive behaviors defined in SOUL.md."""
    scheduler = AsyncIOScheduler()

    # 06:00 daily — morning brief
    scheduler.add_job(
        _morning_brief,
        "cron",
        hour=settings.morning_brief_hour,
        minute=0,
        kwargs={"bot": bot, "user_id": user_id},
        id="morning_brief",
        replace_existing=True,
    )

    # 09:00 Mon-Fri — job board scan
    scheduler.add_job(
        _job_scan,
        "cron",
        day_of_week="mon-fri",
        hour=settings.job_scan_hour,
        minute=0,
        kwargs={"bot": bot, "user_id": user_id},
        id="job_scan",
        replace_existing=True,
    )

    # 18:00 daily — market/portfolio summary
    scheduler.add_job(
        _market_summary,
        "cron",
        hour=settings.market_summary_hour,
        minute=0,
        kwargs={"bot": bot, "user_id": user_id},
        id="market_summary",
        replace_existing=True,
    )

    # Weekly Sunday 08:00 — research digest
    scheduler.add_job(
        _research_digest,
        "cron",
        day_of_week="sun",
        hour=8,
        minute=0,
        kwargs={"bot": bot, "user_id": user_id},
        id="research_digest",
        replace_existing=True,
    )

    return scheduler


async def _send(bot: Bot, user_id: int, text: str) -> None:
    if not text:
        return
    try:
        await bot.send_message(chat_id=user_id, text=text, parse_mode="Markdown")
    except Exception:
        logger.error("scheduler.send_failed", user_id=user_id, exc_info=True)


async def _morning_brief(bot: Bot, user_id: int) -> None:
    from orchestrator.graph import run_turn

    logger.info("scheduler.morning_brief")
    result = await run_turn(
        "Generate morning brief: Bible verse, market summary, top job leads.",
        thread_id=f"scheduled_{user_id}",
        schedule_trigger="morning_brief",
    )
    await _send(bot, user_id, result)


async def _job_scan(bot: Bot, user_id: int) -> None:
    from career import CareerSquad

    logger.info("scheduler.job_scan")
    result = await CareerSquad().daily_job_scan()
    if result:
        await _send(bot, user_id, f"🧑‍💼 *Job Scan*\n\n{result}")


async def _market_summary(bot: Bot, user_id: int) -> None:
    from intelligence import IntelligenceSquad

    logger.info("scheduler.market_summary")
    result = await IntelligenceSquad().daily_market_summary()
    if result:
        await _send(bot, user_id, f"📈 *Market Alert*\n\n{result}")


async def _research_digest(bot: Bot, user_id: int) -> None:
    from intelligence import IntelligenceSquad

    logger.info("scheduler.research_digest")
    result = await IntelligenceSquad().weekly_digest()
    if result:
        await _send(bot, user_id, f"🔬 *Weekly AI Research Digest*\n\n{result}")
