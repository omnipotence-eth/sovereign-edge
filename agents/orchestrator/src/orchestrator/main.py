"""
Sovereign Edge orchestrator — task routing and daily schedule.

Squads register at startup; the APScheduler cron fires the morning
pipeline at CT times.  All blocking squad calls are run in the
default ThreadPoolExecutor so the scheduler event loop stays clear.
"""
from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from core.config import get_settings
from core.squad import BaseSquad
from core.types import Intent, RoutingDecision, TaskRequest, TaskResult
from observability.logging import get_logger, setup_logging

logger = get_logger(__name__, component="orchestrator")


class Orchestrator:
    """Central dispatcher and morning-pipeline scheduler."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._squads: dict[str, BaseSquad] = {}
        self._scheduler = AsyncIOScheduler(timezone=self._settings.timezone)
        self._running = False

    # ------------------------------------------------------------------ #
    # Squad registry                                                       #
    # ------------------------------------------------------------------ #

    def register(self, squad: BaseSquad) -> None:
        """Add a squad to the routing table."""
        self._squads[squad.name] = squad
        logger.info("squad_registered", squad=squad.name)

    # ------------------------------------------------------------------ #
    # Task dispatch                                                        #
    # ------------------------------------------------------------------ #

    async def dispatch(self, request: TaskRequest) -> TaskResult:
        """Route *request* to the appropriate squad."""
        squad_name = self._resolve_squad(request)
        squad = self._squads.get(squad_name)

        if squad is None:
            logging.getLogger(__name__).warning(
                "No squad registered for %s — falling back to general", squad_name
            )
            squad = next(iter(self._squads.values()), None)

        if squad is None:
            from core.types import SquadName
            return TaskResult(
                task_id=request.task_id,
                squad=SquadName.GENERAL,
                content="No squads are registered.",
                model_used="none",
                routing=RoutingDecision.LOCAL,
            )

        return await squad.process(request)

    def _resolve_squad(self, request: TaskRequest) -> str:
        intent_map = {
            Intent.SPIRITUAL: "spiritual",
            Intent.CAREER: "career",
            Intent.INTELLIGENCE: "intelligence",
            Intent.CREATIVE: "creative",
            Intent.GENERAL: "general",
        }
        return intent_map.get(request.intent, "general")

    # ------------------------------------------------------------------ #
    # Health                                                               #
    # ------------------------------------------------------------------ #

    async def health_check_all(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for name, squad in self._squads.items():
            try:
                results[name] = await squad.health_check()
            except Exception:
                logger.error("health_check_failed", squad=name, exc_info=True)
                results[name] = False
        return results

    # ------------------------------------------------------------------ #
    # Morning pipeline jobs                                                #
    # ------------------------------------------------------------------ #

    async def _prefetch_intelligence(self) -> None:
        squad = self._squads.get("intelligence")
        if squad:
            logger.info("pipeline_step", step="intelligence_prefetch")
            try:
                await squad.morning_brief()
            except Exception:
                logger.error("pipeline_error", step="intelligence_prefetch", exc_info=True)

    async def _run_morning_planning(self) -> None:
        logger.info("pipeline_step", step="morning_planning")
        health = await self.health_check_all()
        unhealthy = [k for k, v in health.items() if not v]
        if unhealthy:
            logger.warning("unhealthy_squads", squads=unhealthy)

    async def _spiritual_devotional(self) -> None:
        squad = self._squads.get("spiritual")
        if squad:
            logger.info("pipeline_step", step="spiritual_devotional")
            try:
                await squad.morning_brief()
            except Exception:
                logger.error("pipeline_error", step="spiritual_devotional", exc_info=True)

    async def _send_morning_digest(self) -> None:
        """Collect morning briefs from all squads and dispatch to Telegram."""
        logger.info("pipeline_step", step="morning_digest")
        briefs: dict[str, str] = {}
        for name, squad in self._squads.items():
            try:
                briefs[name] = await squad.morning_brief()
            except Exception:
                logger.error("brief_failed", squad=name, exc_info=True)
                briefs[name] = f"[{name} brief unavailable]"

        # Telegram service picks up briefs via a shared queue / direct call
        # when it is registered.  For now we log the assembled digest.
        digest = "\n\n".join(f"**{k.title()}**\n{v}" for k, v in briefs.items())
        logger.info("morning_digest_ready", char_count=len(digest))

    async def _career_scan(self) -> None:
        squad = self._squads.get("career")
        if squad:
            logger.info("pipeline_step", step="career_scan_morning")
            try:
                await squad.morning_brief()
            except Exception:
                logger.error("pipeline_error", step="career_scan_morning", exc_info=True)

    async def _creative_brief(self) -> None:
        squad = self._squads.get("creative")
        if squad:
            logger.info("pipeline_step", step="creative_brief")
            try:
                await squad.morning_brief()
            except Exception:
                logger.error("pipeline_error", step="creative_brief", exc_info=True)

    async def _career_rescan(self) -> None:
        squad = self._squads.get("career")
        if squad:
            logger.info("pipeline_step", step="career_rescan_evening")
            try:
                await squad.morning_brief()
            except Exception:
                logger.error("pipeline_error", step="career_rescan_evening", exc_info=True)

    # ------------------------------------------------------------------ #
    # Scheduler setup                                                      #
    # ------------------------------------------------------------------ #

    def _register_jobs(self) -> None:
        tz = self._settings.timezone
        jobs: list[tuple[Any, str, str]] = [
            (self._prefetch_intelligence,   "04", "30"),
            (self._run_morning_planning,    "05", "00"),
            (self._spiritual_devotional,    "05", "15"),
            (self._send_morning_digest,     "05", "30"),
            (self._career_scan,             "06", "00"),
            (self._creative_brief,          "07", "00"),
            (self._career_rescan,           "18", "00"),
        ]
        for fn, hour, minute in jobs:
            self._scheduler.add_job(
                fn,
                trigger=CronTrigger(hour=int(hour), minute=int(minute), timezone=tz),
                id=fn.__name__,
                replace_existing=True,
                misfire_grace_time=300,  # 5 min tolerance
            )
        logger.info("scheduler_jobs_registered", count=len(jobs))

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        self._register_jobs()
        self._scheduler.start()
        self._running = True
        logger.info("orchestrator_started")

    async def stop(self) -> None:
        self._scheduler.shutdown(wait=False)
        self._running = False
        logger.info("orchestrator_stopped")


# ---------------------------------------------------------------------- #
# Entry point                                                             #
# ---------------------------------------------------------------------- #

async def run() -> None:
    setup_logging(debug=get_settings().debug_mode)
    orchestrator = Orchestrator()

    # Graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_running_loop()

    def _handle_signal() -> None:
        logger.info("shutdown_signal_received")
        asyncio.ensure_future(orchestrator.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    await orchestrator.start()

    # Keep alive until stop() is called
    while orchestrator._running:
        await asyncio.sleep(1)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
