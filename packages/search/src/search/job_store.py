"""
Job deduplication store — SQLite-backed seen-job tracking.

Prevents the same job from appearing in every morning brief.
The dedup window is configurable via SE_CAREER_DEDUP_WINDOW_DAYS (default: 7 days).

Schema: jobs(job_id, company, title, location, apply_url, source, salary, seen_at, status)
Dedup key: SHA-1(company.lower() + "|" + title.lower()) — stable across sources.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    company     TEXT NOT NULL,
    title       TEXT NOT NULL,
    location    TEXT DEFAULT '',
    apply_url   TEXT DEFAULT '',
    source      TEXT DEFAULT '',
    salary      TEXT DEFAULT '',
    seen_at     TEXT NOT NULL,
    status      TEXT DEFAULT 'new'
);
CREATE INDEX IF NOT EXISTS idx_jobs_seen_at ON jobs(seen_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
"""


def _make_job_id(company: str, title: str) -> str:
    """Stable 16-char hex key from company + title (case-insensitive)."""
    key = f"{company.lower().strip()}|{title.lower().strip()}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]  # noqa: S324 — not crypto


class JobStore:
    """SQLite-backed job deduplication and application tracking.

    Thread-safe for single-process use (sqlite3 serializes via GIL).
    Each public method opens a fresh connection — safe for coroutines that
    call it from asyncio.to_thread().
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript(_SCHEMA)
        logger.info("job_store_initialized path=%s", db_path)

    def filter_new(self, listings: list[Any], dedup_window_days: int = 7) -> list[Any]:
        """Return listings not seen within the dedup window.

        Accepts any list of objects with .company and .title attributes
        (e.g. JobRawListing or JobListing).
        """
        if not listings:
            return listings
        cutoff = (datetime.now(UTC) - timedelta(days=dedup_window_days)).isoformat()
        new: list[Any] = []
        with sqlite3.connect(self._db_path) as conn:
            for listing in listings:
                job_id = _make_job_id(
                    getattr(listing, "company", ""),
                    getattr(listing, "title", ""),
                )
                row = conn.execute(
                    "SELECT 1 FROM jobs WHERE job_id = ? AND seen_at > ?",
                    (job_id, cutoff),
                ).fetchone()
                if row is None:
                    new.append(listing)
        logger.info(
            "job_store_filter total=%d new=%d window_days=%d",
            len(listings),
            len(new),
            dedup_window_days,
        )
        return new

    def mark_seen(self, listings: list[Any]) -> None:
        """Upsert listings as seen (insert new, update seen_at on conflict)."""
        if not listings:
            return
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            for listing in listings:
                job_id = _make_job_id(
                    getattr(listing, "company", ""),
                    getattr(listing, "title", ""),
                )
                conn.execute(
                    """
                    INSERT INTO jobs
                        (job_id, company, title, location, apply_url, source, salary, seen_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        seen_at   = excluded.seen_at,
                        apply_url = excluded.apply_url
                    """,
                    (
                        job_id,
                        getattr(listing, "company", ""),
                        getattr(listing, "title", ""),
                        getattr(listing, "location", ""),
                        getattr(listing, "apply_url", ""),
                        getattr(listing, "source", ""),
                        getattr(listing, "salary", ""),
                        now,
                    ),
                )
        logger.info("job_store_marked_seen count=%d", len(listings))

    def mark_applied(self, company: str, title: str) -> bool:
        """Mark a job as applied. Returns True if the record was found."""
        job_id = _make_job_id(company, title)
        with sqlite3.connect(self._db_path) as conn:
            result = conn.execute("UPDATE jobs SET status = 'applied' WHERE job_id = ?", (job_id,))
        applied = result.rowcount > 0
        if applied:
            logger.info("job_store_marked_applied company=%r title=%r", company, title)
        else:
            logger.warning("job_store_mark_applied_not_found company=%r title=%r", company, title)
        return applied

    def get_stats(self) -> dict[str, int]:
        """Return job tracking stats: total, applied, seen this week."""
        with sqlite3.connect(self._db_path) as conn:
            total: int = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            applied: int = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'applied'"
            ).fetchone()[0]
            week_ago = (datetime.now(UTC) - timedelta(days=7)).isoformat()
            this_week: int = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE seen_at > ?", (week_ago,)
            ).fetchone()[0]
        return {"total": total, "applied": applied, "this_week": this_week}
