"""
Bible verse retrieval via bible-api.com — completely free, no API key.

Supports KJV, WEB, YLT, DARBY, ASV, BBE and more.
API docs: https://bible-api.com/

Production upgrades:
  - Module-level persistent AsyncClient (connection pooling)
  - 2-retry exponential backoff for transient failures
  - aclose() for clean shutdown
"""

from __future__ import annotations

import asyncio
import logging
import re

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://bible-api.com"
_TIMEOUT = 10.0
_TRANSLATION = "kjv"
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 1.0

# Matches common scripture references: "John 3:16", "Psalm 23", "1 Corinthians 13:4-7"
_REF_PATTERN = re.compile(
    r"\b(?:\d\s+)?(?:genesis|exodus|leviticus|numbers|deuteronomy|joshua|judges|ruth"
    r"|(?:1|2|3)\s*samuel|(?:1|2)\s*kings|(?:1|2)\s*chronicles|ezra|nehemiah|esther"
    r"|job|psalms?|proverbs|ecclesiastes|song\s+of\s+solomon|isaiah|jeremiah"
    r"|lamentations|ezekiel|daniel|hosea|joel|amos|obadiah|jonah|micah|nahum"
    r"|habakkuk|zephaniah|haggai|zechariah|malachi|matthew|mark|luke|john|acts"
    r"|romans|(?:1|2)\s*corinthians|galatians|ephesians|philippians|colossians"
    r"|(?:1|2)\s*thessalonians|(?:1|2)\s*timothy|titus|philemon|hebrews|james"
    r"|(?:1|2)\s*peter|(?:1|2|3)\s*john|jude|revelation)"
    r"(?:\s+\d+(?::\d+(?:-\d+)?)?)?",
    re.IGNORECASE,
)


# ── Persistent client ──────────────────────────────────────────────────────────
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=3, max_connections=5),
        )
    return _client


# ── Scripture reference extraction ────────────────────────────────────────────


def extract_reference(text: str) -> str | None:
    """Try to extract a scripture reference from free text."""
    match = _REF_PATTERN.search(text)
    return match.group(0).strip() if match else None


# ── Public API ─────────────────────────────────────────────────────────────────


async def random_verse() -> dict[str, str]:
    """Fetch a random Bible verse."""
    client = _get_client()
    url = f"{_BASE}/?random=verse&translation={_TRANSLATION}"

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            return {
                "reference": data.get("reference", ""),
                "text": (data.get("text") or "").strip(),
                "translation": _TRANSLATION.upper(),
            }
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("bible_random_verse_failed attempt=%d error=%s", attempt + 1, exc)
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BASE_DELAY * (2**attempt))

    return {"reference": "", "text": "", "translation": ""}


async def lookup(reference: str) -> dict[str, str]:
    """Look up a verse or passage (e.g. 'John 3:16', 'Psalm 23', '1 Cor 13:4-7')."""
    clean = reference.strip().replace(" ", "+")
    client = _get_client()
    url = f"{_BASE}/{clean}?translation={_TRANSLATION}"

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            verses = data.get("verses", [])
            text = " ".join((v.get("text") or "").strip() for v in verses).strip()
            return {
                "reference": data.get("reference", reference),
                "text": text,
                "translation": _TRANSLATION.upper(),
            }
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning(
                "bible_lookup_failed reference=%r attempt=%d error=%s",
                reference,
                attempt + 1,
                exc,
            )
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BASE_DELAY * (2**attempt))

    return {"reference": reference, "text": "", "translation": ""}


def format_verse(verse: dict[str, str]) -> str:
    """Format a verse dict as a readable string."""
    if not verse["text"]:
        return ""
    return f'*"{verse["text"]}"* — {verse["reference"]} ({verse["translation"]})'


async def aclose() -> None:
    """Close the persistent HTTP client. Call during application shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
