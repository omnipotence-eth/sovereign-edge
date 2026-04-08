"""arXiv and Hugging Face paper discovery.

Fetches AI/ML research papers from:
  - arXiv Atom feed API (free, no key required)
  - Hugging Face daily papers API (free, no key required)

Usage:
    papers = await get_research_digest(queries=["LLM agents", "inference optimization"])
    print(format_digest(papers))
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx
from observability.logging import get_logger

logger = get_logger(__name__, component="intelligence")

_ARXIV_API = "https://export.arxiv.org/api/query"
_HF_DAILY_API = "https://huggingface.co/api/daily_papers"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_MAX_DIGEST = 15


@dataclass
class Paper:
    """A research paper from arXiv or Hugging Face."""

    title: str
    authors: str
    abstract: str
    url: str
    published: str
    source: str  # "arxiv" | "huggingface"


# ── arXiv ─────────────────────────────────────────────────────────────────────


async def search_arxiv(query: str, max_results: int = 10) -> list[Paper]:
    """Fetch papers from the arXiv Atom feed API.

    Returns an empty list on any network or parse error.
    """
    try:
        params = {"search_query": f"all:{query}", "max_results": max_results}
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(_ARXIV_API, params=params)
            resp.raise_for_status()

        root = ET.fromstring(resp.text)  # noqa: S314 — arXiv API is trusted source
        papers: list[Paper] = []
        for entry in root.findall("atom:entry", _ATOM_NS):
            title_raw = entry.findtext("atom:title", default="", namespaces=_ATOM_NS)
            title = " ".join(title_raw.split())  # normalise whitespace / newlines

            link_el = entry.find("atom:link[@rel='alternate']", _ATOM_NS)
            if link_el is None:
                link_el = entry.find("atom:link", _ATOM_NS)
            url = link_el.get("href", "") if link_el is not None else ""

            abstract = entry.findtext("atom:summary", default="", namespaces=_ATOM_NS).strip()
            published = entry.findtext("atom:published", default="", namespaces=_ATOM_NS)[:10]

            author_names = [
                (a.findtext("atom:name", default="", namespaces=_ATOM_NS) or "").strip()
                for a in entry.findall("atom:author", _ATOM_NS)
            ]
            authors = ", ".join(filter(None, author_names))

            papers.append(
                Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=url,
                    published=published,
                    source="arxiv",
                )
            )

        return papers

    except Exception:
        logger.warning("arxiv.search_failed query=%s", query, exc_info=True)
        return []


# ── Hugging Face daily papers ─────────────────────────────────────────────────


async def get_hf_daily_papers(limit: int = 10) -> list[Paper]:
    """Fetch today's paper highlights from the HF daily papers feed.

    Returns an empty list on any network or parse error.
    """
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(_HF_DAILY_API)
            resp.raise_for_status()

        items = resp.json()
        papers: list[Paper] = []
        for item in items[:limit]:
            p = item.get("paper", {})
            paper_id = p.get("id", "")
            url = f"https://arxiv.org/abs/{paper_id}" if paper_id else ""

            raw_authors = p.get("authors", [])
            if isinstance(raw_authors, list):
                author_names = [
                    (a.get("name", "") if isinstance(a, dict) else str(a)) for a in raw_authors
                ]
            else:
                author_names = []
            authors = ", ".join(filter(None, author_names))

            papers.append(
                Paper(
                    title=p.get("title", ""),
                    authors=authors,
                    abstract=p.get("summary", ""),
                    url=url,
                    published=p.get("publishedAt", ""),
                    source="huggingface",
                )
            )

        return papers

    except Exception:
        logger.warning("hf_daily_papers.fetch_failed", exc_info=True)
        return []


# ── Research digest ───────────────────────────────────────────────────────────


async def get_research_digest(
    queries: list[str] | None = None,
) -> list[Paper]:
    """Aggregate papers from arXiv (per query) and HF daily papers.

    Deduplicates by exact title and caps results at 15.
    """
    if queries is None:
        queries = []

    # Fan-out all arXiv queries + HF in parallel
    tasks = [search_arxiv(q) for q in queries] + [get_hf_daily_papers()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_titles: set[str] = set()
    papers: list[Paper] = []
    for batch in results:
        if isinstance(batch, Exception):
            continue
        for paper in batch:  # type: ignore[union-attr]
            if paper.title not in seen_titles:
                seen_titles.add(paper.title)
                papers.append(paper)
            if len(papers) >= _MAX_DIGEST:
                return papers

    return papers


# ── Formatting ────────────────────────────────────────────────────────────────


def format_digest(papers: list[Paper]) -> str:
    """Format a list of papers as a human-readable digest."""
    if not papers:
        return "No recent papers found."

    lines: list[str] = []
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. {p.title}")
        if p.authors:
            lines.append(f"   Authors: {p.authors}")
        lines.append(f"   URL: {p.url}")
        lines.append("")

    return "\n".join(lines).rstrip()
