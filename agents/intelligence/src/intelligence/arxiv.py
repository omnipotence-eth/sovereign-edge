from __future__ import annotations

from dataclasses import dataclass

import feedparser
import httpx
import structlog

logger = structlog.get_logger(__name__)

_ARXIV_SEARCH = "https://export.arxiv.org/api/query"
_HF_PAPERS = "https://huggingface.co/api/daily_papers"

_DEFAULT_QUERIES = [
    "large language models",
    "reasoning models",
    "agentic AI",
    "retrieval augmented generation",
    "fine-tuning",
]


@dataclass
class Paper:
    title: str
    authors: str
    abstract: str
    url: str
    published: str
    source: str  # "arxiv" | "huggingface"


async def search_arxiv(query: str, *, max_results: int = 5) -> list[Paper]:
    """Search arXiv for recent papers matching query."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                _ARXIV_SEARCH,
                params={
                    "search_query": f"all:{query}",
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "max_results": max_results,
                },
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            papers = []
            for entry in feed.entries:
                authors = ", ".join(a.get("name", "") for a in entry.get("authors", []))
                papers.append(
                    Paper(
                        title=entry.get("title", "").replace("\n", " "),
                        authors=authors[:100],
                        abstract=entry.get("summary", "")[:300],
                        url=entry.get("link", ""),
                        published=entry.get("published", ""),
                        source="arxiv",
                    )
                )
            return papers
    except Exception:
        logger.error("intelligence.arxiv.search_failed", query=query, exc_info=True)
        return []


async def get_hf_daily_papers(*, limit: int = 5) -> list[Paper]:
    """Fetch today's trending papers from Hugging Face."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_HF_PAPERS)
            resp.raise_for_status()
            items = resp.json()
            papers = []
            for item in items[:limit]:
                p = item.get("paper", {})
                papers.append(
                    Paper(
                        title=p.get("title", ""),
                        authors=", ".join(
                            a.get("name", "") for a in p.get("authors", [])[:3]
                        ),
                        abstract=p.get("summary", "")[:300],
                        url=f"https://arxiv.org/abs/{p.get('id', '')}",
                        published=p.get("publishedAt", ""),
                        source="huggingface",
                    )
                )
            return papers
    except Exception:
        logger.error("intelligence.hf_papers.failed", exc_info=True)
        return []


async def get_research_digest(queries: list[str] | None = None) -> list[Paper]:
    """Aggregate papers from arXiv + HF Papers for weekly digest."""
    all_papers: list[Paper] = []
    for q in (queries or _DEFAULT_QUERIES):
        all_papers.extend(await search_arxiv(q, max_results=3))
    all_papers.extend(await get_hf_daily_papers(limit=5))
    # Deduplicate by title
    seen: set[str] = set()
    unique = []
    for p in all_papers:
        key = p.title.lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique[:15]


def format_digest(papers: list[Paper]) -> str:
    if not papers:
        return "No recent papers found."
    lines = ["**AI Research Digest**\n"]
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. **{p.title}**")
        lines.append(f"   {p.authors} ({p.source})")
        lines.append(f"   {p.abstract[:200]}...")
        lines.append(f"   {p.url}\n")
    return "\n".join(lines)
