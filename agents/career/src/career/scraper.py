from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class JobListing:
    title: str
    company: str
    location: str
    description: str
    url: str
    source: str
    date_posted: str
    salary: str = ""
    remote: bool = False
    tags: list[str] = field(default_factory=list)

    def is_dfw(self, target_cities: list[str]) -> bool:
        loc = self.location.lower()
        return any(city.lower() in loc for city in target_cities) or "remote" in loc


async def scrape_jobs(
    roles: list[str],
    location: str,
    *,
    results_per_role: int = 20,
) -> list[JobListing]:
    """Scrape jobs from LinkedIn, Indeed, Glassdoor via JobSpy."""
    try:
        from jobspy import scrape_jobs as jobspy_scrape  # type: ignore[import]
    except ImportError:
        logger.warning("career.scraper.jobspy_not_installed")
        return []

    all_listings: list[JobListing] = []
    for role in roles:
        try:
            df = jobspy_scrape(
                site_name=["linkedin", "indeed", "glassdoor"],
                search_term=role,
                location=location,
                results_wanted=results_per_role,
                hours_old=48,
                country_indeed="USA",
            )
            for _, row in df.iterrows():
                all_listings.append(
                    JobListing(
                        title=str(row.get("title", "")),
                        company=str(row.get("company", "")),
                        location=str(row.get("location", "")),
                        description=str(row.get("description", ""))[:1000],
                        url=str(row.get("job_url", "")),
                        source=str(row.get("site", "")),
                        date_posted=str(row.get("date_posted", "")),
                        salary=str(row.get("salary_source", "")),
                        remote=bool(row.get("is_remote", False)),
                    )
                )
            logger.info("career.scraper.scraped", role=role, count=len(df))
        except Exception:
            logger.error("career.scraper.failed", role=role, exc_info=True)

    return all_listings


def format_listings(listings: list[JobListing], *, limit: int = 10) -> str:
    if not listings:
        return "No jobs found matching your criteria."
    lines = [f"**{len(listings[:limit])} Job Listings**\n"]
    for i, job in enumerate(listings[:limit], 1):
        lines.append(f"{i}. **{job.title}** @ {job.company}")
        lines.append(f"   📍 {job.location} | {job.source} | {job.date_posted}")
        if job.salary:
            lines.append(f"   💰 {job.salary}")
        lines.append(f"   🔗 {job.url}\n")
    return "\n".join(lines)
