"""Tests for career.subgraph — job listing extraction, URL validation, and search queries."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from career.subgraph import (
    _ATS_SITES,
    JobListing,
    JobListingResponse,
    _check_url_live,
    _validate_listings,
    build_search_queries,
    format_job_listings,
)

# ── JobListing.is_dfw_eligible ────────────────────────────────────────────────


def test_is_dfw_eligible_dfw_city() -> None:
    job = JobListing(
        company="Acme", title="ML Engineer", city="Plano", apply_url="https://example.com"
    )
    assert job.is_dfw_eligible is True


def test_is_dfw_eligible_remote() -> None:
    job = JobListing(
        company="Acme",
        title="ML Engineer",
        city="New York",
        work_mode="remote",
        apply_url="https://example.com",
    )
    assert job.is_dfw_eligible is True


def test_is_dfw_eligible_hybrid() -> None:
    job = JobListing(
        company="Acme",
        title="ML Engineer",
        city="Seattle",
        work_mode="hybrid",
        apply_url="https://example.com",
    )
    assert job.is_dfw_eligible is True


def test_is_dfw_eligible_rejects_non_dfw_onsite() -> None:
    job = JobListing(
        company="Acme",
        title="ML Engineer",
        city="San Francisco",
        work_mode="onsite",
        apply_url="https://example.com",
    )
    assert job.is_dfw_eligible is False


def test_is_dfw_eligible_city_normalized_to_title_case() -> None:
    job = JobListing(
        company="Acme", title="ML Engineer", city="DALLAS", apply_url="https://example.com"
    )
    assert job.is_dfw_eligible is True


# ── format_job_listings ────────────────────────────────────────────────────────


def test_format_job_listings_empty_when_no_eligible() -> None:
    response = JobListingResponse(
        jobs=[
            JobListing(
                company="Corp",
                title="ML Engineer",
                city="New York",
                work_mode="onsite",
                apply_url="https://example.com",
            )
        ],
        action_today="Apply to 3 jobs today.",
    )
    result = format_job_listings(response)
    assert "No DFW-eligible" in result
    assert "Apply to 3 jobs today." in result


def test_format_job_listings_renders_eligible_jobs() -> None:
    response = JobListingResponse(
        jobs=[
            JobListing(
                company="Capital One",
                title="ML Engineer",
                city="Plano",
                salary_range="$140k-$170k",
                apply_url="https://boards.greenhouse.io/capitalone/jobs/123",
            )
        ],
        action_today="Update your LinkedIn today.",
    )
    result = format_job_listings(response)
    assert "Capital One" in result
    assert "ML Engineer" in result
    assert "$140k-$170k" in result
    assert "boards.greenhouse.io" in result
    assert "Update your LinkedIn today." in result


def test_format_job_listings_shows_work_mode_when_not_onsite() -> None:
    response = JobListingResponse(
        jobs=[
            JobListing(
                company="AT&T",
                title="AI Engineer",
                city="Dallas",
                work_mode="hybrid",
                apply_url="https://jobs.lever.co/att/abc",
            )
        ],
        action_today="",
    )
    result = format_job_listings(response)
    assert "Hybrid" in result


# ── build_search_queries ──────────────────────────────────────────────────────


def test_build_search_queries_returns_two_queries() -> None:
    queries = build_search_queries()
    assert len(queries) == 2


def test_build_search_queries_first_query_targets_ats() -> None:
    queries = build_search_queries()
    first = queries[0]
    assert "greenhouse.io" in first
    assert "lever.co" in first


def test_build_search_queries_second_query_has_freshness() -> None:
    import datetime

    queries = build_search_queries()
    second = queries[1]
    assert str(datetime.date.today().year) in second


def test_ats_sites_constant_includes_major_platforms() -> None:
    assert "greenhouse.io" in _ATS_SITES
    assert "lever.co" in _ATS_SITES
    assert "workday.com" in _ATS_SITES


# ── _check_url_live ───────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_check_url_live_returns_true_on_200() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.head = AsyncMock(return_value=mock_resp)

    result = await _check_url_live("https://boards.greenhouse.io/company/jobs/123", mock_client)
    assert result is True


@pytest.mark.asyncio()
async def test_check_url_live_returns_false_on_404() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.head = AsyncMock(return_value=mock_resp)

    result = await _check_url_live("https://boards.greenhouse.io/company/jobs/999", mock_client)
    assert result is False


@pytest.mark.asyncio()
async def test_check_url_live_fails_open_on_network_error() -> None:
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.head = AsyncMock(side_effect=httpx.ConnectError("timeout"))

    result = await _check_url_live("https://boards.greenhouse.io/company/jobs/123", mock_client)
    assert result is True  # Fail open: keep listing if unreachable


@pytest.mark.asyncio()
async def test_check_url_live_returns_false_for_empty_url() -> None:
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    result = await _check_url_live("", mock_client)
    assert result is False


# ── _validate_listings ────────────────────────────────────────────────────────


def _make_listing(url: str, city: str = "Dallas") -> JobListing:
    return JobListing(company="Corp", title="ML Engineer", city=city, apply_url=url)


@pytest.mark.asyncio()
async def test_validate_listings_drops_dead_urls() -> None:
    listings = [
        _make_listing("https://greenhouse.io/live"),
        _make_listing("https://greenhouse.io/dead"),
    ]

    async def fake_check(url: str, client: httpx.AsyncClient) -> bool:
        return "live" in url

    with patch("career.subgraph._check_url_live", side_effect=fake_check):
        result = await _validate_listings(listings)

    assert len(result) == 1
    assert result[0].apply_url == "https://greenhouse.io/live"


@pytest.mark.asyncio()
async def test_validate_listings_returns_empty_unchanged() -> None:
    result = await _validate_listings([])
    assert result == []


@pytest.mark.asyncio()
async def test_validate_listings_returns_all_on_client_error() -> None:
    listings = [_make_listing("https://greenhouse.io/jobs/1")]

    with patch("career.subgraph.httpx.AsyncClient", side_effect=RuntimeError("network down")):
        result = await _validate_listings(listings)

    # Fail open — return original listings
    assert result == listings
