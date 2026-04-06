"""Security and validation tests for goals.store and goals.subgraph — audit additions.

Covers:
- _validate_date(): all valid and invalid input variants
- add_goal(): empty/whitespace title rejection
- _UPDATE_RE regex: % / percent / pct all parse correctly
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from goals.store import GoalStore


@pytest.fixture()
def store(tmp_path: Path) -> GoalStore:
    return GoalStore(db_path=tmp_path / "validation_test.db")


# ── _validate_date ─────────────────────────────────────────────────────────────


def test_validate_date_none_returns_none() -> None:
    assert GoalStore._validate_date(None) is None


def test_validate_date_empty_string_returns_none() -> None:
    assert GoalStore._validate_date("") is None


def test_validate_date_whitespace_only_returns_none() -> None:
    assert GoalStore._validate_date("   ") is None


def test_validate_date_valid_iso_date_unchanged() -> None:
    assert GoalStore._validate_date("2026-06-01") == "2026-06-01"


def test_validate_date_full_datetime_normalises_to_date() -> None:
    """Full ISO 8601 datetime strings must be normalised to YYYY-MM-DD only."""
    result = GoalStore._validate_date("2026-12-25T10:30:00")
    assert result == "2026-12-25"


def test_validate_date_datetime_with_offset_normalises() -> None:
    result = GoalStore._validate_date("2026-06-01T00:00:00+00:00")
    assert result == "2026-06-01"


def test_validate_date_slash_format_raises() -> None:
    with pytest.raises(ValueError, match="target_date must be ISO format"):
        GoalStore._validate_date("01/06/2026")


def test_validate_date_us_format_raises() -> None:
    with pytest.raises(ValueError, match="target_date must be ISO format"):
        GoalStore._validate_date("December 25 2026")


def test_validate_date_garbage_raises() -> None:
    with pytest.raises(ValueError):
        GoalStore._validate_date("not-a-date-at-all")


def test_validate_date_invalid_month_raises() -> None:
    """Month 13 is not valid ISO."""
    with pytest.raises(ValueError):
        GoalStore._validate_date("2026-13-01")


# ── add_goal input validation ──────────────────────────────────────────────────


def test_add_goal_empty_title_raises(store: GoalStore) -> None:
    with pytest.raises(ValueError, match="goal title must not be empty"):
        store.add_goal("")


def test_add_goal_whitespace_title_raises(store: GoalStore) -> None:
    with pytest.raises(ValueError, match="goal title must not be empty"):
        store.add_goal("   ")


def test_add_goal_invalid_date_raises(store: GoalStore) -> None:
    with pytest.raises(ValueError, match="target_date must be ISO format"):
        store.add_goal("Valid title", target_date="25/12/2026")


def test_add_goal_valid_datetime_target_stored_as_date(store: GoalStore) -> None:
    """A full datetime target_date must be stored normalised to YYYY-MM-DD."""
    gid = store.add_goal("Normalise datetime goal", target_date="2026-12-25T08:00:00")
    goal = store.get_by_id(gid)
    assert goal is not None
    assert goal.target_date == "2026-12-25"


def test_add_goal_no_date_ok(store: GoalStore) -> None:
    gid = store.add_goal("No-date goal")
    goal = store.get_by_id(gid)
    assert goal is not None
    assert goal.target_date is None


def test_add_goal_strips_title_whitespace(store: GoalStore) -> None:
    gid = store.add_goal("  Padded Title  ")
    goal = store.get_by_id(gid)
    assert goal is not None
    assert goal.title == "Padded Title"


# ── _UPDATE_RE regex — percent / pct variants ──────────────────────────────────


def _load_update_re() -> re.Pattern[str]:
    """Import the regex directly from the subgraph module."""
    from goals.subgraph import _UPDATE_RE

    return _UPDATE_RE


def test_update_re_matches_percent_symbol() -> None:
    m = _load_update_re().search("update goal #1 to 50%")
    assert m is not None
    assert m.group(2) == "50"


def test_update_re_matches_percent_word() -> None:
    m = _load_update_re().search("update goal #2 to 75 percent")
    assert m is not None
    assert m.group(2) == "75"


def test_update_re_matches_pct_abbreviation() -> None:
    m = _load_update_re().search("set goal #3 to 30 pct")
    assert m is not None
    assert m.group(2) == "30"


def test_update_re_extracts_goal_id() -> None:
    m = _load_update_re().search("mark goal #42 to 100%")
    assert m is not None
    assert m.group(1) == "42"


def test_update_re_no_match_without_number() -> None:
    m = _load_update_re().search("update the goal completely")
    assert m is None
