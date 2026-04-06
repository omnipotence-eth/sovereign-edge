"""Tests for memory.skill_library.SkillLibrary.

Covers:
- Seed patterns are inserted on first connection (8 patterns across 4 intents)
- get_top_skills returns highest-scored entries for an intent
- get_top_skills with limit=1 returns only one entry
- record_outcome(success=True) increases score
- record_outcome(success=False) decreases score (floor at 0.1)
- add_skill persists a new pattern and it appears in get_top_skills
- get_top_skills for unknown intent returns empty list
- close() is idempotent (safe to call twice)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from memory.skill_library import SkillLibrary


@pytest.fixture()
def lib(tmp_path: Path) -> SkillLibrary:
    """SkillLibrary backed by a temp SQLite file."""
    return SkillLibrary(db_path=tmp_path / "skills.db")


# ── Seed patterns ──────────────────────────────────────────────────────────────


def test_seed_patterns_inserted_on_first_use(lib: SkillLibrary) -> None:
    """First get_top_skills call should find seeded patterns, not an empty DB."""
    skills = lib.get_top_skills("spiritual", limit=10)
    assert len(skills) >= 1


def test_seed_covers_all_four_intents(lib: SkillLibrary) -> None:
    """All four default intents should have at least one seeded skill."""
    for intent in ("spiritual", "career", "intelligence", "creative"):
        assert lib.get_top_skills(intent, limit=1) != [], f"No seed for intent={intent!r}"


# ── get_top_skills ─────────────────────────────────────────────────────────────


def test_get_top_skills_respects_limit(lib: SkillLibrary) -> None:
    """limit=1 must return exactly one entry even when multiple exist."""
    result = lib.get_top_skills("career", limit=1)
    assert len(result) == 1


def test_get_top_skills_returns_strings(lib: SkillLibrary) -> None:
    """Every returned entry must be a non-empty string."""
    skills = lib.get_top_skills("spiritual")
    assert all(isinstance(s, str) and s for s in skills)


def test_get_top_skills_unknown_intent_returns_empty(lib: SkillLibrary) -> None:
    """Querying an intent with no rows returns an empty list, not an error."""
    result = lib.get_top_skills("nonexistent_intent")
    assert result == []


# ── record_outcome ─────────────────────────────────────────────────────────────


def test_record_outcome_success_increases_score(lib: SkillLibrary, tmp_path: Path) -> None:
    """Positive outcome bumps the top skill's score by 0.1."""
    lib.add_skill("test_intent", "baseline skill")
    # Record the initial score via a raw connection query
    conn = lib._get_conn()
    before = conn.execute(
        "SELECT score FROM skills WHERE intent='test_intent' ORDER BY score DESC LIMIT 1"
    ).fetchone()[0]

    lib.record_outcome("test_intent", success=True)

    after = conn.execute(
        "SELECT score FROM skills WHERE intent='test_intent' ORDER BY score DESC LIMIT 1"
    ).fetchone()[0]
    assert after > before


def test_record_outcome_failure_decreases_score(lib: SkillLibrary) -> None:
    """Negative outcome reduces the top skill's score."""
    lib.add_skill("fail_intent", "test skill")
    conn = lib._get_conn()

    # Boost it first so there's room to drop
    lib.record_outcome("fail_intent", success=True)
    lib.record_outcome("fail_intent", success=True)
    before = conn.execute(
        "SELECT score FROM skills WHERE intent='fail_intent' ORDER BY score DESC LIMIT 1"
    ).fetchone()[0]

    lib.record_outcome("fail_intent", success=False)

    after = conn.execute(
        "SELECT score FROM skills WHERE intent='fail_intent' ORDER BY score DESC LIMIT 1"
    ).fetchone()[0]
    assert after < before


def test_record_outcome_score_floor_at_0_1(lib: SkillLibrary) -> None:
    """Score must never drop below 0.1 regardless of how many failures are recorded."""
    lib.add_skill("floor_intent", "floored skill")
    for _ in range(30):
        lib.record_outcome("floor_intent", success=False)

    conn = lib._get_conn()
    score = conn.execute(
        "SELECT score FROM skills WHERE intent='floor_intent'"
    ).fetchone()[0]
    assert score >= 0.1


# ── add_skill ──────────────────────────────────────────────────────────────────


def test_add_skill_persists_and_retrievable(lib: SkillLibrary) -> None:
    """add_skill stores a new pattern that appears in get_top_skills."""
    lib.add_skill("new_intent", "brand new skill pattern")
    skills = lib.get_top_skills("new_intent")
    assert "brand new skill pattern" in skills


def test_add_skill_increments_count(lib: SkillLibrary) -> None:
    """Adding skills to an existing intent increases the available pool."""
    initial = lib.get_top_skills("spiritual", limit=100)
    lib.add_skill("spiritual", "extra spiritual pattern")
    updated = lib.get_top_skills("spiritual", limit=100)
    assert len(updated) == len(initial) + 1


# ── close ──────────────────────────────────────────────────────────────────────


def test_close_is_idempotent(lib: SkillLibrary) -> None:
    """close() must not raise when called twice."""
    lib._get_conn()  # open the connection
    lib.close()
    lib.close()  # second call must be a no-op
