"""Tests for health.auth and health.dashboard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from health.server import app

_TOKEN = "test-secret-token"

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _patch_settings() -> None:  # type: ignore[return]
    """Inject a known dashboard token for all tests."""
    from unittest.mock import patch

    from pydantic import SecretStr

    mock_settings = MagicMock()
    mock_settings.dashboard_token = SecretStr(_TOKEN)
    mock_settings.lancedb_path = "/nonexistent"
    mock_settings.logs_path = "/nonexistent"
    mock_settings.ssd_root = "/nonexistent"

    with (
        patch("health.auth.get_settings", return_value=mock_settings),
        patch("health.server.get_settings", return_value=mock_settings),
    ):
        yield


# ── Auth tests ─────────────────────────────────────────────────────────────────


def test_requires_auth(client: TestClient) -> None:
    resp = client.get("/api/v1/stats")
    assert resp.status_code == 401


def test_wrong_token_rejected(client: TestClient) -> None:
    resp = client.get("/api/v1/stats", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_valid_token_200(client: TestClient) -> None:
    with patch("observability.traces.TraceStore") as mock_cls:
        mock_cls.return_value.get_daily_stats.return_value = {"total_requests": 5}
        resp = client.get("/api/v1/stats", headers={"Authorization": f"Bearer {_TOKEN}"})
    assert resp.status_code == 200


# ── Data shape tests ───────────────────────────────────────────────────────────


def test_stats_shape(client: TestClient) -> None:
    with patch("observability.traces.TraceStore") as mock_cls:
        mock_cls.return_value.get_daily_stats.return_value = {
            "total_requests": 10,
            "total_cost_usd": 0.05,
        }
        resp = client.get("/api/v1/stats", headers={"Authorization": f"Bearer {_TOKEN}"})
    data = resp.json()
    assert "total_requests" in data


def test_skills_covers_all_intents(client: TestClient) -> None:
    from core.types import Intent

    with patch("memory.skill_library.SkillLibrary") as mock_cls:
        mock_cls.return_value.get_top_skills.return_value = ["pattern1"]
        resp = client.get("/api/v1/skills", headers={"Authorization": f"Bearer {_TOKEN}"})

    data = resp.json()
    for intent in Intent:
        assert intent.value in data


def test_root_returns_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Sovereign Edge" in resp.text
