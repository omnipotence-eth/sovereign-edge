"""Tests for whatsapp.bot."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from whatsapp.bot import app

_OWNER = "+12145550001"
_FROM = f"whatsapp:{_OWNER}"
_BODY = "Show me my goals"
_SIG = "valid-sig"

# ── Settings fixture ───────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _mock_settings() -> None:  # type: ignore[return]
    from pydantic import SecretStr

    settings = MagicMock()
    settings.twilio_account_sid = SecretStr("ACtest")
    settings.twilio_auth_token = SecretStr("authtest")
    settings.twilio_whatsapp_from = "+19995550000"
    settings.whatsapp_owner_number = _OWNER

    with patch("whatsapp.bot.get_settings", return_value=settings):
        yield


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ── Signature validation ───────────────────────────────────────────────────────


def test_rejects_missing_signature(client: TestClient) -> None:
    """Requests without X-Twilio-Signature must be rejected."""
    with patch("whatsapp.bot._validate_twilio_signature", return_value=False):
        resp = client.post("/webhook", data={"From": _FROM, "Body": _BODY})
    assert resp.status_code == 403


def test_rejects_invalid_signature(client: TestClient) -> None:
    """Requests with a wrong signature must be rejected."""
    with patch("whatsapp.bot._validate_twilio_signature", return_value=False):
        resp = client.post(
            "/webhook",
            data={"From": _FROM, "Body": _BODY},
            headers={"X-Twilio-Signature": "bad-sig"},
        )
    assert resp.status_code == 403


# ── Owner check ────────────────────────────────────────────────────────────────


def test_rejects_wrong_sender(client: TestClient) -> None:
    """Messages from non-owner numbers must be rejected."""
    with patch("whatsapp.bot._validate_twilio_signature", return_value=True):
        resp = client.post(
            "/webhook",
            data={"From": "whatsapp:+19999999999", "Body": _BODY},
            headers={"X-Twilio-Signature": _SIG},
        )
    assert resp.status_code == 403


# ── Happy path ────────────────────────────────────────────────────────────────


def test_dispatches_to_orchestrator(client: TestClient) -> None:
    """Valid owner message is dispatched and a reply is sent."""
    mock_result = MagicMock()
    mock_result.content = "Here are your goals: ..."

    mock_orch = MagicMock()
    mock_orch.dispatch = AsyncMock(return_value=mock_result)

    with (
        patch("whatsapp.bot._validate_twilio_signature", return_value=True),
        patch("whatsapp.bot._get_orchestrator", return_value=mock_orch),
        patch("whatsapp.bot._send_whatsapp", new_callable=AsyncMock),
        patch("router.classifier.IntentRouter") as mock_router_cls,
    ):
        from core.types import Intent, RoutingDecision

        mock_router_cls.return_value.aroute = AsyncMock(
            return_value=(Intent.GOALS, 0.9, RoutingDecision.CLOUD)
        )
        resp = client.post(
            "/webhook",
            data={"From": _FROM, "Body": _BODY},
            headers={"X-Twilio-Signature": _SIG},
        )

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Media / empty body ────────────────────────────────────────────────────────


def test_handles_empty_body_gracefully(client: TestClient) -> None:
    """Empty body (e.g. image-only message) returns 200 without dispatching."""
    with patch("whatsapp.bot._validate_twilio_signature", return_value=True):
        resp = client.post(
            "/webhook",
            data={"From": _FROM, "Body": ""},
            headers={"X-Twilio-Signature": _SIG},
        )
    assert resp.status_code == 200
