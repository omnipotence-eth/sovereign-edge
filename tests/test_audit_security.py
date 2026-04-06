"""Security and correctness audit tests — cross-package coverage.

Covers audit fixes applied during the professional security review:
1. ONNX bounds-check fallback (router.classifier)
2. Jina cache LRU eviction hard-cap (_MAX_CACHE_SIZE)
3. WhatsApp Content-Type guard (HTTP 415)
4. Health server X-Request-ID propagation
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

# ── 1. ONNX out-of-range index → keyword fallback ─────────────────────────────


def test_onnx_out_of_range_idx_falls_back_to_keywords() -> None:
    """_classify_onnx must not raise when argmax index >= len(Intent).

    Guards against a model trained with more output classes than the current
    Intent enum — produces a safe keyword-fallback result instead of IndexError.
    """
    import numpy as np
    from core.types import Intent
    from router.classifier import IntentRouter

    router = IntentRouter()
    router._use_onnx = True

    # Tokenizer mock: callable, returns dict with input_ids + attention_mask
    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": np.zeros((1, 4), dtype=np.int64),
        "attention_mask": np.ones((1, 4), dtype=np.int64),
    }
    router._tokenizer = mock_tok

    # ONNX output with 10 classes (> 6 Intent values); argmax at index 9 → OOB
    n_classes = 10  # more than len(Intent) = 6
    logits = np.zeros(n_classes)
    logits[-1] = 5.0  # argmax = 9, which is >= len(list(Intent)) = 6
    mock_session = MagicMock()
    mock_session.run.return_value = [np.array([logits])]
    router._onnx_session = mock_session

    # Must not raise; keyword fallback maps "bible" → SPIRITUAL
    intent, confidence = router._classify_onnx("bible verse about faith")

    assert intent == Intent.SPIRITUAL
    assert 0.0 < confidence <= 1.0


def test_onnx_in_range_idx_returned_directly() -> None:
    """When argmax is within range, _classify_onnx returns that label directly."""
    import numpy as np
    from core.types import Intent
    from router.classifier import IntentRouter

    router = IntentRouter()
    router._use_onnx = True

    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": np.zeros((1, 4), dtype=np.int64),
        "attention_mask": np.ones((1, 4), dtype=np.int64),
    }
    router._tokenizer = mock_tok

    labels = list(Intent)
    # Put the max at index 0 (first label)
    logits = np.zeros(len(labels))
    logits[0] = 5.0
    mock_session = MagicMock()
    mock_session.run.return_value = [np.array([logits])]
    router._onnx_session = mock_session

    intent, confidence = router._classify_onnx("anything")

    assert intent == labels[0]
    assert confidence > 0.0


# ── 2. Jina cache hard-cap eviction ───────────────────────────────────────────


def test_jina_cache_does_not_exceed_max_size() -> None:
    """_cache_set must enforce _MAX_CACHE_SIZE and not grow unboundedly."""
    import search.jina as jina

    original = dict(jina._search_cache)
    jina._search_cache.clear()
    try:
        # Add MAX_CACHE_SIZE + 1 unique entries via the public _cache_set path
        n = jina._MAX_CACHE_SIZE + 1
        for i in range(n):
            jina._cache_set(f"audit_key_{i}", f"value_{i}")

        assert len(jina._search_cache) <= jina._MAX_CACHE_SIZE
    finally:
        jina._search_cache.clear()
        jina._search_cache.update(original)


def test_jina_cache_evicts_lowest_expiry_entry() -> None:
    """When the cap is exceeded the entry with the earliest expiry is dropped."""
    import search.jina as jina

    original = dict(jina._search_cache)
    jina._search_cache.clear()
    try:
        # Insert a stale-ish entry with a very low expiry (inserted directly)
        stale_key = "_audit_stale_"
        jina._search_cache[stale_key] = ("stale", time.monotonic() + 0.1)

        # Fill the remaining slots via _cache_set (each gets now + 1800)
        for i in range(jina._MAX_CACHE_SIZE):
            jina._cache_set(f"_audit_fresh_{i}", "fresh")

        # stale_key must have been evicted (smallest expiry in a full cache)
        assert stale_key not in jina._search_cache
        assert len(jina._search_cache) <= jina._MAX_CACHE_SIZE
    finally:
        jina._search_cache.clear()
        jina._search_cache.update(original)


def test_jina_cache_expired_entries_evicted_on_periodic_flush() -> None:
    """Every 50th write calls _evict_expired(), removing already-expired entries."""
    import search.jina as jina

    original = dict(jina._search_cache)
    jina._search_cache.clear()
    try:
        # Insert an already-expired entry directly
        expired_key = "_audit_expired_"
        jina._search_cache[expired_key] = ("expired", time.monotonic() - 1.0)

        # Trigger the periodic eviction by writing 50 entries
        for i in range(50):
            jina._cache_set(f"_audit_trigger_{i}", "val")

        # The expired entry should have been cleared
        assert expired_key not in jina._search_cache
    finally:
        jina._search_cache.clear()
        jina._search_cache.update(original)


# ── 3. WhatsApp Content-Type guard ────────────────────────────────────────────


def test_whatsapp_rejects_json_content_type() -> None:
    """Webhook returns 415 when Content-Type is application/json."""
    from fastapi.testclient import TestClient
    from pydantic import SecretStr
    from whatsapp.bot import app

    mock_settings = MagicMock()
    mock_settings.twilio_account_sid = SecretStr("ACtest")
    mock_settings.twilio_auth_token = SecretStr("authtest")
    mock_settings.twilio_whatsapp_from = "+19995550000"
    mock_settings.whatsapp_owner_number = "+12145550001"

    client = TestClient(app, raise_server_exceptions=False)
    with patch("whatsapp.bot.get_settings", return_value=mock_settings):
        resp = client.post(
            "/webhook",
            content=b'{"From": "whatsapp:+12145550001", "Body": "hi"}',
            headers={"Content-Type": "application/json"},
        )
    assert resp.status_code == 415


def test_whatsapp_accepts_urlencoded_past_content_type_check() -> None:
    """application/x-www-form-urlencoded passes the 415 guard (hits next check)."""
    from fastapi.testclient import TestClient
    from pydantic import SecretStr
    from whatsapp.bot import app

    mock_settings = MagicMock()
    mock_settings.twilio_account_sid = SecretStr("ACtest")
    mock_settings.twilio_auth_token = SecretStr("authtest")
    mock_settings.twilio_whatsapp_from = "+19995550000"
    mock_settings.whatsapp_owner_number = "+12145550001"

    client = TestClient(app, raise_server_exceptions=False)
    with (
        patch("whatsapp.bot.get_settings", return_value=mock_settings),
        patch("whatsapp.bot._validate_twilio_signature", return_value=False),
    ):
        resp = client.post(
            "/webhook",
            data={"From": "whatsapp:+12145550001", "Body": "hi"},
        )
    # Passes 415 check, fails at signature (403)
    assert resp.status_code == 403


# ── 4. Health server X-Request-ID propagation ─────────────────────────────────


def test_health_response_has_request_id_header() -> None:
    """Every response must include an X-Request-ID header."""
    from fastapi.testclient import TestClient
    from health.server import app

    mock_settings = MagicMock()
    mock_settings.lancedb_path = "/nonexistent"
    mock_settings.logs_path = "/nonexistent"
    mock_settings.ssd_root = "/nonexistent"

    client = TestClient(app, raise_server_exceptions=False)
    with patch("health.server.get_settings", return_value=mock_settings):
        resp = client.get("/health")

    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) > 0


def test_health_propagates_caller_request_id() -> None:
    """When client sends X-Request-ID, the same value must be echoed back."""
    from fastapi.testclient import TestClient
    from health.server import app

    mock_settings = MagicMock()
    mock_settings.lancedb_path = "/nonexistent"
    mock_settings.logs_path = "/nonexistent"
    mock_settings.ssd_root = "/nonexistent"

    client = TestClient(app, raise_server_exceptions=False)
    with patch("health.server.get_settings", return_value=mock_settings):
        resp = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})

    assert resp.headers.get("x-request-id") == "trace-abc-123"


def test_health_generates_uuid_when_no_request_id() -> None:
    """Without an inbound X-Request-ID, the server generates a UUID."""
    import re

    from fastapi.testclient import TestClient
    from health.server import app

    _UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

    mock_settings = MagicMock()
    mock_settings.lancedb_path = "/nonexistent"
    mock_settings.logs_path = "/nonexistent"
    mock_settings.ssd_root = "/nonexistent"

    client = TestClient(app, raise_server_exceptions=False)
    with patch("health.server.get_settings", return_value=mock_settings):
        resp = client.get("/health")

    request_id = resp.headers.get("x-request-id", "")
    assert _UUID_RE.match(request_id), f"Expected UUID, got: {request_id!r}"
