"""Tests for memory.conversation.ConversationStore — audit additions.

Covers:
- chat_id validation: empty and whitespace-only are silently rejected
- role validation: only 'user', 'assistant', 'system' are accepted
- ordering guarantee: get_recent returns oldest-first
- n-limit: get_recent(n=k) returns at most k turns
- clear: removes only the specified chat's history
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def store(tmp_path: Path):
    """ConversationStore backed by a temp SQLite file (no real settings needed)."""
    from memory.conversation import ConversationStore

    mock_settings = MagicMock()
    mock_settings.logs_path = tmp_path

    with patch("memory.conversation.get_settings", return_value=mock_settings):
        return ConversationStore()


# ── chat_id validation ─────────────────────────────────────────────────────────


def test_empty_chat_id_is_silently_skipped(store) -> None:
    """Empty chat_id should be skipped without raising."""
    store.add_turn("", "user", "hello")
    assert store.get_recent("") == []


def test_whitespace_chat_id_is_silently_skipped(store) -> None:
    """Whitespace-only chat_id is invalid and should be silently skipped."""
    store.add_turn("   ", "user", "hello")
    assert store.get_recent("   ") == []


# ── role validation ────────────────────────────────────────────────────────────


def test_invalid_role_human_is_skipped(store) -> None:
    store.add_turn("chat1", "human", "hello")  # "human" not in valid set
    assert store.get_recent("chat1") == []


def test_invalid_role_bot_is_skipped(store) -> None:
    store.add_turn("chat1", "bot", "reply")
    assert store.get_recent("chat1") == []


def test_invalid_role_empty_is_skipped(store) -> None:
    store.add_turn("chat1", "", "reply")
    assert store.get_recent("chat1") == []


def test_role_user_is_accepted(store) -> None:
    store.add_turn("chat1", "user", "hello")
    turns = store.get_recent("chat1")
    assert len(turns) == 1
    assert turns[0]["role"] == "user"
    assert turns[0]["content"] == "hello"


def test_role_assistant_is_accepted(store) -> None:
    store.add_turn("chat1", "assistant", "I can help with that")
    turns = store.get_recent("chat1")
    assert len(turns) == 1
    assert turns[0]["role"] == "assistant"


def test_role_system_is_accepted(store) -> None:
    store.add_turn("chat1", "system", "You are a helpful assistant")
    turns = store.get_recent("chat1")
    assert len(turns) == 1
    assert turns[0]["role"] == "system"


# ── ordering guarantee ────────────────────────────────────────────────────────


def test_get_recent_returns_oldest_first(store) -> None:
    """Turns must be returned chronologically (oldest first)."""
    store.add_turn("chat1", "user", "first")
    store.add_turn("chat1", "assistant", "second")
    store.add_turn("chat1", "user", "third")

    turns = store.get_recent("chat1")
    assert [t["content"] for t in turns] == ["first", "second", "third"]


def test_get_recent_n_limit(store) -> None:
    """get_recent(n=2) returns only the 2 most recent turns, oldest-first."""
    for i in range(5):
        store.add_turn("chat1", "user", f"msg{i}")

    turns = store.get_recent("chat1", n=2)
    assert len(turns) == 2
    # Most recent two messages are msg3 and msg4 (oldest-first)
    assert turns[0]["content"] == "msg3"
    assert turns[1]["content"] == "msg4"


# ── clear ─────────────────────────────────────────────────────────────────────


def test_clear_removes_all_turns_for_chat(store) -> None:
    store.add_turn("chat1", "user", "hello")
    store.add_turn("chat1", "assistant", "hi")
    store.clear("chat1")
    assert store.get_recent("chat1") == []


def test_clear_does_not_affect_other_chats(store) -> None:
    store.add_turn("chat1", "user", "hello")
    store.add_turn("chat2", "user", "world")
    store.clear("chat1")
    assert store.get_recent("chat1") == []
    assert len(store.get_recent("chat2")) == 1


# ── get_recent_json ────────────────────────────────────────────────────────────


def test_get_recent_json_returns_empty_string_when_no_turns(store) -> None:
    result = store.get_recent_json("nonexistent_chat")
    assert result == ""


def test_get_recent_json_returns_valid_json(store) -> None:
    import json

    store.add_turn("chat1", "user", "test message")
    result = store.get_recent_json("chat1")
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert parsed[0]["content"] == "test message"
