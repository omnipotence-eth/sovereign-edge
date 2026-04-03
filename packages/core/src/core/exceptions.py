from __future__ import annotations


class SovereignError(Exception):
    """Base exception for all Sovereign Edge errors."""


class ConfigurationError(SovereignError):
    """Raised when required configuration is missing or invalid."""


class LLMError(SovereignError):
    """Raised when all LLM providers fail or return unusable output."""


class MemoryError(SovereignError):
    """Raised on memory store read/write failures."""


class RouterError(SovereignError):
    """Raised when the intent router fails to classify input."""


class AgentError(SovereignError):
    """Raised when a squad agent encounters an unrecoverable error."""


class HITLRequired(SovereignError):
    """Raised to signal that human approval is required before proceeding."""

    def __init__(self, action: str, payload: dict | None = None) -> None:
        self.action = action
        self.payload = payload or {}
        super().__init__(f"HITL approval required for: {action}")
