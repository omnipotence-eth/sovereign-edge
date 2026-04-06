"""Custom exception hierarchy for Sovereign Edge."""

from __future__ import annotations


class SovereignEdgeError(Exception):
    """Base for all Sovereign Edge exceptions."""


class LLMError(SovereignEdgeError):
    """Raised when the LLM gateway fails to produce a response."""


class SovereignMemoryError(SovereignEdgeError):
    """Raised when memory operations (Mem0 or LanceDB) fail."""


class RouterError(SovereignEdgeError):
    """Raised when the intent router cannot process input."""
