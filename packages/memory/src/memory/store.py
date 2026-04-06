"""Unified memory store — Mem0 episodic memory + LanceDB vector store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.config import Settings
from core.exceptions import SovereignMemoryError
from observability.logging import get_logger

logger = get_logger(__name__, component="memory")

# Imported at module level so tests can patch memory.store.Memory
try:
    from mem0 import Memory  # type: ignore[import-untyped]
except ImportError:
    Memory = None  # type: ignore[assignment,misc]


@dataclass
class MemoryEntry:
    """A single retrieved memory with relevance score."""

    text: str
    score: float
    metadata: dict[str, Any]


class MemoryStore:
    """Wraps Mem0 (episodic memory) and LanceDB (vector store) behind a unified API.

    Both backends are lazily initialised so the class can be instantiated without
    requiring Mem0 or LanceDB to be installed (e.g. in unit tests that mock them).
    Assign to ``_mem0`` or ``_lance`` before calling methods to inject test doubles.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._mem0: Any = None   # set externally in tests; lazy-inited otherwise
        self._lance: Any = None  # set externally in tests; lazy-inited otherwise

    # ── Mem0 (episodic) ───────────────────────────────────────────────────────

    def _get_mem0(self) -> Any:  # noqa: ANN401 — Mem0 type is opaque
        """Lazily initialise Mem0. Raises SovereignMemoryError on failure."""
        if Memory is None:
            raise SovereignMemoryError("Mem0 not installed — install mem0ai")
        try:
            return Memory()
        except Exception as exc:
            raise SovereignMemoryError("Failed to initialize Mem0") from exc

    def add_memory(self, text: str, user_id: str | None = None) -> None:
        """Add a memory to Mem0.

        Uses ``settings.mem0_user_id`` when ``user_id`` is not specified.
        Raises SovereignMemoryError on failure.
        """
        mem0 = self._mem0 if self._mem0 is not None else self._get_mem0()
        uid = user_id if user_id is not None else self._settings.mem0_user_id
        try:
            mem0.add(text, user_id=uid)
        except Exception as exc:
            raise SovereignMemoryError("Failed to add memory") from exc

    def search_memory(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """Search Mem0 for relevant memories. Returns [] on any failure."""
        try:
            mem0 = self._mem0 if self._mem0 is not None else self._get_mem0()
            raw = mem0.search(query, user_id=self._settings.mem0_user_id)
            results = raw.get("results", []) if isinstance(raw, dict) else []
            return [
                MemoryEntry(
                    text=r.get("memory", ""),
                    score=float(r.get("score", 0.0)),
                    metadata=r.get("metadata", {}),
                )
                for r in results[:limit]
            ]
        except Exception:
            logger.warning("memory_store.search_failed", exc_info=True)
            return []

    def format_context(self, query: str) -> str:
        """Return a formatted string of memories relevant to *query*, or '' if none."""
        entries = self.search_memory(query)
        if not entries:
            return ""
        lines = ["Relevant context from memory:"]
        lines.extend(f"- {e.text}" for e in entries)
        return "\n".join(lines)

    # ── LanceDB (vector) ──────────────────────────────────────────────────────

    def _get_lance(self) -> Any:  # noqa: ANN401 — LanceDB connection type is opaque
        """Lazily connect to LanceDB. Raises SovereignMemoryError on failure."""
        try:
            import lancedb  # type: ignore[import-untyped]

            return lancedb.connect(str(self._settings.lancedb_path))
        except Exception as exc:
            raise SovereignMemoryError("Failed to initialize LanceDB") from exc

    def upsert_chunks(self, table_name: str, chunks: list[dict[str, Any]]) -> None:
        """Insert *chunks* into LanceDB, creating the table if it doesn't exist.

        Raises SovereignMemoryError on failure.
        """
        lance = self._lance if self._lance is not None else self._get_lance()
        try:
            if table_name in lance.table_names():
                table = lance.open_table(table_name)
                table.add(chunks)
            else:
                lance.create_table(table_name, data=chunks)
        except Exception as exc:
            raise SovereignMemoryError("Failed to upsert chunks") from exc

    def vector_search(
        self,
        table_name: str,
        vector: list[float],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search *table_name* by vector similarity. Returns [] on missing table or error."""
        lance = self._lance if self._lance is not None else self._get_lance()
        try:
            if table_name not in lance.table_names():
                return []
            table = lance.open_table(table_name)
            return table.search(vector).limit(limit).to_list()
        except Exception:
            logger.warning("memory_store.vector_search_failed", exc_info=True)
            return []
