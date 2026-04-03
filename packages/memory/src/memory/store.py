from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import structlog
from core.config import Settings, get_settings
from core.exceptions import MemoryError as SovereignMemoryError
from mem0 import Memory

logger = structlog.get_logger(__name__)


@dataclass
class MemoryEntry:
    text: str
    score: float
    metadata: dict[str, Any]


class MemoryStore:
    """Dual-layer memory: Mem0 for episodic recall, LanceDB for semantic search.

    Mem0 stores structured memories (facts, preferences, past interactions).
    LanceDB stores arbitrary text chunks for RAG retrieval.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._mem0: Memory | None = None
        self._lance: lancedb.DBConnection | None = None

    # ── Mem0 (episodic memory) ────────────────────────────────────────────────

    def _get_mem0(self) -> Memory:
        if self._mem0 is None:
            try:
                self._mem0 = Memory()
                logger.info("memory.mem0_ready")
            except Exception as exc:
                raise SovereignMemoryError(f"Failed to initialize Mem0: {exc}") from exc
        return self._mem0

    def add_memory(self, text: str, *, user_id: str | None = None) -> None:
        uid = user_id or self._settings.mem0_user_id
        try:
            self._get_mem0().add(text, user_id=uid)
            logger.info("memory.add", user_id=uid, length=len(text))
        except Exception as exc:
            logger.error("memory.add_failed", exc_info=True)
            raise SovereignMemoryError(f"Failed to add memory: {exc}") from exc

    def search_memory(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        uid = user_id or self._settings.mem0_user_id
        try:
            results = self._get_mem0().search(query, user_id=uid, limit=limit)
            return [
                MemoryEntry(
                    text=r.get("memory", ""),
                    score=r.get("score", 0.0),
                    metadata=r.get("metadata", {}),
                )
                for r in results.get("results", [])
            ]
        except Exception:
            logger.error("memory.search_failed", exc_info=True)
            return []

    def format_context(self, query: str, *, user_id: str | None = None) -> str:
        """Return a formatted string of top memories for LLM injection."""
        entries = self.search_memory(query, user_id=user_id)
        if not entries:
            return ""
        lines = [f"- {e.text}" for e in entries if e.text]
        return "Relevant context from memory:\n" + "\n".join(lines)

    # ── LanceDB (vector / RAG) ────────────────────────────────────────────────

    def _get_lance(self) -> lancedb.DBConnection:
        if self._lance is None:
            path = self._settings.lancedb_path
            Path(path).mkdir(parents=True, exist_ok=True)
            try:
                self._lance = lancedb.connect(str(path))
                logger.info("memory.lancedb_ready", path=str(path))
            except Exception as exc:
                raise SovereignMemoryError(f"Failed to connect to LanceDB: {exc}") from exc
        return self._lance

    def upsert_chunks(
        self,
        table_name: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Upsert text chunks into a LanceDB table.

        Each chunk dict must contain at least 'text' and 'vector' (list[float]).
        """
        db = self._get_lance()
        try:
            if table_name in db.table_names():
                tbl = db.open_table(table_name)
                tbl.add(chunks)
            else:
                db.create_table(table_name, data=chunks)
            logger.info("memory.upsert", table=table_name, count=len(chunks))
        except Exception as exc:
            logger.error("memory.upsert_failed", table=table_name, exc_info=True)
            raise SovereignMemoryError(f"Failed to upsert chunks: {exc}") from exc

    def vector_search(
        self,
        table_name: str,
        query_vector: list[float],
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        db = self._get_lance()
        try:
            if table_name not in db.table_names():
                return []
            tbl = db.open_table(table_name)
            results = tbl.search(query_vector).limit(limit).to_list()
            return results
        except Exception:
            logger.error("memory.vector_search_failed", table=table_name, exc_info=True)
            return []
