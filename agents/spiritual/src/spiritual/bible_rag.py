from __future__ import annotations

import csv
import re
from pathlib import Path

import structlog
from memory.store import MemoryStore

logger = structlog.get_logger(__name__)

_BIBLE_TABLE = "bible_verses"
_TSK_TABLE = "tsk_references"
_STEPBIBLE_PATH = Path("data/stepbible")
_TSK_PATH = Path("data/tsk")

# Allowlist for reference strings — only book names, digits, spaces, dots, colons.
# Rejects any character that could be used for SQL/filter injection in LanceDB.
_REF_SAFE = re.compile(r"^[A-Za-z0-9 .:]+$")


def _parse_ref(ref: str) -> tuple[str, int, int] | None:
    """Parse 'Gen.1.1' or 'GEN 1:1' style references. Returns (book, chapter, verse)."""
    ref = ref.strip()
    for sep in (".", ":"):
        parts = ref.split(sep)
        if len(parts) >= 3:
            try:
                return parts[0], int(parts[1]), int(parts[2])
            except ValueError:
                pass
    return None


class BibleRAG:
    """Retrieval layer over STEPBible data stored in LanceDB.

    On first call, loads verses from CSV if the LanceDB table is empty.
    Embedding is handled via Ollama nomic-embed-text (local, no cloud).
    """

    _EMBED_MODEL = "nomic-embed-text"

    def __init__(self) -> None:
        self._store = MemoryStore()
        self._embedder: object | None = None

    def _get_embedder(self) -> object | None:
        """Return the Ollama client, or None if not installed."""
        if self._embedder is None:
            try:
                import ollama  # type: ignore

                self._embedder = ollama
            except ImportError:
                logger.warning("spiritual.rag.ollama_not_installed")
        return self._embedder

    def _embed(self, text: str) -> list[float] | None:
        embedder = self._get_embedder()
        if embedder is None:
            return None
        try:
            resp = embedder.embeddings(model=self._EMBED_MODEL, prompt=text)  # type: ignore
            return resp["embedding"]  # type: ignore[index]
        except Exception:
            logger.error("spiritual.rag.embed_failed", exc_info=True)
            return None

    def is_indexed(self) -> bool:
        db = self._store._get_lance()
        return _BIBLE_TABLE in db.table_names()

    def index_stepbible(self) -> int:
        """Index STEPBible CSV files into LanceDB. Returns number of verses indexed."""
        csv_files = list(_STEPBIBLE_PATH.glob("*.csv"))
        if not csv_files:
            logger.warning("spiritual.rag.no_stepbible_data", path=str(_STEPBIBLE_PATH))
            return 0

        chunks = []
        for csv_file in csv_files:
            try:
                with open(csv_file, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        text = row.get("EnglishText") or row.get("text") or ""
                        ref = row.get("OsisRef") or row.get("ref") or ""
                        if not text or not ref:
                            continue
                        vector = self._embed(text)
                        if vector is None:
                            continue
                        chunks.append(
                            {
                                "vector": vector,
                                "text": text,
                                "ref": ref,
                                "book": ref.split(".")[0] if "." in ref else "",
                            }
                        )
            except Exception:
                logger.error("spiritual.rag.index_failed", file=str(csv_file), exc_info=True)

        if chunks:
            self._store.upsert_chunks(_BIBLE_TABLE, chunks)
        logger.info("spiritual.rag.indexed", count=len(chunks))
        return len(chunks)

    def search(self, query: str, *, limit: int = 5) -> list[dict]:
        vector = self._embed(query)
        if vector is None:
            return []
        return self._store.vector_search(_BIBLE_TABLE, vector, limit=limit)

    def lookup_verse(self, reference: str) -> str | None:
        """Direct lookup by reference string (e.g. 'John 3:16').

        Only references matching [A-Za-z0-9 .:]+ are accepted; all others are
        rejected before reaching the database to prevent filter injection.
        """
        ref_stripped = reference.strip()
        if not _REF_SAFE.match(ref_stripped):
            logger.warning("spiritual.rag.invalid_ref_rejected", ref=ref_stripped[:50])
            return None

        db = self._store._get_lance()
        if _BIBLE_TABLE not in db.table_names():
            return None
        try:
            tbl = db.open_table(_BIBLE_TABLE)
            norm = ref_stripped.replace(" ", ".").replace(":", ".")
            results = tbl.search(norm).where(f"ref LIKE '%{norm}%'").limit(1).to_list()
            return results[0]["text"] if results else None
        except Exception:
            logger.error("spiritual.rag.lookup_failed", ref=ref_stripped, exc_info=True)
            return None
