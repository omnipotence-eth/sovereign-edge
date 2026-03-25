"""
Ollama embedding client — local, free, zero API cost.

Returns unit-normalized float32 vectors for semantic similarity tasks.
Degrades gracefully: all functions return None when Ollama is unavailable.

Default model: configured via SE_EMBEDDING_MODEL (default: qwen3-embedding:0.6b).
Typical latency: 20-80ms on CPU, 5-20ms on GPU.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from core.config import get_settings

logger = logging.getLogger(__name__)


def embed(text: str) -> np.ndarray | None:
    """Synchronous embedding. Returns unit-normalized float32 vector or None."""
    settings = get_settings()
    try:
        import httpx

        resp = httpx.post(
            f"{settings.ollama_host}/api/embeddings",
            json={"model": settings.embedding_model, "prompt": text},
            timeout=10.0,
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["embedding"], dtype=np.float32)
        return _normalize(vec)
    except Exception as exc:
        logger.debug("embed_failed model=%s: %s", settings.embedding_model, exc)
        return None


async def aembed(text: str) -> np.ndarray | None:
    """Async embedding. Returns unit-normalized float32 vector or None."""
    settings = get_settings()
    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{settings.ollama_host}/api/embeddings",
                json={"model": settings.embedding_model, "prompt": text},
            )
            resp.raise_for_status()
            vec = np.array(resp.json()["embedding"], dtype=np.float32)
            return _normalize(vec)
    except Exception as exc:
        logger.debug("aembed_failed model=%s: %s", settings.embedding_model, exc)
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two unit-normalized vectors equals cosine similarity."""
    return float(np.dot(a, b))


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0.0 else vec


@lru_cache(maxsize=512)
def embed_cached(text: str) -> tuple[float, ...] | None:
    """
    LRU-cached sync embedding — returns tuple (hashable) or None.

    Use for exemplar/prototype embeddings that are computed once and reused.
    Not suitable for user queries (too many unique strings to benefit from caching).
    """
    result = embed(text)
    return tuple(result.tolist()) if result is not None else None
