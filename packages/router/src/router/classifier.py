from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from core.config import Settings, get_settings
from core.exceptions import RouterError
from core.types import IntentClass, RouterResult

logger = structlog.get_logger(__name__)

# ── Keyword fallback rules (used when ONNX model is absent) ──────────────────
_KEYWORD_RULES: dict[IntentClass, list[str]] = {
    IntentClass.SPIRITUAL: [
        "bible", "scripture", "verse", "psalm", "proverb", "gospel", "jesus",
        "god", "prayer", "pray", "devotional", "theology", "faith", "church",
        "holy", "spirit", "salvation", "grace", "baptism", "matthew", "john",
        "genesis", "exodus", "isaiah", "romans", "corinthians", "revelation",
    ],
    IntentClass.CAREER: [
        "job", "jobs", "resume", "cv", "career", "interview", "salary",
        "hiring", "linkedin", "application", "apply", "recruiter", "role",
        "position", "engineer", "developer", "opening", "offer", "cover letter",
        "glassdoor", "indeed", "work", "employment",
    ],
    IntentClass.INTELLIGENCE: [
        "market", "stock", "price", "nvda", "nvidia", "msft", "microsoft",
        "aapl", "apple", "googl", "google", "meta", "portfolio", "ticker",
        "arxiv", "paper", "research", "ai news", "llm", "model release",
        "rss", "news", "trend", "alpha vantage", "finance", "s&p", "nasdaq",
    ],
    IntentClass.CREATIVE: [
        "video", "script", "youtube", "content", "manim", "animation",
        "diagram", "generate", "write a", "create a", "design", "tts",
        "narrate", "voiceover", "social media", "post", "tweet", "reel",
    ],
}


class IntentRouter:
    """Classifies user intent to one of four IntentClass values.

    Attempts ONNX DistilBERT INT8 inference first; falls back to
    keyword matching when the model file is absent (development mode).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._session: Any = None  # onnxruntime.InferenceSession
        self._tokenizer: Any = None  # transformers.AutoTokenizer
        self._labels = list(IntentClass)
        self._loaded = False

    def _load(self) -> bool:
        """Lazy-load ONNX model + tokenizer. Returns True if model is available."""
        if self._loaded:
            return self._session is not None

        model_path = self._settings.router_model_path
        if not Path(model_path).exists():
            logger.warning(
                "router.model_not_found",
                path=str(model_path),
                fallback="keyword",
            )
            self._loaded = True
            return False

        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._settings.router_tokenizer_name
            )
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            logger.info("router.model_loaded", path=str(model_path))
        except Exception as exc:
            logger.error("router.load_failed", exc_info=True)
            raise RouterError(f"Failed to load router model: {exc}") from exc

        self._loaded = True
        return True

    def classify(self, text: str) -> RouterResult:
        """Classify text into an IntentClass with confidence score.

        Raises RouterError only if ONNX inference fails hard (not on fallback).
        """
        if not text or not text.strip():
            raise RouterError("Cannot classify empty input")

        t0 = time.perf_counter()

        if self._load() and self._session is not None:
            result = self._onnx_classify(text)
        else:
            result = self._keyword_classify(text)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "router.classified",
            intent=result.intent.value,
            confidence=round(result.confidence, 3),
            latency_ms=round(elapsed_ms, 2),
            method="onnx" if self._session else "keyword",
        )
        return result

    def _onnx_classify(self, text: str) -> RouterResult:
        inputs = self._tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        feeds = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        logits = self._session.run(None, feeds)[0][0]
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        return RouterResult(intent=self._labels[idx], confidence=float(probs[idx]))

    def _keyword_classify(self, text: str) -> RouterResult:
        lower = text.lower()
        scores: dict[IntentClass, int] = {cls: 0 for cls in IntentClass}
        for intent, keywords in _KEYWORD_RULES.items():
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                    scores[intent] += 1

        best = max(scores, key=lambda k: scores[k])
        total = sum(scores.values())

        if total == 0:
            # Default to intelligence (most general) with low confidence
            return RouterResult(intent=IntentClass.INTELLIGENCE, confidence=0.4)

        confidence = scores[best] / total
        return RouterResult(intent=best, confidence=min(confidence, 0.95))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
