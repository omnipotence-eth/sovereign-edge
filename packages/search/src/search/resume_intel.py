"""
Resume intelligence — extract skill profile from PDF resumes.

Reads all PDFs from SE_CAREER_RESUME_PATH (default ~/Documents/Job Search/Resumes/).
Extracts skills via keyword matching against a curated ML/AI/engineering taxonomy.
No paid API required — pure text extraction via pypdf.

Install pypdf: `uv add "pypdf>=4.0,<5.0"` in the sovereign-edge-search package.
Falls back gracefully (empty profile) when pypdf is missing or no PDFs are found.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Skill taxonomy ────────────────────────────────────────────────────────────
# Grouped by category for clean system-prompt injection.
# Keep ordered: longer multi-word skills first within each list (avoids partial matches).

_TAXONOMY: dict[str, list[str]] = {
    "llm_serving": [
        "tensorrt-llm",
        "exllamav2",
        "vllm",
        "sglang",
        "tensorrt",
        "litellm",
        "ollama",
        "openai api",
        "inference server",
    ],
    "training": [
        "fine-tuning",
        "fine tuning",
        "supervised fine-tuning",
        "grpo",
        "orpo",
        "rlhf",
        "dpo",
        "sft",
        "qlora",
        "lora",
        "peft",
        "trl",
        "unsloth",
    ],
    "agentic_frameworks": [
        "model context protocol",
        "langgraph",
        "langchain",
        "llamaindex",
        "llama-index",
        "multi-agent",
        "tool use",
        "function calling",
        "mcp",
        "orchestration",
        "agentic",
    ],
    "mlops": [
        "weights & biases",
        "wandb",
        "opentelemetry",
        "mlflow",
        "otel",
        "prometheus",
        "grafana",
        "structured logging",
        "dvc",
        "ci/cd",
    ],
    "languages": [
        "typescript",
        "javascript",
        "python",
        "rust",
        "bash",
        "shell",
        "sql",
        "go",
    ],
    "ml_core": [
        "hugging face",
        "huggingface",
        "transformers",
        "accelerate",
        "datasets",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "pytorch",
        "tensorflow",
        "numpy",
        "pandas",
        "jax",
    ],
    "infrastructure": [
        "github actions",
        "kubernetes",
        "fastapi",
        "docker",
        "postgresql",
        "redis",
        "sqlite",
        "aws",
        "gcp",
        "azure",
    ],
    "edge_hardware": [
        "jetson orin",
        "jetson",
        "blackwell",
        "edge ai",
        "arm64",
        "jetpack",
        "fp8",
        "cuda",
        "rtx",
    ],
    "rag_search": [
        "retrieval augmented generation",
        "retrieval augmented",
        "vector database",
        "hybrid search",
        "chromadb",
        "lancedb",
        "weaviate",
        "pinecone",
        "reranking",
        "embedding",
        "bm25",
        "rag",
    ],
}

# Flat list: (skill_text, category)
_SKILL_PAIRS: list[tuple[str, str]] = [
    (skill, cat) for cat, skills in _TAXONOMY.items() for skill in skills
]


@dataclass
class ResumeProfile:
    """Extracted skill profile built from one or more PDF resumes."""

    skills: dict[str, list[str]] = field(default_factory=dict)
    all_skills_flat: list[str] = field(default_factory=list)
    resume_count: int = 0

    def is_empty(self) -> bool:
        return not self.all_skills_flat

    def to_context_string(self) -> str:
        """Compact skills summary suitable for LLM system prompt injection."""
        if self.is_empty():
            return ""
        lines = [f"CANDIDATE SKILL PROFILE ({self.resume_count} resume(s) parsed):"]
        for cat, skills_list in self.skills.items():
            label = cat.replace("_", " ").title()
            lines.append(f"  {label}: {', '.join(skills_list)}")
        return "\n".join(lines)


def _extract_text(pdf_path: Path) -> str:
    """Return lowercased text from a PDF. Returns '' if pypdf is unavailable."""
    try:
        import pypdf  # type: ignore[import-untyped]

        reader = pypdf.PdfReader(str(pdf_path))
        return " ".join(page.extract_text() or "" for page in reader.pages).lower()
    except ImportError:
        logger.warning(
            "resume_intel_pypdf_missing — install pypdf>=4.0 for resume parsing; "
            "run: uv add 'pypdf>=4.0,<5.0'"
        )
        return ""
    except Exception:
        logger.warning("resume_intel_pdf_read_error path=%s", pdf_path, exc_info=True)
        return ""


def build_resume_profile(resume_dir: Path) -> ResumeProfile:
    """Parse all PDFs in resume_dir and return a unified ResumeProfile.

    Returns an empty profile (no error raised) when:
    - pypdf is not installed
    - the directory does not exist
    - no PDF files are found
    """
    profile = ResumeProfile()

    if not resume_dir.exists():
        logger.info("resume_intel_dir_not_found path=%s", resume_dir)
        return profile

    pdf_files = list(resume_dir.glob("*.pdf"))
    if not pdf_files:
        logger.info("resume_intel_no_pdfs path=%s", resume_dir)
        return profile

    combined = " ".join(_extract_text(p) for p in pdf_files)
    if not combined.strip():
        return profile

    profile.resume_count = len(pdf_files)
    found: dict[str, list[str]] = {cat: [] for cat in _TAXONOMY}

    for skill, cat in _SKILL_PAIRS:
        # Word-boundary matching for short tokens prevents false positives.
        # Example: "go" would match "good" without it.
        if len(skill) <= 5:
            pattern = rf"\b{re.escape(skill)}\b"
        else:
            pattern = re.escape(skill)
        if re.search(pattern, combined) and skill not in found[cat]:
            found[cat].append(skill)

    profile.skills = {cat: sl for cat, sl in found.items() if sl}
    profile.all_skills_flat = [s for sl in found.values() for s in sl]

    logger.info(
        "resume_intel_built resumes=%d categories=%d total_skills=%d",
        profile.resume_count,
        len(profile.skills),
        len(profile.all_skills_flat),
    )
    return profile
