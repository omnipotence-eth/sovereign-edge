from __future__ import annotations

from observability.setup import setup_observability
from observability.tracing import get_tracer, traced

__all__ = ["setup_observability", "get_tracer", "traced"]
