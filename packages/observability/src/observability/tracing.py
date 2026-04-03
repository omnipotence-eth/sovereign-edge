from __future__ import annotations

import functools
import uuid
from collections.abc import Callable
from typing import Any, TypeVar

import structlog
from opentelemetry import trace
from opentelemetry.trace import Span

F = TypeVar("F", bound=Callable[..., Any])

_SERVICE_NAME = "sovereign-edge"


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def traced(span_name: str | None = None) -> Callable[[F], F]:
    """Decorator that wraps a function in an OTEL span and binds a correlation ID."""

    def decorator(fn: F) -> F:
        name = span_name or fn.__qualname__

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            tracer = get_tracer(_SERVICE_NAME)
            correlation_id = str(uuid.uuid4())
            structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("correlation_id", correlation_id)
                return await fn(*args, **kwargs)

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            tracer = get_tracer(_SERVICE_NAME)
            correlation_id = str(uuid.uuid4())
            structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("correlation_id", correlation_id)
                return fn(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def current_span() -> Span:
    return trace.get_current_span()
