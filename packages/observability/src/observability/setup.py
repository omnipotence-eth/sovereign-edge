from __future__ import annotations

import structlog
from core.config import Settings
from core.logging import configure_logging
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = structlog.get_logger(__name__)


def setup_observability(settings: Settings) -> None:
    """Bootstrap logging and tracing for the process.

    Call once at startup, before importing agent modules.
    """
    configure_logging(json=settings.log_json, level=settings.log_level)
    _setup_tracing(settings)
    logger.info("observability.ready", json=settings.log_json, level=settings.log_level)


def _setup_tracing(settings: Settings) -> None:
    resource = Resource.create({"service.name": "sovereign-edge"})
    provider = TracerProvider(resource=resource)

    if settings.otel_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("tracing.otlp", endpoint=settings.otel_endpoint)
        except Exception:
            logger.warning("tracing.otlp_failed", exc_info=True)
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        # No-op in dev when no endpoint is configured
        pass

    trace.set_tracer_provider(provider)
