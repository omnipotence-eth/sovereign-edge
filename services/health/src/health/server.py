"""Lightweight health / readiness endpoint.

Runs as a background thread inside the main process, or as a standalone
process via ``python -m health.server``.

Endpoints
---------
GET /health  — liveness probe; always returns 200 if the process is up.
GET /ready   — readiness probe; checks that persistent storage is accessible.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import uvicorn
from core.config import get_settings
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from health.dashboard import router as dashboard_router

logger = logging.getLogger(__name__)

_START_TIME: float = time.monotonic()
_STATIC_DIR = Path(__file__).parent.parent.parent / "static"

app = FastAPI(
    title="Sovereign Edge Health",
    version="0.2.0",
    docs_url=None,  # disable Swagger in production
    redoc_url=None,
)

# CORS — allow browser requests only from localhost (dashboard is container-internal)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
)


@app.middleware("http")
async def _add_request_id(request: Request, call_next: object) -> JSONResponse:
    """Attach a correlation ID to every request and response."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response = await call_next(request)  # type: ignore[operator]
    response.headers["X-Request-ID"] = request_id
    return response  # type: ignore[return-value]


# Mount static assets and dashboard routes
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
app.include_router(dashboard_router)


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe — returns 200 as long as the process is alive."""
    return {
        "status": "ok",
        "uptime_s": round(time.monotonic() - _START_TIME, 2),
    }


@app.get("/ready")
def ready(response: Response) -> dict[str, object]:
    """Readiness probe — verifies that persistent storage paths exist.

    Returns 200 when all required paths are present, 503 otherwise.
    The Docker HEALTHCHECK uses ``/health`` (liveness), so a failing readiness
    probe won't restart the container — it is intended for orchestrators and
    deployment scripts that want to know when the bot is truly ready.
    """
    settings = get_settings()

    checks: dict[str, bool] = {
        "lancedb": Path(settings.lancedb_path).exists(),
        "logs_dir": Path(settings.logs_path).exists(),
        "data_root": Path(settings.ssd_root).exists(),
    }

    all_ok = all(checks.values())
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning("readiness_check_failed checks=%s", checks)
    else:
        logger.debug("readiness_check_passed checks=%s", checks)

    return {
        "status": "ready" if all_ok else "not_ready",
        "checks": checks,
        "uptime_s": round(time.monotonic() - _START_TIME, 2),
    }


def main() -> None:
    """Entry point for the ``health-server`` console script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("Starting health server on :8080")
    uvicorn.run(
        "health.server:app",
        host="0.0.0.0",  # noqa: S104 — intentional; container-internal only
        port=8080,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
