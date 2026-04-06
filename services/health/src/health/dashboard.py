"""Dashboard API routes — all require Bearer token auth.

Routes
------
GET /api/v1/stats    — today's usage and cost (TraceStore)
GET /api/v1/memory   — recent episodic memory entries
GET /api/v1/skills   — top skill patterns per intent
GET /api/v1/briefs   — last 7 days of morning brief traces
GET /api/v1/jobs     — most recent career trace content
GET /                — HTMX dashboard HTML
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse

from health.auth import require_token

logger = logging.getLogger(__name__)

router = APIRouter()
_API = APIRouter(prefix="/api/v1", dependencies=[Depends(require_token)])


@_API.get("/stats")
def get_stats() -> dict[str, Any]:
    """Today's aggregate usage stats from TraceStore."""
    try:
        from observability.traces import TraceStore

        stats = TraceStore().get_daily_stats()
        return stats or {}
    except Exception as exc:
        logger.warning("dashboard_stats_failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Stats unavailable"
        ) from exc


@_API.get("/memory")
def get_memory() -> list[dict[str, Any]]:
    """Recent episodic memory entries."""
    try:
        from memory.episodic import EpisodicMemory

        return EpisodicMemory().get_all()
    except Exception:
        logger.warning("dashboard_memory_failed", exc_info=True)
        return []


@_API.get("/skills")
def get_skills() -> dict[str, Any]:
    """Top skill patterns per intent from SkillLibrary."""
    try:
        from core.types import Intent
        from memory.skill_library import SkillLibrary

        lib = SkillLibrary()
        return {intent.value: lib.get_top_skills(intent.value) for intent in Intent}
    except Exception:
        logger.warning("dashboard_skills_failed", exc_info=True)
        return {}


@_API.get("/briefs")
def get_briefs() -> dict[str, Any]:
    """Last 7 days of traces per expert."""
    try:
        from core.types import ExpertName
        from observability.traces import TraceStore

        store = TraceStore()
        return {name.value: store.get_expert_stats(name.value, days=7) for name in ExpertName}
    except Exception:
        logger.warning("dashboard_briefs_failed", exc_info=True)
        return {}


@_API.get("/jobs")
def get_jobs() -> list[dict[str, Any]]:
    """Most recent career expert traces (last 24 h)."""
    try:
        from observability.traces import TraceStore

        return TraceStore().get_expert_stats("career", days=1)
    except Exception:
        logger.warning("dashboard_jobs_failed", exc_info=True)
        return []


@router.get("/", response_class=HTMLResponse)
def dashboard_root() -> str:
    """Minimal HTMX dashboard that polls the API routes every 30 s.

    Requires Bearer token auth (same token as API routes).
    """
    return _HTML


_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sovereign Edge Dashboard</title>
  <script src="/static/htmx.min.js"></script>
  <link rel="stylesheet" href="/static/style.css">
  <script>
    // Inject Bearer token from localStorage into every HTMX request.
    // This avoids embedding the token in static HTML.
    document.addEventListener('htmx:configRequest', function(e) {
      var token = localStorage.getItem('se_token');
      if (token) { e.detail.headers['Authorization'] = 'Bearer ' + token; }
    });
    // Show login overlay when no token is stored.
    window.addEventListener('load', function() {
      if (!localStorage.getItem('se_token')) {
        document.getElementById('login-overlay').style.display = 'flex';
      }
    });
    function saveToken() {
      var t = document.getElementById('token-input').value.trim();
      if (t) { localStorage.setItem('se_token', t); location.reload(); }
    }
  </script>
</head>
<body>
  <!-- Login overlay — shown once on first visit until token is stored -->
  <div id="login-overlay" style="display:none; position:fixed; inset:0;
       background:rgba(0,0,0,.75); align-items:center; justify-content:center; z-index:100;">
    <div style="background:#1e1e2e; padding:2rem; border-radius:8px;
                text-align:center; min-width:280px;">
      <h2 style="margin-top:0">Sovereign Edge</h2>
      <p>Enter your dashboard token</p>
      <input id="token-input" type="password" placeholder="Bearer token"
             style="width:100%; margin:.5rem 0; padding:.5rem; box-sizing:border-box;"
             onkeydown="if(event.key==='Enter') saveToken()">
      <button onclick="saveToken()"
              style="width:100%; padding:.5rem; margin-top:.5rem; cursor:pointer;">
        Login
      </button>
    </div>
  </div>

  <header>
    <h1>&#128065;&#65039; Sovereign Edge</h1>
    <p id="refresh-note">Auto-refreshes every 30s</p>
  </header>

  <main>
    <section id="stats-section"
             hx-get="/api/v1/stats"
             hx-trigger="load, every 30s"
             hx-target="#stats-body">
      <h2>Today&#39;s Usage</h2>
      <div id="stats-body"><p>Loading&hellip;</p></div>
    </section>

    <section id="jobs-section"
             hx-get="/api/v1/jobs"
             hx-trigger="load, every 30s"
             hx-target="#jobs-body">
      <h2>Latest Career Scan</h2>
      <div id="jobs-body"><p>Loading&hellip;</p></div>
    </section>

    <section id="memory-section"
             hx-get="/api/v1/memory"
             hx-trigger="load, every 30s"
             hx-target="#memory-body">
      <h2>Recent Memory</h2>
      <div id="memory-body"><p>Loading&hellip;</p></div>
    </section>
  </main>
</body>
</html>
"""

# Attach sub-router so server.py can include a single object
router.include_router(_API)
