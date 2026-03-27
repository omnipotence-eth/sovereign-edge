#!/usr/bin/env bash
# entrypoint.sh — Container entry point for Sovereign Edge
#
# 1. Starts the health/readiness server (port 8080) in the background.
# 2. Execs the Telegram bot as the main process (PID 1 replacement via exec).
#
# The health server is a best-effort sidecar: if it fails to start the bot
# still launches. Docker's HEALTHCHECK will report unhealthy in that case,
# alerting the operator without killing the bot mid-session.

set -euo pipefail

echo "[entrypoint] Starting Sovereign Edge health server on :8080..."
python -m uvicorn health.server:app \
    --host 0.0.0.0 \
    --port 8080 \
    --log-level warning \
    --no-access-log &

HEALTH_PID=$!
echo "[entrypoint] Health server PID: ${HEALTH_PID}"

# Brief pause to let uvicorn bind the port before healthcheck fires
sleep 1

echo "[entrypoint] Starting Telegram bot..."
exec python -m telegram_bot.bot
