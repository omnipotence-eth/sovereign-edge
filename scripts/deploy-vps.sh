#!/usr/bin/env bash
# deploy-vps.sh — Bootstrap Sovereign Edge on a fresh VPS
#
# Run as the non-root user that will own the deployment, e.g.:
#   bash scripts/deploy-vps.sh
#
# The script is idempotent: safe to re-run for updates.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
ENV_FILE="${REPO_ROOT}/.env"
ENV_EXAMPLE="${REPO_ROOT}/.env.example"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'
YLW='\033[1;33m'
GRN='\033[0;32m'
NC='\033[0m'

info()  { echo -e "${GRN}[deploy]${NC} $*"; }
warn()  { echo -e "${YLW}[warn]${NC}  $*"; }
die()   { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── 1. Preflight checks ───────────────────────────────────────────────────────
info "Checking prerequisites..."

command -v docker >/dev/null 2>&1 \
    || die "docker is not installed. Install Docker Engine: https://docs.docker.com/engine/install/"

# Accept both 'docker compose' (v2 plugin) and legacy 'docker-compose'
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
else
    die "docker compose (v2 plugin) or docker-compose is required."
fi

info "Docker:  $(docker --version)"
info "Compose: $($COMPOSE_CMD version)"

# ── 2. Data directory ─────────────────────────────────────────────────────────
info "Creating data directory at ${DATA_DIR}..."
mkdir -p \
    "${DATA_DIR}/lancedb" \
    "${DATA_DIR}/logs" \
    "${DATA_DIR}/models" \
    "${DATA_DIR}/ollama"

# Container runs as uid 1001; make data writable
chmod -R 755 "${DATA_DIR}"
# If running as root (not recommended), ensure uid 1001 can write
chown -R 1001:1001 "${DATA_DIR}" 2>/dev/null || warn "Could not chown ${DATA_DIR} — container writes may fail if not uid 1001."

info "Data directory ready."

# ── 3. .env setup ─────────────────────────────────────────────────────────────
if [[ ! -f "${ENV_FILE}" ]]; then
    if [[ -f "${ENV_EXAMPLE}" ]]; then
        cp "${ENV_EXAMPLE}" "${ENV_FILE}"
        warn ".env created from .env.example — EDIT IT NOW before starting the stack:"
        warn "  nano ${ENV_FILE}"
        warn "  (set at minimum: SE_TELEGRAM_BOT_TOKEN and SE_TELEGRAM_OWNER_CHAT_ID)"
        echo ""
        read -r -p "Press ENTER once you have filled in .env, or Ctrl-C to abort..." _
    else
        die ".env.example not found. Cannot create .env template."
    fi
else
    info ".env already exists — skipping template copy."
fi

# Quick sanity check: fail loudly if the two required keys are still blank
source <(grep -E "^SE_TELEGRAM_BOT_TOKEN=|^SE_TELEGRAM_OWNER_CHAT_ID=" "${ENV_FILE}" || true)
if [[ -z "${SE_TELEGRAM_BOT_TOKEN:-}" ]]; then
    die "SE_TELEGRAM_BOT_TOKEN is not set in .env. Aborting."
fi
if [[ -z "${SE_TELEGRAM_OWNER_CHAT_ID:-}" ]]; then
    die "SE_TELEGRAM_OWNER_CHAT_ID is not set in .env. Aborting."
fi

# ── 4. Pull / build images ────────────────────────────────────────────────────
info "Building sovereign-edge image (this may take a few minutes on first run)..."
cd "${REPO_ROOT}"
$COMPOSE_CMD build sovereign-edge

# ── 5. Start the stack ────────────────────────────────────────────────────────
info "Starting stack in detached mode..."
$COMPOSE_CMD up -d

# Wait briefly then show status
sleep 3
info "Container status:"
$COMPOSE_CMD ps

# ── 6. Tail logs ──────────────────────────────────────────────────────────────
info "Tailing sovereign-edge logs (Ctrl-C to stop tailing — stack keeps running):"
$COMPOSE_CMD logs -f sovereign-edge
