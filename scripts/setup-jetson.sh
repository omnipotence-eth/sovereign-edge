#!/usr/bin/env bash
# setup-jetson.sh — Phase 1: Jetson Orin Nano Foundation
# Run this on the Jetson after flashing JetPack 6 to NVMe.
# Assumes NVMe is mounted at /ssd and JetPack 6.2+ is installed.
set -euo pipefail

echo "=== Phase 1.2: Disable GUI, ZRAM, configure swap ==="

# Disable desktop environment (saves ~800 MB RAM)
sudo systemctl set-default multi-user.target

# Disable ZRAM (frees ~500 MB effective RAM at cost of swap)
sudo systemctl disable nvzramconfig

# Create 16 GB NVMe swap
sudo fallocate -l 16G /ssd/16GB.swap
sudo chmod 600 /ssd/16GB.swap
sudo mkswap /ssd/16GB.swap
sudo swapon /ssd/16GB.swap

# Persist swap across reboots
grep -q '/ssd/16GB.swap' /etc/fstab || \
    echo "/ssd/16GB.swap none swap sw 0 0" | sudo tee -a /etc/fstab

# Set 15W power mode (stable for stock cooling; use 0 for MAXN if actively cooled)
sudo nvpmodel -m 1
sudo jetson_clocks

echo "Swap configured. Rebooting in 5 seconds (Ctrl+C to cancel)..."
sleep 5
sudo reboot

# After reboot, resume from the next section:
echo ""
echo "=== Phase 1.3: Install core dependencies ==="

sudo apt update && sudo apt install -y \
    python3.11 python3.11-venv python3.11-dev \
    ffmpeg libcairo2-dev libpango1.0-dev \
    sqlite3 libsqlite3-dev \
    git curl build-essential

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Install Ollama (ARM64 build)
curl -fsSL https://ollama.com/install.sh | sh

# Redirect Ollama models to NVMe SSD (avoids eMMC writes)
sudo mkdir -p /ssd/ollama
grep -q 'OLLAMA_MODELS' /etc/environment || \
    echo 'OLLAMA_MODELS=/ssd/ollama' | sudo tee -a /etc/environment
grep -q 'OLLAMA_HOST' /etc/environment || \
    echo 'OLLAMA_HOST=127.0.0.1:11434' | sudo tee -a /etc/environment

# Pull embedding model (no LLM — cloud-first strategy)
source /etc/environment
ollama pull nomic-embed-text

echo "=== Install SOPS + age for secrets management ==="
SOPS_VERSION=$(curl -s https://api.github.com/repos/getsops/sops/releases/latest | grep '"tag_name"' | cut -d'"' -f4)
curl -LO "https://github.com/getsops/sops/releases/download/${SOPS_VERSION}/sops-${SOPS_VERSION}.linux.arm64"
sudo mv "sops-${SOPS_VERSION}.linux.arm64" /usr/local/bin/sops
sudo chmod +x /usr/local/bin/sops
sudo apt install -y age

echo "=== Install Tailscale ==="
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

echo ""
echo "=== Phase 1.4: Install NemoClaw / OpenClaw ==="
echo "NOTE: NemoClaw is <2 weeks old (as of March 2026). Check availability first."
echo ""
echo "Option A — NemoClaw (preferred if Jetson ARM64 is supported):"
echo "  curl -fsSL https://install.nemoclaw.nvidia.com | sh"
echo ""
echo "Option B — Pydantic AI fallback (pure Python, ~0 MB overhead):"
echo "  uv pip install pydantic-ai"
echo ""
echo "After installing, check RAM usage with: jtop"
echo "If NemoClaw + OpenShell idle > 800 MB → use Pydantic AI fallback."
echo ""
echo "=== Phase 1.5: Initialize monorepo ==="
echo "Clone the repo to ~/sovereign-edge and run: uv sync && pre-commit install"

echo ""
echo "Verify baseline:"
free -h
echo ""
echo "Target: ~2.5-2.7 GB used with all services idle."
