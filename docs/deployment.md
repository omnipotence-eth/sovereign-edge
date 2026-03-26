# Deployment

Sovereign Edge runs as a systemd service on a Jetson Nano (or any Linux ARM/x86 host with Ollama). This document covers the full setup from a fresh board to a running service.

---

## Prerequisites

- Jetson Nano with JetPack 6 (Ubuntu 22.04 base) or any Linux host
- Python 3.11+
- `uv` package manager
- Ollama installed and running
- SOPS + Age for secrets management
- `rsync` on the development machine (for deployment)

---

## First-Time Jetson Setup

### 1. Install system dependencies

```bash
sudo apt update && sudo apt install -y \
    python3.11 python3.11-venv \
    curl git rsync

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
```

### 3. Pull the required models

```bash
# Chat model (local fallback)
ollama pull qwen3:0.6b

# Embedding model (intent routing Tier 1)
ollama pull qwen3-embedding:0.6b
```

### 4. Clone the repository

```bash
git clone https://github.com/your-username/sovereign-edge.git /home/omnipotence/sovereign-edge
cd /home/omnipotence/sovereign-edge
```

### 5. Install Python dependencies

```bash
uv sync --all-packages
```

### 6. Create the data directory on the SSD

```bash
mkdir -p /mnt/ssd/sovereign-edge-data/{lancedb,logs,models}
```

---

## Secrets Management

Sovereign Edge uses SOPS with Age encryption to store API keys. Keys are never stored in plaintext in the repository.

### Install SOPS and Age

```bash
# Age
sudo apt install age

# SOPS
curl -LO https://github.com/getsops/sops/releases/latest/download/sops-v3.x.x.linux.arm64
sudo install sops-v3.x.x.linux.arm64 /usr/local/bin/sops
```

### Generate an Age key pair

```bash
age-keygen -o ~/.config/sops/age/keys.txt
```

Copy the public key from the output (starts with `age1...`).

### Create the encrypted secrets file

On your development machine, create `secrets/env.yaml`:

```yaml
SE_TELEGRAM_BOT_TOKEN: your_bot_token
SE_TELEGRAM_OWNER_CHAT_ID: "your_chat_id"
SE_GROQ_API_KEY: gsk_...
SE_GOOGLE_API_KEY: AIza...
SE_CEREBRAS_API_KEY: csk_...
SE_MISTRAL_API_KEY: ...
SE_SSD_ROOT: /mnt/ssd/sovereign-edge-data
```

Encrypt it with the Jetson's public key:

```bash
sops --encrypt --age age1... secrets/env.yaml > secrets/env.yaml.enc
mv secrets/env.yaml.enc secrets/env.yaml
```

The encrypted `secrets/env.yaml` is safe to commit. The plaintext version is not.

### `.sops.yaml` configuration

Create `.sops.yaml` at the project root to configure key discovery:

```yaml
creation_rules:
  - path_regex: secrets/.*\.yaml$
    age: age1...your_public_key...
```

---

## systemd Service

The service file is at `systemd/telegram-bot.service`.

### Install the service

```bash
sudo cp systemd/telegram-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable telegram-bot
```

### Service configuration highlights

```ini
[Service]
User=omnipotence
WorkingDirectory=/home/omnipotence/sovereign-edge

# Decrypt secrets before start — fails hard if SOPS errors
ExecStartPre=/bin/bash -c 'set -euo pipefail; \
    sops --decrypt /home/omnipotence/sovereign-edge/secrets/env.yaml \
    | sed "s/: /=/" \
    > /home/omnipotence/sovereign-edge/secrets/.env.decrypted \
    && chmod 600 /home/omnipotence/sovereign-edge/secrets/.env.decrypted'

ExecStart=uv run python -m telegram_bot
EnvironmentFile=/home/omnipotence/sovereign-edge/secrets/.env.decrypted

# Resource limits
MemoryLimit=2G
CPUQuota=80%

# Hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict

# Fast restart
TimeoutStopSec=15
Restart=on-failure
RestartSec=10
```

The `set -euo pipefail` in `ExecStartPre` is critical: if `sops` fails (wrong key, missing file, corrupted ciphertext), the shell exits non-zero and systemd aborts the start rather than launching the service with no secrets.

### Service management

```bash
# Start / stop / restart
sudo systemctl start telegram-bot
sudo systemctl stop telegram-bot
sudo systemctl restart telegram-bot

# View status
sudo systemctl status telegram-bot

# Follow logs
journalctl -u telegram-bot -f

# View today's logs
journalctl -u telegram-bot --since today
```

---

## Deployment from Development Machine

The project includes a `Taskfile.yml` that automates deploy + restart over SSH.

```bash
# Deploy and restart the service on Jetson
task deploy
```

This runs `rsync` to sync the project directory to the Jetson (excluding `.git`, `__pycache__`, and secrets), then SSHs in to restart the service.

Configure the Jetson hostname/IP in `Taskfile.yml`:

```yaml
vars:
  JETSON_HOST: jetson.local   # or IP address
  JETSON_USER: omnipotence
  JETSON_PATH: /home/omnipotence/sovereign-edge
```

---

## Verifying the Deployment

1. Send `/start` to your bot in Telegram — you should receive the welcome message.
2. Send `/stats` to confirm the trace store is recording.
3. Send a test message ("What's a good paper on LLM inference?") — verify the intelligence squad responds with a properly formatted reply.
4. Check logs for structured JSON output:

```bash
journalctl -u telegram-bot -f | python3 -m json.tool
```

5. At 05:15 Central the next morning, verify the spiritual brief arrives.

---

## ONNX Router Model (Optional)

The Tier 2 ONNX DistilBERT classifier requires a fine-tuned model file. Without it, the router falls through to Tier 3 (keyword matching) automatically.

If you have the model:

```bash
mkdir -p data/models
cp router.onnx data/models/
```

The model path is `{SE_MODELS_PATH}/router.onnx`. The HuggingFace tokenizer is fetched at startup and pinned to a specific commit hash for supply chain security.
