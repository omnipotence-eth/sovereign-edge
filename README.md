# Sovereign Edge Personal Intelligence System

Always-on, multi-agent AI running on a Jetson Orin Nano 8GB. $0 upfront, ~$2-3/month electricity.

## Architecture

```
JETSON ORIN NANO 8GB (~5.2 GB peak / 2.8 GB headroom)

  ONNX Router (DistilBERT INT8, <10ms) → 4-class intent classifier
       │
  NemoClaw/OpenClaw Orchestrator
       │
  ┌────┴────────────────────────┐
  │                             │
  Spiritual Squad    Career Squad    Intelligence Squad    Creative Squad
  (Bible RAG)        (JobSpy)        (Alpha Vantage/arXiv) (Manim/FFmpeg)
       │
  Shared: LanceDB · Mem0 · structlog · SQLite · APScheduler

       │ (cloud, via LiteLLM library)
  Groq → Gemini Flash → Cerebras → Mistral  (~8-11M tokens/day free)
```

## Phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Jetson foundation + monorepo | In Progress |
| 2 | ONNX router + LiteLLM gateway + memory | Planned |
| 3 | Squads + Telegram delivery | Planned |
| 4 | Voice interface (OpenWakeWord + whisper.cpp) | Planned |

## Quick Start (on Jetson)

```bash
# 1. Flash JetPack 6 — see scripts/setup-jetson.sh
# 2. Clone and sync
git clone <repo> ~/sovereign-edge && cd ~/sovereign-edge
uv sync
pre-commit install

# 3. Configure secrets
age-keygen -o ~/.config/sops/age/keys.txt
# Edit secrets/.sops.yaml with your public key
cp secrets/env.example.yaml secrets/env.enc.yaml
sops -e -i secrets/env.enc.yaml  # then edit to add real keys

# 4. Install systemd services
sudo cp systemd/*.service systemd/*.target /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-stack.target
task start
```

## Task Runner

```bash
task start       # Start all services
task stop        # Stop all services
task status      # Show service status
task monitor     # Live jtop resource monitor
task logs        # Tail all service logs
task lint        # Ruff lint + format
task test        # Run all tests
task eval        # Run eval harness
task train-router  # Fine-tune ONNX router
task secrets-edit  # Edit encrypted secrets
```

## RAM Budget

| Component | RAM |
|-----------|-----|
| JetPack 6 OS (headless) | ~1.2 GB |
| NemoClaw + OpenClaw | ~300-500 MB |
| ONNX Router (DistilBERT INT8) | ~200 MB |
| LiteLLM (library) | ~75 MB |
| Ollama (nomic-embed-text only) | ~400 MB |
| LanceDB + Mem0 | ~300 MB |
| **Baseline total** | **~2.5-2.7 GB** |
| **Peak (content tools active)** | **~4.5-5.2 GB** |
