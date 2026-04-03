# Sovereign Edge Personal Intelligence System

Always-on, privacy-first multi-agent AI running on a Jetson Orin Nano 8GB (~$2–3/month electricity). Classifies user intent in under 10ms via an ONNX-quantized router, dispatches to specialized domain squads, and delegates generation to free-tier cloud LLMs through a unified LiteLLM gateway.

Built to explore edge inference, agentic orchestration, and cost-conscious LLM deployment on constrained hardware.

> **Status:** Phase 1 in progress — monorepo structure, tooling, and Jetson bootstrap are being established.

## Architecture

```
JETSON ORIN NANO 8GB (~5.1 GB peak / 2.9 GB headroom)

  ┌─────────────────────────────────────────────────────┐
  │  ONNX Intent Router (DistilBERT INT8, <10ms)        │
  │  4-class classifier → SPIRITUAL | CAREER |          │
  │                        INTELLIGENCE | CREATIVE      │
  └──────────────────────┬──────────────────────────────┘
                         │ intent + confidence
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  LangGraph StateGraph Orchestrator                  │
  │                                                     │
  │  State: { messages, intent, memory_context,         │
  │           squad_result, hitl_required, schedule }   │
  │                                                     │
  │  router_node ──(conditional edges)──▶ squad_node    │
  │       │                                   │         │
  │  memory_node ◀────────────────────────────┘         │
  │       │                                             │
  │  hitl_node  (interrupt — Telegram approval gate)    │
  │       │                                             │
  │  delivery_node ──▶ Telegram / Voice                 │
  └─────────────────────────────────────────────────────┘
         │             │             │             │
    Spiritual      Career       Intelligence   Creative
    Squad          Squad        Squad          Squad
    (Bible RAG)    (JobSpy +    (Alpha         (Manim /
                   Jina)        Vantage /       FFmpeg /
                                arXiv)          Kokoro)
         │
  Shared: LanceDB · Mem0 · structlog · SQLite · APScheduler

         │ (cloud generation via LiteLLM — all free tiers)
  Groq → Gemini Flash → Cerebras → Mistral  (~8–11M tokens/day)
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| LangGraph orchestration | Stateful `StateGraph` provides persistent multi-turn context, built-in checkpointing across sessions, and a first-class human-in-the-loop interrupt pattern — maps directly to SOUL.md's HITL approval gates. Preferred over LangChain chains or ad-hoc orchestration for production agents. |
| ONNX INT8 router | DistilBERT quantized to INT8 keeps intent classification at <10ms without consuming GPU/VRAM, leaving full headroom for generation and embedding models |
| LiteLLM library | Single interface across 4 free-tier LLM providers with automatic failover on rate limits — no vendor lock-in |
| LanceDB | Embedded vector DB: no server process, persists to disk, ~300 MB footprint vs. a Chroma/Qdrant server |
| SOPS + age encryption | API secrets committed to the repo encrypted — safe for public GitHub without `.env` juggling across devices |
| systemd target | `ai-stack.target` ensures all agents start on boot and restart on crash — no cron, no screen sessions, no manual intervention |
| uv | 10–100x faster than pip for dependency installs; lockfile committed for reproducible Jetson builds |

## Agent Squads

| Squad | Tools | Purpose |
|-------|-------|---------|
| **Spiritual** | LanceDB Bible RAG, scripture index | Biblical Q&A, verse retrieval, devotional queries |
| **Career** | JobSpy scraper, resume templates | Job search automation, listing aggregation and filtering |
| **Intelligence** | Alpha Vantage, arXiv API | Market data snapshots, research paper summaries |
| **Creative** | Manim, FFmpeg | Animated video generation, media processing pipelines |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Python 3.11, Jetson JetPack 6 |
| Agent orchestration | LangGraph (`StateGraph` + checkpointing + HITL interrupts) |
| Inference gateway | LiteLLM (Groq, Gemini Flash, Cerebras, Mistral) |
| Intent routing | ONNX Runtime + DistilBERT INT8 |
| Vector DB | LanceDB |
| Memory | Mem0 |
| Scheduling | APScheduler |
| Delivery | Telegram Bot API |
| Observability | structlog (structured JSON) + OpenTelemetry |
| Secrets management | SOPS + age |
| Package manager | uv |
| Task runner | Taskfile |

## Phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Jetson foundation + monorepo structure + CI | In Progress |
| 2 | ONNX router + LangGraph StateGraph + LiteLLM gateway + Mem0 memory | Planned |
| 3 | Agent squads + Telegram delivery + HITL gates | Planned |
| 4 | Voice interface (OpenWakeWord + whisper.cpp) | Planned |

## Quick Start (on Jetson Orin Nano)

**Prerequisites:** Jetson Orin Nano with JetPack 6 flashed, SSH access, `uv` installed.

```bash
# 1. Bootstrap OS-level dependencies
bash scripts/setup-jetson.sh      # Docker, uv, ONNX runtime, Ollama for embeddings

# 2. Clone and sync Python environment
git clone <repo> ~/sovereign-edge && cd ~/sovereign-edge
uv sync
pre-commit install

# 3. Configure encrypted secrets
age-keygen -o ~/.config/sops/age/keys.txt
# Paste the public key into secrets/.sops.yaml
sops -e -i secrets/env.enc.yaml   # then add real API keys inside

# 4. Install and enable systemd services
sudo cp systemd/*.service systemd/*.target /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-stack.target

# 5. Start
task start
```

## Task Runner

```bash
task start          # Start all services via systemd
task stop           # Stop all services
task status         # Show service status
task monitor        # Live resource monitor (jtop)
task logs           # Tail all service logs
task lint           # Ruff lint + format check
task test           # Run test suite
task eval           # Run evaluation harness
task train-router   # Fine-tune ONNX intent router
task secrets-edit   # Edit encrypted secrets with SOPS
```

## Memory Budget

| Component | RAM |
|-----------|-----|
| JetPack 6 OS (headless) | ~1.2 GB |
| LangGraph orchestrator + Python runtime | ~150–200 MB |
| ONNX Router (DistilBERT INT8) | ~200 MB |
| LiteLLM library | ~75 MB |
| Ollama (nomic-embed-text only) | ~400 MB |
| LanceDB + Mem0 | ~300 MB |
| **Baseline** | **~2.3–2.4 GB** |
| **Peak (content tools active)** | **~4.5–5.1 GB** |

## License

MIT
