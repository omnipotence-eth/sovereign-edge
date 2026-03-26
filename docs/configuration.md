# Configuration

All configuration is loaded from environment variables via Pydantic `BaseSettings`. The prefix for every variable is `SE_`.

On a local dev machine, place variables in a `.env` file at the project root. On the Jetson, the SOPS-encrypted `secrets/env.yaml` is decrypted at service startup and written to `secrets/.env.decrypted`.

---

## Required Variables

The service will not function correctly without these.

| Variable | Description |
|---|---|
| `SE_TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `SE_TELEGRAM_OWNER_CHAT_ID` | Your personal Telegram chat ID — only this ID can send messages to the bot |

At least one cloud LLM key is required for cloud routing. All four enables the full fallback chain.

| Variable | Provider | Free Tier |
|---|---|---|
| `SE_GROQ_API_KEY` | Groq (Priority 1) | 500K tokens/day, 30 RPM |
| `SE_GOOGLE_API_KEY` | Gemini (Priority 2) | 250K tokens/day, 15 RPM |
| `SE_CEREBRAS_API_KEY` | Cerebras (Priority 3) | 1M tokens/day, 30 RPM |
| `SE_MISTRAL_API_KEY` | Mistral (Priority 4) | 33M tokens/day, 2 RPM |

---

## Ollama (Local Inference)

| Variable | Default | Description |
|---|---|---|
| `SE_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `SE_OLLAMA_MODEL` | `qwen3:0.6b` | Model for chat completions (local fallback) |
| `SE_OLLAMA_EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Model for intent routing embeddings |

Ollama is the Priority 5 fallback — it is always available and has no rate limit. It is also the only provider used when routing is forced `LOCAL` (PII detected).

---

## Storage Paths

| Variable | Default | Description |
|---|---|---|
| `SE_PROJECT_ROOT` | Auto-detected | Root of the sovereign-edge repo |
| `SE_SSD_ROOT` | `~/sovereign-edge-data` | Base directory for persistent data |
| `SE_LANCEDB_PATH` | `{ssd_root}/lancedb` | Semantic cache vector store |
| `SE_LOGS_PATH` | `{ssd_root}/logs` | Structured log output |
| `SE_MODELS_PATH` | `{ssd_root}/models` | ONNX router model and other artifacts |

On Jetson, set `SE_SSD_ROOT` to a path on the NVMe SSD (e.g., `/mnt/ssd/sovereign-edge-data`) to avoid wearing out the eMMC.

---

## Rate Limits

These control the per-provider token bucket. Override only if your actual API tier differs from the defaults.

| Variable | Default | Description |
|---|---|---|
| `SE_GROQ_RPM` | `30` | Groq requests per minute |
| `SE_GROQ_DAILY_TOKENS` | `500000` | Groq daily token cap |
| `SE_GEMINI_RPM` | `15` | Gemini RPM |
| `SE_GEMINI_DAILY_TOKENS` | `250000` | Gemini daily token cap |
| `SE_CEREBRAS_RPM` | `30` | Cerebras RPM |
| `SE_CEREBRAS_DAILY_TOKENS` | `1000000` | Cerebras daily token cap |
| `SE_MISTRAL_RPM` | `2` | Mistral RPM |
| `SE_MISTRAL_DAILY_TOKENS` | `33000000` | Mistral daily token cap |

---

## Feature Flags

| Variable | Default | Description |
|---|---|---|
| `SE_ENABLE_SEMANTIC_CACHE` | `true` | LanceDB semantic cache (cosine ≥ 0.92) |
| `SE_ENABLE_EPISODIC_MEMORY` | `false` | Mem0 long-term memory (requires `mem0ai` dependency) |
| `SE_ENABLE_ONNX_ROUTER` | `true` | ONNX DistilBERT Tier 2 classifier (requires model file) |

---

## Scheduling

| Variable | Default | Description |
|---|---|---|
| `SE_MORNING_WAKE_HOUR` | `5` | Hour (24h, Central Time) for the morning pipeline start |
| `SE_TIMEZONE` | `US/Central` | APScheduler timezone |

The full morning pipeline schedule is fixed at specific times. Changing `SE_MORNING_WAKE_HOUR` shifts the health check; the other briefs run at offsets from that hour.

---

## Input Limits

| Variable | Default | Description |
|---|---|---|
| `SE_MAX_MESSAGE_LENGTH` | `2000` | Characters — messages are truncated before processing |
| `SE_RATE_LIMIT_SECONDS` | `2` | Minimum seconds between requests per chat_id |
| `SE_CONVERSATION_HISTORY_TURNS` | `40` | Max turns stored per chat_id in SQLite |
| `SE_CONTEXT_TURNS_INJECTED` | `8` | Recent turns injected into each LLM request |
| `SE_CACHE_TTL_HOURS` | `24` | Semantic cache entry expiry |

---

## Example `.env`

```bash
# Telegram
SE_TELEGRAM_BOT_TOKEN=your_bot_token_here
SE_TELEGRAM_OWNER_CHAT_ID=your_chat_id_here

# Cloud LLM keys (all free tier)
SE_GROQ_API_KEY=gsk_...
SE_GOOGLE_API_KEY=AIza...
SE_CEREBRAS_API_KEY=csk_...
SE_MISTRAL_API_KEY=...

# Storage (Jetson — use SSD path)
SE_SSD_ROOT=/mnt/ssd/sovereign-edge-data

# Ollama (running locally on Jetson)
SE_OLLAMA_BASE_URL=http://localhost:11434
```
