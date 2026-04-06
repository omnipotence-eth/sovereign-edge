# Experts

Each expert is a specialized AI agent implemented as a compiled LangGraph `StateGraph` subgraph. Requests flow through a multi-node pipeline — fetching live data, ranking or searching, then synthesizing a response — before passing to the shared LLM gateway. All experts write to the same trace store and fall back gracefully to a direct LLM call when LangGraph is unavailable.

---

## Intelligence Expert

**Intent:** `INTELLIGENCE`

Synthesizes AI/ML research into actionable briefs. Fetches live papers from arXiv and HuggingFace Daily Papers so every response reflects what published this week, not what was in the training data.

### Live Data Sources

| Source | Endpoint | What it fetches |
|---|---|---|
| arXiv | Atom API (free, no auth) | Latest papers from cs.LG, cs.AI, cs.CL, cs.CV |
| HuggingFace Daily Papers | HF Hub API (free) | Trending community-curated papers |

Papers are de-duplicated across calls via a daily novelty filter — the same paper won't appear in consecutive requests on the same day.

### Capabilities

- Research synthesis: summarize and connect ideas across multiple papers
- Trend spotting: identify emerging methods and architectural patterns
- Concept explanation: break down novel techniques with concrete examples
- Literature context: situate a paper within the broader research landscape

### Pipeline

```
arxiv_fetcher ──┐
                ├──► ranker (FlashRank cross-encoder) ──► synthesizer
hf_fetcher    ──┘
```

`arxiv_fetcher` and `hf_fetcher` run in the same LangGraph superstep (true parallel execution). Both write to a shared `raw_papers` list via `operator.add` merge. `ranker` waits for both before scoring, then `synthesizer` produces the final response.

### Morning Brief

Runs at 05:30 (configurable via `SE_MORNING_WAKE_HOUR` / `SE_TIMEZONE`). Fetches the latest papers from both sources in parallel, builds a digest of the top findings, and pushes it to Telegram. Scoped to 300 tokens to keep it scannable.

### Response Format

Each paper entry:

```
*Paper Title* — _venue/source_
One-sentence summary of the contribution.
Why it matters for practitioners.
[Read more](https://link)
```

---

## Career Expert

**Intent:** `CAREER`

An ML/AI career strategist with live access to job listings. Every response is grounded with real-time search results from Jina so job data, salary ranges, and company intel are current.

Personalized via environment variables — see [Configuration](configuration.md#career-personalization).

### Live Data Sources

| Source | What it fetches |
|---|---|
| Jina Search | Live job listings for target roles in target location |
| Jina Search | Company news, salary data, market context |

### Capabilities

- Job search: find and summarize open roles matching `SE_CAREER_TARGET_ROLES`
- Resume coaching: tailor language and keywords to a specific JD
- Interview prep: behavioral and technical question practice
- Market intelligence: hiring trends and salary ranges for target location
- Offer evaluation: total compensation analysis

### Pipeline

```
job_searcher (Jina live search) ──► strategist (LLM synthesis)
```

### Morning Brief

Runs at 06:00 and 18:00 (configurable via `SE_MORNING_WAKE_HOUR` / `SE_TIMEZONE`). Searches for new openings in the target location and roles, extracts the highest-value listing, and provides one concrete action to take that day. Scoped to 250 tokens.

---

## Creative Expert

**Intent:** `CREATIVE`

A content strategist and creative director with live trend awareness. Grounds creative output in what is actually working right now in the creator and content economy, not generic advice.

### Live Data Sources

| Source | What it fetches |
|---|---|
| Jina Search | Current content trends, platform conventions, creator economy news |
| Jina Search | Examples of high-performing content in the requested format |

### Capabilities

- Long-form writing: blog posts, essays, articles with a defined voice
- Social media: LinkedIn, Twitter/X, and short-form content tailored to format
- Content strategy: editorial calendar, pillar content, repurposing frameworks
- Brand voice development: articulate and document tone, vocabulary, persona
- Storytelling: narrative structure, hooks, and audience-specific framing

### Pipeline

```
trend_researcher (Jina live search) ──► writer (LLM generation)
```

### Morning Brief

Runs at 07:00 (configurable via `SE_MORNING_WAKE_HOUR` / `SE_TIMEZONE`). Searches for AI content creation and LinkedIn strategy trends, then generates one micro-challenge: a specific, actionable creative exercise completable in 15–20 minutes. Scoped to 150 tokens.

---

## Spiritual Expert

**Intent:** `SPIRITUAL`

A contemplative guide rooted in Christian faith. Every response involving scripture is grounded with live Bible verse retrieval so quotations are always accurate and properly cited.

### Live Data Sources

| Source | Endpoint | What it fetches |
|---|---|---|
| bible-api.com | Free, no auth required | KJV verse by reference or random |

Supports KJV, WEB, YLT, DARBY, ASV, and BBE translations. KJV is the default.

### Capabilities

- Scripture study: exegesis, cross-references, historical context
- Prayer composition: guided or impromptu prayer for specific situations
- Devotionals: short daily reflections anchored to scripture
- Theological questions: Christian doctrine, church history, apologetics
- Faith application: connecting scripture to daily life and decision-making

### Pipeline

```
scripture_fetcher (Bible API) ──► theologian (LLM devotional)
```

### Scripture Handling

When the user's message contains a scripture reference (e.g., "John 3:16" or "Psalm 23"), the expert extracts the reference via regex and fetches the exact verse before responding. If no reference is given, it fetches a random verse to anchor the response.

Scripture is always quoted in italics with full citation:
`_"For God so loved the world..."_ — John 3:16 KJV`

### Morning Brief

Runs at 05:15 (configurable via `SE_MORNING_WAKE_HOUR` / `SE_TIMEZONE`). Fetches a random verse, writes a brief morning devotional (verse + 2–3 sentences of reflection + one-sentence prayer), and pushes it to Telegram. Under 120 words.

---

## Goals Expert

**Intent:** `GOALS`

A personal accountability coach backed by a persistent SQLite goal store. Tracks goals with deadlines and progress percentages, surfaces the most urgent ones each morning, and generates one concrete action item per day.

### Data Sources

All data is local — no external API calls. Goals are stored in `SE_GOALS_DB_PATH` (default `data/goals.db`) using WAL-mode SQLite with a threading lock for safe concurrent access.

### Capabilities

- Goal creation: add a titled goal with optional description and target date
- Progress tracking: update completion percentage (0–100, clamped)
- Status management: mark goals as `active`, `paused`, or `complete`
- Urgent retrieval: surface the top 3 goals sorted by proximity to target date
- Daily coaching: provide one specific action step toward the most urgent goal

### Morning Brief

Runs at 07:30. Fetches the top 3 active goals sorted by urgency (closest `target_date`), formats a check-in with completion percentages, and appends an LLM-generated action item for the day. Silently skips delivery when no active goals exist.

### Response Format

```
Goals for today:
1. Land ML Engineer role (45% complete, due 2026-06-01)
2. Finish Sovereign Edge v1.0 (70% complete, due 2026-05-01)

Action: Apply to one Capital One ML posting today and tailor the resume summary.
```

---

## PII Routing

All experts respect the routing decision set by the PII detector. If the user's message contains SSN, credit card, email, phone, or IP address patterns, routing is forced to `LOCAL` and no external data sources (Jina, arXiv, HuggingFace, Bible API) are called. The response comes from the local Ollama model only.
