# AGENTS.md — Squad Definitions and Capabilities

---

## Orchestrator
**Runtime:** LangGraph `StateGraph`
**Responsibility:** Receive ONNX router output, dispatch to specialist squad nodes, persist session state via checkpointing, enforce HITL approval gates, and deliver aggregated responses via Telegram.

### Graph State
```python
class SovereignState(TypedDict):
    messages: list[BaseMessage]       # full conversation history
    intent: str                       # ONNX router classification
    intent_confidence: float          # router confidence score
    memory_context: str               # Mem0 retrieved context
    squad_result: str                 # output from active squad node
    hitl_required: bool               # True = pause for Telegram approval
    hitl_approved: bool | None        # None = pending, True/False = resolved
    schedule_trigger: str | None      # APScheduler job ID if proactive
```

### Graph Structure
```
router_node
    │  (conditional: route by intent)
    ├──▶ spiritual_node
    ├──▶ career_node
    ├──▶ intelligence_node
    └──▶ creative_node
              │
         memory_node  (write result to Mem0 + LanceDB)
              │
         hitl_node    (interrupt if hitl_required=True)
              │
         delivery_node (Telegram send)
```

### HITL Pattern
LangGraph's `interrupt()` mechanism pauses graph execution at `hitl_node` and resumes only after the user sends an approval reply in Telegram. This implements the approval gates defined in SOUL.md with no polling loop.

---

## Spiritual Squad
**Purpose:** Bible study, devotionals, prayer, theology, pastoral support
**Data sources:**
- STEPBible datasets (CC BY 4.0) — stored in `data/stepbible/`
- Treasury of Scripture Knowledge (TSK) — stored in `data/tsk/`
- AO Lab Bible tools
**Capabilities:**
- Verse lookup and cross-reference (TSK)
- Thematic search via LanceDB RAG
- Devotional generation
- Theological Q&A grounded in Scripture

**Routing trigger:** `IntentRouter` class = `SPIRITUAL` (confidence > 0.7)

---

## Career Squad
**Purpose:** Job search, resume tailoring, interview prep, salary research
**Tools:**
- JobSpy (multi-board job scraping)
- Jina reranker (relevance scoring)
- python-docx / LaTeX (document generation)
**Capabilities:**
- Daily job board scan (LinkedIn, Indeed, Glassdoor, RemoteOK)
- Resume tailoring to specific JD
- Cover letter generation
- Interview question bank by role

**Target roles:** ML Engineer, AI Engineer, Backend Python Engineer
**Resume location:** `~/Documents/Job Search/`

**Routing trigger:** `IntentRouter` class = `CAREER`

---

## Intelligence Squad
**Purpose:** Markets, AI research, news, trends
**Data sources:**
- Alpha Vantage (stock/market data)
- arXiv API (research papers)
- Hugging Face Papers API
- Custom RSS feeds
**Capabilities:**
- Watchlist price alerts (>2% move)
- Daily market summary
- Weekly AI research digest
- Breaking news triage

**Routing trigger:** `IntentRouter` class = `INTELLIGENCE`

---

## Creative Squad
**Purpose:** Content creation, social media, video production, technical diagrams
**Tools:**
- Manim (mathematical animation)
- Kokoro/Piper TTS (text-to-speech)
- D2 (diagram generation)
- FFmpeg (video processing)
**Capabilities:**
- Script writing for YouTube/social
- Diagram generation from descriptions
- Short-form video script + narration
- Technical tutorial drafting

**Routing trigger:** `IntentRouter` class = `CREATIVE`

---

## Delivery Channels
| Channel | Library | Use |
|---------|---------|-----|
| Telegram | python-telegram-bot | Primary async delivery + HITL gates |
| Voice (Phase 4) | Kokoro/Piper | Local TTS output |
| Wake word (Phase 4) | OpenWakeWord | Always-on trigger |
