# Sweep Neural Mesh — Architecture

## Overview

Sweep is a **neural reasoning system** built on a biologically-inspired architecture. It processes queries through multiple specialized modules that mirror brain regions, then uses multi-core parallel processing with consensus voting to produce accurate, well-calibrated answers.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SWEEP NEURAL MESH                            │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ General  │   │  World   │   │  Live    │   │  Multi-  │   │
│  │ Intel.   │   │ Knowl.   │   │ Knowl.   │   │  Core    │   │
│  │ (static) │   │ (trained)│   │ (APIs)   │   │(5 cores) │   │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   │
│       │              │              │              │           │
│       └──────────────┴──────────────┴──────────────┘           │
│                            │                                   │
│                    ┌───────┴───────┐                           │
│                    │    Cortex     │                           │
│                    │ (orchestrator)│                           │
│                    └───────┬───────┘                           │
│                            │                                   │
│       ┌────────────────────┼────────────────────┐              │
│       │                    │                    │              │
│  ┌────┴────┐         ┌─────┴─────┐        ┌────┴────┐        │
│  │Hindbrain│         │ Midbrain  │        │Forebrain │        │
│  │(filter) │         │ (route)   │        │(process) │        │
│  └─────────┘         └───────────┘        └──────────┘        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Self-Evolution System                       │   │
│  │  Learning → Evolution → Acquisition → Performance       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. General Intelligence (`general_intelligence.py`)

**Purpose**: Fast-path answering for common knowledge questions.

- Pre-compiled regex patterns (922+) indexed by keywords.
- Covers physics, biology, geography, astronomy, chemistry, everyday facts.
- **Latency**: ~0.1ms per query (keyword index → regex match).
- **Confidence**: 0.7 – 0.99 depending on match quality.

```
Query → keyword index → candidate patterns → regex match → answer
```

### 2. World Knowledge (`knowledge_training.py` + `knowledge_supplement.py`)

**Purpose**: Structured knowledge base with 722+ entries across 29 domains.

- Entries have: topic, answer, source, confidence, category, relationship.
- Loaded into the Cortex at init time.
- Used for evidence grounding and fact verification.

### 3. Live Knowledge (`live_knowledge.py`)

**Purpose**: Real-time knowledge retrieval from external APIs.

- **Wikipedia API** — encyclopedic knowledge.
- **Wikidata API** — structured entity data.
- LRU cache (2000 entries) to avoid redundant calls.
- 3s timeout with connection pooling.

### 3a. Web Scraper (`web_scraper/`)

**Purpose**: Multi-source web scraping and content extraction.

```
neurons/web_scraper/
├── __init__.py         — Package exports
├── scraper.py          — Core WebScraper: multi-strategy fetching
├── content.py          — ContentExtractor: HTML → clean text
└── researcher.py       — WebResearcher: multi-query research
```

**Sources**:
| Source | Method | Speed | Coverage |
|--------|--------|-------|----------|
| Wikipedia | API (structured) | Fast | Encyclopedia |
| Wikidata | API (structured) | Fast | Entity data |
| arXiv | API (XML) | Medium | Academic papers |
| OpenAlex | API (JSON) | Medium | Academic works |
| Generic HTML | HTTP + regex parsing | Slow | Any webpage |

**Features**:
- Connection pooling with keep-alive (10 connections, 30s expiry)
- Domain-level rate limiting (100ms between same-domain requests)
- LRU cache with TTL (1000 pages, 1hr expiry)
- Content deduplication by title and text hash
- Boilerplate removal (nav, footer, scripts, ads)
- Entity extraction from scraped content
- Key fact extraction (prioritizes sentences with numbers/dates)

**Usage**:
```python
# Direct scraping
scraper = WebScraper()
page = scraper.fetch("https://en.wikipedia.org/wiki/Quantum_computing")
print(page.title, page.text[:200])

# Multi-source research
researcher = WebResearcher()
report = researcher.research("quantum computing applications")
for finding in report.findings:
    print(f"[{finding.source}] {finding.title}")

# Through Cortex
cortex = ReasoningCortex()
report = cortex.web_research("quantum computing")
text = cortex.fetch_web_page("https://example.com/article")
```

### 4. Multi-Core Neural Processing (`cores/`)

**Purpose**: Parallel specialized processing with consensus voting.

```
┌─────────────────────────────────────────────────────────┐
│  MultiCoreCoordinator                                   │
│                                                         │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐            │
│  │ Factual   │ │ Reasoning │ │ Evidence  │            │
│  │ Core      │ │ Core      │ │ Core      │            │
│  │ 80+ facts │ │ 30+ logic │ │ relevance │            │
│  └───────────┘ └───────────┘ └───────────┘            │
│  ┌───────────┐ ┌───────────┐                           │
│  │ Temporal  │ │ Causal    │                           │
│  │ Core      │ │ Core      │                           │
│  │ dates/time│ │ cause→eff │                           │
│  └───────────┘ └───────────┘                           │
│       ↓              ↓              ↓                   │
│  ┌─────────────────────────────────────────────┐       │
│  │  Consensus Engine (voting + agreement)      │       │
│  └─────────────────────────────────────────────┘       │
│                     ↓                                   │
│              Final Decision                             │
└─────────────────────────────────────────────────────────┘
```

| Core | File | Responsibility |
|------|------|----------------|
| **FactualCore** | `cores/factual_core.py` | Knowledge lookup, number extraction |
| **ReasoningCore** | `cores/reasoning_core.py` | Logic, common sense, yes/no, math |
| **EvidenceCore** | `cores/evidence_core.py` | Evidence relevance, definition extraction |
| **TemporalCore** | `cores/temporal_core.py` | Dates, historical events |
| **CausalCore** | `cores/causal_core.py` | Cause-effect chains |

All cores implement `NeuralCoreProtocol` from `core_protocol.py`.

**Consensus methods**:
- `voting` — high agreement (>80%) → boost confidence
- `weighted` — medium agreement (50-80%) → use highest confidence
- `fallback` — low agreement (<50%) → penalise confidence
- `single` — only one core responded

### 5. Cortex (`cortex.py`)

**Purpose**: Master orchestrator that runs the full reasoning pipeline.

Implements a **three-division brain architecture**:

| Division | Components | Function |
|----------|-----------|----------|
| **Hindbrain** | Brainstem, Cerebellum | Fast filtering, reflexes, energy gating |
| **Midbrain** | Thalamus, VTA | Signal routing, attention, reward prediction |
| **Forebrain** | Cortex, Basal Ganglia, Hippocampus | Processing centers, memory, decision-making |

**Reasoning flow**:

```
Raw Input
    ↓
GI Fast Path → answer? → return (0ms)
    ↓
Live Knowledge → answer? → return
    ↓
Hindbrain → filter → reflex check → energy gate
    ↓
Midbrain → value predict → salience → inhibit → route
    ↓
Forebrain → workspace → working memory → processing centers
    ↓
Cortex-BG-Thalamus loop → metacognition → output
```

### 6. Self-Evolution (`evolution/`)

**Purpose**: Enables the system to learn and adapt from interactions.

| Module | File | Function |
|--------|------|----------|
| **LearningModule** | `evolution/learning.py` | Track successes/failures, learn from feedback |
| **EvolutionEngine** | `evolution/engine.py` | Mutate failing patterns, cross-pollinate |
| **KnowledgeAcquisition** | `evolution/knowledge.py` | Acquire new knowledge, validate, deduplicate |
| **PerformanceTracker** | `evolution/tracker.py` | Monitor metrics, calibrate confidence, suggest optimisations |
| **Coordinator** | `evolution/coordinator.py` | Orchestrate all the above |

---

## Data Flow

### Fast Path (common questions)

```
Query → General Intelligence (0.1ms) → answer with confidence ≥ 0.85 → return
```

### Medium Path (questions needing evidence)

```
Query → Cortex → Hindbrain → Forebrain → Processing Centers → answer
```

### Slow Path (complex reasoning)

```
Query → Cortex → Full brain pipeline → Multi-Core parallel → Consensus → answer
```

### Learning Path (after answering)

```
Answer → Self-Evolution → learn → evolve → acquire → optimise
```

---

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `cortex.py` | ~500 | Master orchestrator (slimmed from 1740) |
| `trace.py` | ~200 | ReasoningTrace + ReasoningResult data classes |
| `fast_path.py` | ~140 | Early-exit for simple queries |
| `evidence_pipeline.py` | ~135 | Cross-referencing and corroboration |
| `complexity.py` | ~90 | Adaptive pipeline depth classification |
| `human_reasoning.py` | ~160 | Common sense, abductive, ToM, narrative, analogical, causal, counterfactual |
| `core_protocol.py` | ~120 | NeuralCoreProtocol + CoreResult + ConsensusResult |
| `cores/*.py` | ~100 each | Individual neural cores (factual, reasoning, evidence, temporal, causal) |
| `evolution/*.py` | ~80 each | Self-evolution modules (learning, engine, knowledge, tracker) |
| `general_intelligence.py` | ~600 | Fast-path knowledge lookup (922+ patterns) |
| `live_knowledge.py` | ~300 | External API retrieval (Wikipedia, Wikidata) |
| `knowledge_training.py` | ~1500 | 500-entry knowledge base |
| `knowledge_supplement.py` | ~800 | 222-entry supplementary knowledge |

---

## Cortex Refactoring (2026-08-29)

The original `cortex.py` was **1740 lines** with a single `reason()` method of **780 lines**. It was refactored into focused modules:

```
cortex.py (500 lines, orchestrator only)
  ├── trace.py              — ReasoningTrace + ReasoningResult
  ├── fast_path.py          — early-exit for simple queries
  ├── evidence_pipeline.py  — cross-referencing & corroboration
  ├── complexity.py         — adaptive pipeline depth
  └── human_reasoning.py    — 7 human-like reasoning modules
```

Benefits:
- **45% smaller** cortex (1740 → 500 lines)
- **Each concern** in its own file with clear responsibilities
- **`reason()`** reduced from 780 → ~200 lines (delegates to helpers)
- **Testable** — each module can be unit-tested independently
- **Backwards compatible** — `from .cortex import ReasoningResult` still works

## Design Principles

1. **Protocol-driven**: All cores implement `NeuralCoreProtocol`.
2. **Pre-compiled**: Regex patterns compiled once at init, not per-query.
3. **Lazy-loaded**: Expensive modules (ML engines, live APIs) loaded on first use.
4. **Fail-safe**: External failures (APIs, ML) never crash the pipeline.
5. **Observable**: Every result includes latency, confidence, and reasoning trace.
6. **Evolvable**: The system learns from interactions and adapts over time.
7. **Modular**: Each concern lives in its own file (~100-200 lines).
