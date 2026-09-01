# Sweep Neural Engine — Architecture

## Overview

The Sweep Neural Engine is a **CPU-first** reasoning system that uses original orchestration logic to combine formal reasoning, pretrained models, and deterministic algorithms.

## Architecture Diagram

```
                    SWEEP
                      │
                 INPUT LAYER
                      │
                 INTENT CORE (task_handlers/router.py)
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       LANGUAGE     VISION      AUDIO
          │           │           │
          └───────────┼───────────┘
                      ↓
              LOGIC ENGINES (PRIMARY)
              ├── LogicalInferenceEngine
              ├── NeuralProofMesh
              └── BayesianReasoner
                      ↓
                MEMORY LAYER
                ├── Working Memory
                ├── Evidence Memory
                ├── Semantic Memory
                └── User Memory
                      ↓
              RETRIEVAL ENGINE (web_scraper/)
                      ↓
              REASONING ENGINE (engine/reasoning.py)
                      ↓
                TOOL ROUTER (engine/tools.py)
                      ↓
              VERIFICATION CORE (engine/verification.py)
                      ↓
                 OUTPUT CORE
```

## Key Design Decisions

### 1. Logic Over Rules

The cortex routes queries through formal logic engines FIRST:

1. **NeuralProofMesh** — atom/bond grounding with fuzzy t-norm propagation
2. **LogicalInferenceEngine** — modus ponens, modus tollens, transitivity, syllogisms
3. **BayesianReasoner** — evidence-weighted probability updating

Regex-based task handlers serve as fallback when logic engines can't parse the query.

### 2. CPU-First Design

- No CUDA dependency
- Automatic hardware detection (CPU cores, RAM, GPU)
- Three runtime profiles: LOW_RESOURCE, BALANCED, HIGH_PERFORMANCE
- Lazy model loading with memory-aware scheduling

### 3. Provider Abstraction

All model capabilities are accessed through provider interfaces:

```
LanguageProvider  → text understanding
VisionProvider    → image understanding
AudioProvider     → audio understanding
EmbeddingProvider → semantic similarity
RetrievalProvider → information retrieval
```

Providers are replaceable. Sweep's core logic never depends on specific models.

### 4. Memory Architecture

Four independent memory layers:

- **Working Memory**: Temporary task-specific data
- **Evidence Memory**: Sources, claims, entities with provenance tracking
- **Semantic Memory**: Embedding-based similarity search
- **User Memory**: Only explicitly authorized persistent data

### 5. Verification Core

Every answer is self-checked for:
- Hallucination indicators
- Overconfidence without evidence
- Evidence contradictions
- Unsupported claims
- Calculation errors

## File Structure

```
sweep_neural_mesh/
├── engine/                    # NEW: CPU-first architecture
│   ├── __init__.py
│   ├── providers.py           # Provider abstraction layer
│   ├── memory.py              # 4-layer memory system
│   ├── tools.py               # ToolProvider/Registry/Router
│   ├── verification.py        # Self-checking core
│   └── reasoning.py           # Reasoning orchestrator
├── neurons/                   # Core neural mesh
│   ├── cortex.py              # Main orchestrator (uses logic engines)
│   ├── proof_mesh.py          # Proof propagation (atoms/bonds)
│   ├── logical_inference.py   # Formal logic (modus ponens/tollens)
│   ├── bayesian.py            # Bayesian reasoning
│   ├── fuzzy_logic.py         # Multi-valued logic
│   ├── task_handlers/         # Regex fallback handlers
│   ├── cores/                 # Specialized cores
│   └── ...
├── training/                  # Training infrastructure
├── benchmarks/                # Evaluation suite
└── reports/                   # Status reports
```

## Reasoning Pipeline

When a query enters the cortex:

1. **GI Fast Path** — Check general intelligence for known facts
2. **Logic Engines** — Try formal reasoning (proof_mesh → logical_inference)
3. **Task Router** — Regex-based pattern matching (fallback)
4. **Live Knowledge** — Web search if no answer found
5. **Hindbrain** — Sanity check
6. **Forebrain** — 6 processing centers
7. **Integration** — Combine all signals
8. **Verification** — Self-check before output

## Third-Party Dependencies

| Package | License | Purpose |
|---------|---------|---------|
| sentence-transformers | Apache 2.0 | Text embeddings (MiniLM) |
| torch | BSD | Neural network inference |
| requests | Apache 2.0 | HTTP client |
| beautifulsoup4 | MIT | HTML parsing |

Sweep's orchestration, reasoning, and verification logic is independently implemented.
