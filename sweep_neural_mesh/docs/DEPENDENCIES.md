# Sweep Dependencies

## Third-Party Packages

| Package | Version | License | Purpose | Modifies? |
|---------|---------|---------|---------|-----------|
| sentence-transformers | 2.x | Apache 2.0 | Text embeddings (MiniLM) | No — used as-is |
| torch | 2.x | BSD | Neural network inference | No — used as-is |
| requests | 2.x | Apache 2.0 | HTTP client | No |
| beautifulsoup4 | 4.x | MIT | HTML parsing | No |
| numpy | 1.x | BSD | Numerical operations | No |

## Pretrained Models

| Model | Source | License | Size | Purpose |
|-------|--------|---------|------|---------|
| all-MiniLM-L6-v2 | sentence-transformers | Apache 2.0 | 80MB | Text embeddings |
| Relay Transformer (nano) | Sweep-trained | Sweep Original | 8.4MB | Token prediction |
| Relay Transformer (small) | Sweep-trained | Sweep Original | 42MB | Token prediction |

## Sweep-Original Components

- `engine/` — Provider abstraction, memory, tools, verification, reasoning
- `neurons/cortex.py` — Main reasoning orchestrator
- `neurons/proof_mesh.py` — Atom/bond proof propagation
- `neurons/logical_inference.py` — Formal logic engine
- `neurons/bayesian.py` — Bayesian reasoning
- `neurons/fuzzy_logic.py` — Fuzzy set operations
- `neurons/task_handlers/` — Regex-based query routing
- `neurons/cores/` — Specialized reasoning cores
- `training/` — Training infrastructure
- `benchmarks/` — Evaluation suite
