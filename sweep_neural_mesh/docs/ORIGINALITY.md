# Sweep Originality Audit

## Sweep-Original Code

All of the following are independently implemented for Sweep:

### Engine Layer (`engine/`)
- `providers.py` — Provider abstraction (LanguageProvider, VisionProvider, etc.)
- `memory.py` — 4-layer memory system (Working, Evidence, Semantic, User)
- `tools.py` — ToolProvider, ToolRegistry, ToolRouter
- `verification.py` — Self-checking verification core
- `reasoning.py` — Reasoning orchestrator pipeline

### Neural Mesh (`neurons/`)
- `cortex.py` — Main reasoning orchestrator with logic-first pipeline
- `proof_mesh.py` — Atom/bond proof propagation with fuzzy t-norms
- `logical_inference.py` — Modus ponens, modus tollens, transitivity, syllogisms
- `bayesian.py` — Bayesian evidence updating
- `fuzzy_logic.py` — Fuzzy set operations and reasoning
- `task_handlers/` — Logic, math, evidence, temporal, causal handlers
- `cores/` — Factual, reasoning, evidence, temporal, causal cores
- `intelligence/` — Intelligence pipeline
- `evolution/` — Self-evolution system
- `web_scraper/` — Web scraping with PDF and headless browser support

### Training (`training/`)
- `real_training.py` — Relay Transformer training
- `hybrid_engine.py` — Multi-model ensemble
- `comprehensive_benchmark.py` — Full evaluation suite
- All training infrastructure (curriculum, adversarial, ablation, etc.)

## Third-Party (Used as Dependencies)

- sentence-transformers (Apache 2.0) — loaded as pretrained model
- torch (BSD) — used for tensor operations
- requests (Apache 2.0) — HTTP client
- beautifulsoup4 (MIT) — HTML parsing

## Key Principle

Sweep's orchestration, reasoning pipeline, data structures, interfaces, evaluation logic, routing system, memory architecture, and application logic are all independently implemented.

Third-party models are used as legitimate dependencies through provider interfaces. Sweep's core logic never depends on specific models.
