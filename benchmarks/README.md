# Sweep Neural Engine Scientific Benchmark

A rigorous, reproducible scientific evaluation framework for the Sweep Neural Engine.

Implements all 48 sections of the Master Prompt specification.

## Quick Start

```bash
# Run the full benchmark with all features
sweep-benchmark full

# Run a quick validation
sweep-benchmark run --suite quick --cases 50

# Run specific suites
sweep-benchmark run --suite reasoning
sweep-benchmark run --suite multimodal
sweep-benchmark run --suite sweep-specific

# With ablation study and honesty report
sweep-benchmark run --suite full --ablation --honesty

# Multi-run testing for statistical robustness
sweep-benchmark run --suite full --multi-run --runs 10

# Run with external models (requires API keys)
OPENAI_API_KEY=sk-... sweep-benchmark run --suite full
ANTHROPIC_API_KEY=sk-... sweep-benchmark run --suite full

# Run contamination checks
sweep-benchmark contamination-check

# Run ablation study
sweep-benchmark ablation
```

## Suites

| Suite | Description | Categories |
|-------|-------------|------------|
| `full` | Complete benchmark | All 18 categories |
| `reasoning` | Reasoning-focused | Reasoning, Math, Planning, Uncertainty |
| `multimodal` | Multimodal perception | Multimodal, Data Analysis |
| `sweep-specific` | Sweep private tasks | Sweep-specific, Entity Resolution, Evidence Reasoning |
| `quick` | Quick validation | Reasoning, Mathematics, Coding |

## Task Categories (18)

1. **Reasoning** — Deductive, inductive, abductive, analogical, counterfactual, common sense, causal
2. **Mathematics** — Arithmetic, algebra, geometry, statistics, word problems, symbolic
3. **Coding** — Generation, debugging, comprehension, refactoring, test generation
4. **Knowledge** — Factual, scientific, historical, geographical, technical
5. **Instruction Following** — Formatting, constraints, nested requirements, schema compliance
6. **Language** — Comprehension, summarization, paraphrasing, translation, grammar
7. **Data Analysis** — Trends, statistics, anomalies, correlations, missing info
8. **Multimodal** — Image understanding, document analysis, scene description
9. **Retrieval** — Direct, multi-hop, query expansion, source triangulation
10. **Entity Resolution** — Same entity, different entity, insufficient evidence
11. **Evidence Reasoning** — Direct observation, supported, contradicted, unknown
12. **Memory** — Short-term, delayed retrieval, interference, false memory
13. **Planning** — Multi-step, dependency handling, failure recovery
14. **Tool Use** — Correct selection, unnecessary calls, failure recovery
15. **Web Research** — Closed world, open world, source quality
16. **Uncertainty** — Known answer, unknown, insufficient evidence, disagreement
17. **Adversarial** — Misleading instructions, contradictory docs, fake evidence
18. **Sweep Specific** — Investigation-style, contradiction detection, temporal reasoning

## Architecture

```
benchmarks/
├── cli.py                  # CLI entry point
├── config/                 # Configuration files
│   ├── benchmark.yaml      # Master benchmark config
│   ├── models.yaml         # Model configurations
│   └── hardware.yaml       # Hardware reporting
├── core/                   # Core engine
│   ├── engine.py           # BenchmarkEngine orchestrator
│   ├── task.py             # Task definitions
│   ├── ablation.py         # Ablation study framework
│   └── normalization.py    # Prompt normalization
├── tasks/                  # Task generators
│   └── generator.py        # Generates all 18 categories
├── evaluators/             # Scoring
│   └── scorer.py           # Deterministic scoring
├── adapters/               # Model adapters
│   ├── base.py             # Abstract adapter
│   ├── sweep_adapter.py    # Sweep Neural Mesh
│   ├── openai_adapter.py   # OpenAI API
│   ├── anthropic_adapter.py# Anthropic API
│   └── google_adapter.py   # Google Gemini
├── contamination/          # Integrity control
│   └── controller.py       # Hash management, holdout verification
├── reports/                # Report generation
│   └── generator.py        # HTML, JSON, CSV, terminal
├── metrics/                # Statistical analysis
│   └── statistics.py       # Brier, ECE, effect sizes, CI
├── datasets/               # Dataset management
│   ├── public/
│   ├── private/
│   ├── generated/
│   └── contamination_holdout/
├── results/                # Execution results
├── logs/                   # Run logs
└── reproducibility/        # Manifests
```

## Key Principles

1. **Honesty** — If Sweep scores 20%, we report 20%
2. **Determinism** — Same seed produces same tasks
3. **Fairness** — Identical prompts, identical conditions
4. **Statistical rigor** — Multi-run with confidence intervals
5. **Contamination control** — SHA-256 hashing, holdout sets
6. **Reproducibility** — Full environment manifests

## Ablation Study (Section 34-35)

Tests six configurations to isolate component contributions:

| Config | Neural | Retrieval | Tools | Description |
|--------|--------|-----------|-------|-------------|
| A | ❌ | ❌ | ❌ | Baseline (rule-based only) |
| B | ✅ | ❌ | ❌ | Neural engine only |
| C | ✅ | ✅ | ❌ | Neural + retrieval |
| D | ✅ | ❌ | ✅ | Neural + tools |
| E | ✅ | ✅ | ✅ | Neural + retrieval + tools |
| F | ✅ | ✅ | ✅ | Full production |

## Three Fair Comparison Modes (Section 24)

| Mode | Description |
|------|-------------|
| `raw_model` | No external search, no tools — core model capability |
| `tool_augmented` | All systems receive equivalent tool access |
| `full_system` | Each system in its production configuration |

## Reporting

Generated reports:
- `final_report.html` — Comprehensive HTML report with all sections
- `final_report.json` — Machine-readable full results
- `leaderboard.csv` — Model comparison leaderboard
- `failure_analysis.csv` — Categorized failure breakdown
- `contamination_report.html` — Integrity verification
- `ablation_report.html` — Component contribution analysis
- `reproducibility_manifest.json` — Environment and config record

## Statistical Analysis (Section 32)

For every comparison:
- Absolute and relative difference
- Confidence intervals (Wilson score)
- Statistical significance (z-test for proportions)
- Effect size (Cohen's h)
- Brier score and ECE for calibration
- McNemar's test for paired comparisons

## Contamination Control (Section 21)

Categories:
- **PUBLIC** — Known benchmark datasets
- **PRIVATE** — Never-public benchmark questions
- **FRESH** — Questions generated after training cutoff
- **HOLDOUT** — Questions never exposed to development system
- **ADVERSARIAL_HOLDOUT** — Hidden test set from independent pipeline

## CLI Reference

```
sweep-benchmark full                              # Full benchmark with all features
sweep-benchmark run --suite [full|reasoning|multimodal|sweep_specific|quick]
                    --cases N
                    --seed N
                    --runs N
                    --output DIR
                    --verbose
                    --ablation
                    --honesty
                    --multi-run
                    --enable-ml
                    --openai-model MODEL
                    --anthropic-model MODEL
                    --google-model MODEL

sweep-benchmark compare --results-dir DIR
sweep-benchmark report --results-dir DIR
sweep-benchmark contamination-check
sweep-benchmark ablation
```

## Section Coverage

| Section | Feature | Status |
|---------|---------|--------|
| 1 | Fundamental Rule (honesty) | ✅ |
| 2 | Benchmark Framework | ✅ |
| 3 | Baseline Benchmarks | ✅ |
| 4 | Mathematical Reasoning | ✅ |
| 5 | Coding | ✅ |
| 6 | Instruction Following | ✅ |
| 7 | Language Capability | ✅ |
| 8 | Data Analysis | ✅ |
| 9 | Memory Test | ✅ |
| 10 | Multimodal Perception | ✅ |
| 11 | Entity Resolution | ✅ |
| 12 | Evidence Reasoning | ✅ |
| 13 | Source Reliability | ✅ |
| 14 | Web Research Benchmark | ✅ |
| 15 | Search Robustness | ✅ |
| 16 | Tool-Use Benchmark | ✅ |
| 17 | Planning Benchmark | ✅ |
| 18 | Self-Correction | ✅ |
| 19 | Adversarial Robustness | ✅ |
| 20 | Unknown/Abstention | ✅ |
| 21 | Contamination Control | ✅ |
| 22 | Sweep-Specific Private Tasks | ✅ |
| 23 | Cross-Model Comparison | ✅ |
| 24 | Three Fair Comparison Modes | ✅ |
| 25 | Prompt Normalization | ✅ |
| 26 | Randomization | ✅ |
| 27 | Multi-Run Testing | ✅ |
| 28 | Latency and Efficiency | ✅ |
| 29 | Failure Taxonomy | ✅ |
| 30 | Confidence Calibration | ✅ |
| 31 | No LLM-Judge Primary | ✅ |
| 32 | Statistical Analysis | ✅ |
| 33 | Avoid Benchmark Gaming | ✅ |
| 34 | Ablation Study | ✅ |
| 35 | Neural-Mesh Ablation | ✅ |
| 36 | Scaling Test | ✅ |
| 37 | Report Generation | ✅ |
| 38 | Leaderboard | ✅ |
| 39 | Overall Score | ✅ |
| 40 | Reproducibility | ✅ |
| 41 | Benchmark Integrity | ✅ |
| 42 | Honesty Requirement | ✅ |
| 43 | Anti-Leak Rule | ✅ |
| 44 | External Model Testing | ✅ |
| 45 | Environment Report | ✅ |
| 46 | Final Command | ✅ |
| 47 | Final Output | ✅ |
| 48 | Most Important Requirement | ✅ |
