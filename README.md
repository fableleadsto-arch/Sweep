# Sweep — Web Intelligence Platform

A comprehensive web intelligence platform for OSINT, web scraping, search engine interaction, entity extraction, and evidence-based research.

## Architecture

```
Web Surfing / Discovery
    ↓
Scraping / Crawling (rate-limited, retried, cached)
    ↓
Search & Source Discovery (multi-engine: DuckDuckGo, Bing, Brave, Google, Tavily, Exa)
    ↓
Content Extraction (3-layer: structured → heuristic → semantic)
    ↓
Normalization / Structuring (canonicalize, dedup, SimHash)
    ↓
Entity / Person / Organization Extraction
    ↓
Evidence & Relationship Graph
    ↓
ML / Intelligence Brain [to be built]
    ↓
Analysis / Ranking / Reasoning
    ↓
User Interface
```

## Tech Stack

- **Frontend**: React 19, TanStack Router, TanStack Query, Tailwind CSS 4
- **Backend**: TanStack Start (Vite-based SSR), Express API server
- **Python Service**: FastAPI companion brain with compute backends
- **Search Engines**: DuckDuckGo, Bing, Brave, Mojeek, Google (HTML parsers) + Tavily, Exa, SearXNG, Jina (API providers)
- **Browser**: Playwright via WebSocket (browserless.io) with HTTP fallback
- **ML Infrastructure**: PyTorch, TensorFlow, ONNX, scikit-learn, NumPy, pandas (lazy-loaded)
- **Database**: Supabase (PostgreSQL), Qdrant (vectors)

## Key Features

### Web Intelligence
- **Multi-engine search**: 6 HTML parsers + 4 API providers with intelligent routing
- **Resilient HTTP**: Rate limiting, retries, SSRF guard, CAPTCHA detection
- **Bounded crawling**: Configurable depth, concurrency, budgets
- **Structured extraction**: JSON-LD, microdata, heuristic regex, semantic LLM
- **Document normalization**: Canonical URLs, SimHash dedup, content cleaning

### Platform Adapters
- Reddit, X (Twitter), GitHub, YouTube, Instagram, LinkedIn
- Honest access mode reporting (public/authenticated/unavailable)

### Browser Automation
- Playwright via WebSocket with HTTP fallback
- Session management with back/forward history
- Page data extraction (text, links, headings, metadata)

### Research Engine
- Multi-step research with planning
- Evidence collection and source scoring
- Cited report synthesis

### Python ML Service
- 35+ registered capabilities with lazy-loading
- Neural network registry, training, inference
- Knowledge ingestion from 9 source adapters
- Embedding pipeline and vector search

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Run tests
npm test

# Typecheck
npm run typecheck
```

## Project Structure

```
src/
├── components/          # React UI components
│   ├── lead-intel/      # Lead intelligence panel
│   ├── nav/             # Navigation (hotbar, topbar)
│   ├── surf/            # Surf browser components
│   └── ui/              # Shared UI primitives
├── lib/
│   ├── lead-intelligence/  # B2B lead discovery
│   ├── nav/               # Navigation config
│   ├── surf/              # Web surfing layer
│   ├── web-intelligence/  # Data contracts (Document, Entity, Evidence)
│   └── rpc/               # RPC runtime
├── RelAI/              # Core web intelligence layer
│   ├── browser/        # Playwright automation
│   ├── compute/        # Computation engine
│   ├── core/           # Agent orchestration
│   ├── tools/          # Tool registry
│   └── web/            # HTTP, search, crawl, extract, normalize
├── routes/             # TanStack Router routes
├── surf/               # Surf layer (search, browse, research, platforms)
└── styles/             # CSS
companion/              # Python ML service
├── neural/             # Neural network infrastructure
├── ingest/             # Knowledge ingestion
├── tools/              # ML/NLP/CS tools
└── compute/            # Backend management
```

## ML Roadmap

See [docs/ML_ROADMAP.md](docs/ML_ROADMAP.md) for the plan to build the intelligence brain from scratch.

See [docs/ARCHITECTURE_AUDIT.md](docs/ARCHITECTURE_AUDIT.md) for the full architectural audit.

## License

Private — All rights reserved.
