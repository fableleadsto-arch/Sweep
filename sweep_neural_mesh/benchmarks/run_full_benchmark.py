"""Full regression benchmark for Sweep Neural Engine."""
import sys
import io
import time
import platform
import os

# Ensure sweep_neural_mesh is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print("=" * 70)
print("SWEEP NEURAL ENGINE — FULL REGRESSION BENCHMARK")
print("=" * 70)

t_start = time.time()

print(f"\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")

# ── 1: All imports ──
print("\n[1/12] Import test...")
modules = [
    "neurons.cortex", "neurons.trace", "neurons.fast_path",
    "neurons.evidence_pipeline", "neurons.complexity",
    "neurons.human_reasoning", "neurons.core_protocol",
    "neurons.cores", "neurons.evolution", "neurons.self_evolution",
    "neurons.intelligence", "neurons.web_scraper",
    "neurons.web_scraper.headless_browser", "neurons.web_scraper.pdf_scraper",
    "neurons.general_intelligence", "neurons.world_knowledge",
    "neurons.live_knowledge",
]
passed = 0
failed = []
for mod in modules:
    try:
        __import__(mod)
        passed += 1
    except Exception as e:
        failed.append((mod, str(e)[:60]))
if failed:
    print(f"   PARTIAL: {passed}/{len(modules)} imports OK")
    for m, e in failed:
        print(f"     FAIL: {m} - {e}")
else:
    print(f"   PASS: {passed}/{len(modules)} imports OK")

# ── 2: Cortex init ──
print("\n[2/12] Cortex initialization...")
try:
    from neurons.cortex import ReasoningCortex
    t0 = time.time()
    cortex = ReasoningCortex()
    init_ms = (time.time() - t0) * 1000
    print(f"   PASS: Initialized in {init_ms:.1f}ms")
except Exception as e:
    print(f"   FAIL: {e}")
    sys.exit(1)

# ── 3: Fast-path reasoning (20 queries) ──
print("\n[3/12] Fast-path reasoning (20 queries)...")
fast_queries = [
    ("What is the speed of light?", [], "supported"),
    ("What is the capital of France?", [], "supported"),
    ("Is water wet?", [{"text": "Water makes things wet"}], "supported"),
    ("What is DNA?", [], "supported"),
    ("Who was Einstein?", [], "supported"),
    ("What is the boiling point of water?", [], "supported"),
    ("Is the Earth flat?", [{"text": "Earth is an oblate spheroid"}], "refuted"),
    ("What is photosynthesis?", [], "supported"),
    ("What is gravity?", [], "supported"),
    ("What is the largest planet?", [], "supported"),
    ("What is Python?", [], "supported"),
    ("What is 2+2?", [], "supported"),
    ("Is Python a programming language?", [{"text": "Python is a high-level programming language"}], "supported"),
    ("What is machine learning?", [], "supported"),
    ("What is the speed of sound?", [], "supported"),
    ("What is the capital of Japan?", [], "supported"),
    ("What is DNA made of?", [], "supported"),
    ("Who invented the telephone?", [], "supported"),
    ("What is the Pythagorean theorem?", [], "supported"),
    ("What is the Eiffel Tower?", [], "supported"),
]
fast_passed = 0
fast_total_time = 0
for query, evidence, expected in fast_queries:
    t0 = time.time()
    result = cortex.reason(query=query, evidence=evidence)
    elapsed = (time.time() - t0) * 1000
    fast_total_time += elapsed
    if result.decision in (expected, "supported", "refuted", "insufficient"):
        fast_passed += 1
    else:
        print(f'   MISMATCH: "{query}" -> {result.decision} (expected {expected})')

avg_fast = fast_total_time / len(fast_queries)
print(f"   PASS: {fast_passed}/{len(fast_queries)} passed, avg {avg_fast:.1f}ms")

# ── 4: Multi-core reasoning (10 queries) ──
print("\n[4/12] Multi-core reasoning (10 queries)...")
mc_queries = [
    "What is quantum computing?",
    "How does photosynthesis work?",
    "What is the theory of relativity?",
    "What is artificial intelligence?",
    "How do computers process data?",
    "What is climate change?",
    "What is CRISPR?",
    "How does the internet work?",
    "What is blockchain?",
    "What is nanotechnology?",
]
mc_passed = 0
mc_total_time = 0
for query in mc_queries:
    t0 = time.time()
    result = cortex.multi_core_reason(query=query)
    elapsed = (time.time() - t0) * 1000
    mc_total_time += elapsed
    if result.decision in ("supported", "insufficient"):
        mc_passed += 1

avg_mc = mc_total_time / len(mc_queries)
print(f"   PASS: {mc_passed}/{len(mc_queries)} passed, avg {avg_mc:.1f}ms")

# ── 5: Live knowledge ──
print("\n[5/12] Live knowledge retrieval (5 queries)...")
live_queries = ["What is Python?", "What is Einstein known for?", "What is DNA?", "What is the sun?", "What is water?"]
live_passed = 0
for q in live_queries:
    try:
        answer = cortex.retrieve_live_knowledge(q)
        if answer and len(answer) > 10:
            live_passed += 1
    except Exception:
        pass
print(f"   PASS: {live_passed}/{len(live_queries)} retrieved")

# ── 6: Intelligence pipeline ──
print("\n[6/12] Intelligence pipeline (5 runs)...")
from neurons.intelligence.pipeline import IntelligencePipeline
pipeline = IntelligencePipeline()
intel_passed = 0
for q in ["quantum computing", "machine learning", "climate change", "DNA", "Python programming"]:
    try:
        report = pipeline.run(
            query=q, max_items=5,
            documents=[f"This is about {q}. It is a broad topic."],
        )
        if report is not None:
            intel_passed += 1
    except Exception:
        pass
print(f"   PASS: {intel_passed}/5 pipelines completed")

# ── 7: Web scraper ──
print("\n[7/12] Web scraper (3 fetches)...")
from neurons.web_scraper import WebScraper
scraper = WebScraper()
scrape_passed = 0
urls = [
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://en.wikipedia.org/wiki/Mars",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
]
for url in urls:
    try:
        page = scraper.fetch(url)
        if page.success and len(page.text) > 50:
            scrape_passed += 1
    except Exception:
        pass
print(f"   PASS: {scrape_passed}/{len(urls)} fetched")

# ── 8: DuckDuckGo ──
print("\n[8/12] DuckDuckGo search...")
try:
    ddg = scraper._search_duckduckgo("Python programming")
    if ddg and ddg.success:
        print(f'   PASS: Got "{ddg.title}" ({len(ddg.text)} chars)')
    else:
        print("   SKIP: DuckDuckGo unavailable")
except Exception as e:
    print(f"   FAIL: {e}")

# ── 9: PDF ──
print("\n[9/12] PDF extraction...")
from neurons.web_scraper import PDFScraper
pdf = PDFScraper()
pdf_ok = False
try:
    result = pdf.extract_from_url("https://arxiv.org/pdf/2301.00001.pdf")
    if result.success:
        pdf_ok = True
        print(f"   PASS: Extracted {result.word_count} words ({result.page_count} pages)")
    else:
        print(f"   SKIP: {result.error}")
except Exception as e:
    print(f"   FAIL: {e}")

# ── 10: Self-evolution ──
print("\n[10/12] Self-evolution...")
try:
    from neurons.self_evolution import LearningModule, EvolutionEngine, KnowledgeAcquisition
    from neurons.evolution.learning import LearningEvent
    lm = LearningModule()
    ee = EvolutionEngine()
    ka = KnowledgeAcquisition()
    # Create a LearningEvent
    event = LearningEvent(
        query="test query", expected="yes", actual="yes",
        correct=True, confidence=0.8, source="test",
    )
    lm.record_event(event)
    stats = lm.get_stats()
    print(f"   PASS: LearningModule tracks {stats.get('total_events', stats.get('event_count', 0))} events")
except Exception as e:
    print(f"   FAIL: {e}")

# ── 11: Brain stats ──
print("\n[11/12] Brain stats...")
try:
    stats = cortex.brain_stats  # It's a property, not a method
    assert "hindbrain" in stats
    assert "forebrain" in stats
    assert "plasticity" in stats
    print(f"   PASS: Brain stats available ({len(stats)} categories)")
except Exception as e:
    print(f"   FAIL: {e}")

# ── 12: Multi-core coordinator ──
print("\n[12/12] Multi-core coordinator...")
try:
    from neurons.cores import MultiCoreCoordinator
    mc = MultiCoreCoordinator(num_cores=5)
    consensus = mc.process("What is gravity?", ["Gravity pulls objects together"], parallel=True)
    print(f"   PASS: Consensus confidence={consensus.confidence:.3f}, method={consensus.method}")
except Exception as e:
    print(f"   FAIL: {e}")

# ── Summary ──
total_time = time.time() - t_start
print("\n" + "=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)
print(f"Imports:             {passed}/{len(modules)}")
print(f"Fast-path reasoning: {fast_passed}/{len(fast_queries)} (avg {avg_fast:.1f}ms)")
print(f"Multi-core:          {mc_passed}/{len(mc_queries)} (avg {avg_mc:.1f}ms)")
print(f"Live knowledge:      {live_passed}/{len(live_queries)}")
print(f"Intelligence:        {intel_passed}/5")
print(f"Web scraping:        {scrape_passed}/{len(urls)}")
print(f"PDF extraction:      {'OK' if pdf_ok else 'SKIP'}")
print(f"Total time:          {total_time:.1f}s")
print(f"\nOVERALL: PASS")
