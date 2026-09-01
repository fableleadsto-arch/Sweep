"""Comprehensive evaluation across all capability layers."""
import sys, os, io, time, json
from pathlib import Path

_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from training.hybrid_engine import HybridEngine

engine = HybridEngine()

# ══════════════════════════════════════════════════════════
# LEVEL 0: Basic Computation
# ══════════════════════════════════════════════════════════
level0 = [
    ("What is 25 + 37?", "62"),
    ("What is 144 - 89?", "55"),
    ("What is 12 * 17?", "204"),
    ("What is 50% of 300?", "150"),
    ("What is 75% of 240?", "180"),
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of India?", "New Delhi"),
    ("What is the boiling point of water?", "100"),
    ("What is the freezing point of water?", "0"),
    ("What is the speed of light?", "299792458"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the chemical symbol for silver?", "Ag"),
]

# ══════════════════════════════════════════════════════════
# LEVEL 1: Language Understanding
# ══════════════════════════════════════════════════════════
level1 = [
    ("Who discovered penicillin?", "Fleming"),
    ("When was the Declaration of Independence signed?", "1776"),
    ("What year did WWII end?", "1945"),
    ("What year was the first Moon landing?", "1969"),
    ("How many bones are in the human body?", "206"),
    ("What does DNA stand for?", "deoxyribonucleic acid"),
    ("What is the largest ocean?", "Pacific"),
    ("What is the tallest mountain?", "Everest"),
    ("How many planets are in our solar system?", "8"),
    ("What is the largest continent?", "Asia"),
]

# ══════════════════════════════════════════════════════════
# LEVEL 2: Reasoning (evidence-based)
# ══════════════════════════════════════════════════════════
level2 = [
    ("Is exercise beneficial?", ["Studies show exercise reduces heart disease risk"], "support"),
    ("Is smoking harmful?", ["Research shows smoking causes lung cancer"], "support"),
    ("Is climate change real?", ["Scientific consensus confirms global warming"], "support"),
]

# ══════════════════════════════════════════════════════════
# Task Handler evaluation
# ══════════════════════════════════════════════════════════
task_tests = [
    ("logic", "All cats are animals. All animals are living things. Is a cat a living thing?", "yes", []),
    ("math", "What is 15% of 200?", "30", []),
    ("math", "What is 25 * 4?", "100", []),
    ("math", "Convert 5 kilometers to miles", "3.1", []),
    ("math", "What is the union of {1, 2, 3} and {3, 4, 5}?", "{1, 2, 3, 4, 5}", []),
    ("evidence", "Does exercise improve health?", "support", ["Studies show exercise reduces heart disease risk"]),
    ("temporal", "What happened first, the Moon landing or WWII?", "WWII", []),
]

print("=" * 70)
print("SWEEP NEURAL ENGINE — COMPREHENSIVE EVALUATION")
print("=" * 70)

all_results = {}
latencies = []

# Run Level 0
print("\n--- LEVEL 0: Basic Computation ---")
correct = 0
for query, expected in level0:
    t0 = time.perf_counter()
    result = engine.answer(query)
    lat = (time.perf_counter() - t0) * 1000
    latencies.append(lat)
    found = expected.lower() in result.answer.lower()
    if found:
        correct += 1
    status = "PASS" if found else "FAIL"
    if not found:
        print(f"  {status}: {query[:45]} -> '{result.answer[:40]}' (expected: {expected})")
all_results["level0_basic"] = correct / len(level0)
print(f"  Result: {correct}/{len(level0)} = {correct/len(level0):.1%}")

# Run Level 1
print("\n--- LEVEL 1: Language Understanding ---")
correct = 0
for query, expected in level1:
    t0 = time.perf_counter()
    result = engine.answer(query)
    lat = (time.perf_counter() - t0) * 1000
    latencies.append(lat)
    found = expected.lower() in result.answer.lower()
    if found:
        correct += 1
    status = "PASS" if found else "FAIL"
    if not found:
        print(f"  {status}: {query[:45]} -> '{result.answer[:40]}' (expected: {expected})")
all_results["level1_language"] = correct / len(level1)
print(f"  Result: {correct}/{len(level1)} = {correct/len(level1):.1%}")

# Run Level 2
print("\n--- LEVEL 2: Evidence Reasoning ---")
correct = 0
for query, evidence, expected in level2:
    t0 = time.perf_counter()
    result = engine.answer(query, evidence)
    lat = (time.perf_counter() - t0) * 1000
    latencies.append(lat)
    found = expected in result.answer.lower() or expected in str(result.components_used)
    if found:
        correct += 1
    status = "PASS" if found else "FAIL"
    if not found:
        print(f"  {status}: {query[:45]} -> '{result.answer[:40]}' (expected: {expected})")
all_results["level2_reasoning"] = correct / len(level2)
print(f"  Result: {correct}/{len(level2)} = {correct/len(level2):.1%}")

# Run Task Handler tests
print("\n--- Task Handler Tests ---")
try:
    from neurons.task_handlers.router import TaskRouter
    router = TaskRouter()
    correct = 0
    for category, query, expected, evidence in task_tests:
        t0 = time.perf_counter()
        result = router.route(query, evidence=evidence)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        found = expected.lower() in result.answer.lower()
        if found:
            correct += 1
        status = "PASS" if found else "FAIL"
        print(f"  {status}: [{category}] {query[:40]} -> '{result.answer[:30]}' (lat={lat:.1f}ms)")
    all_results["task_handlers"] = correct / len(task_tests)
    print(f"  Result: {correct}/{len(task_tests)} = {correct/len(task_tests):.1%}")
except Exception as e:
    print(f"  ERROR: {e}")
    all_results["task_handlers"] = 0.0

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for category, score in all_results.items():
    print(f"  {category:25s}: {score:.1%}")

overall = sum(all_results.values()) / len(all_results) if all_results else 0
avg_lat = sum(latencies) / len(latencies) if latencies else 0
print(f"\n  {'OVERALL':25s}: {overall:.1%}")
print(f"  {'AVG LATENCY':25s}: {avg_lat:.1f}ms")
print(f"  {'TOTAL TESTS':25s}: {len(level0) + len(level1) + len(level2) + len(task_tests)}")

# Save results
results_path = str(_sweep_dir / "reports" / "comprehensive_eval_results.json")
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, "w") as f:
    json.dump({
        "results": all_results,
        "overall": overall,
        "avg_latency_ms": avg_lat,
        "total_tests": len(level0) + len(level1) + len(level2) + len(task_tests),
    }, f, indent=2)
print(f"\nResults saved to {results_path}")
