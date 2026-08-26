"""
Verify benchmark results are genuine.
Re-run Sweep on ALL 1000 cases and compute accuracy from scratch,
comparing against the saved results to detect any caching/staleness.
"""
import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")

from sweep_neural_mesh.neurons.cortex import ReasoningCortex

# Load the SAME dataset used by the benchmark
dataset = json.load(open(r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII\benchmarks\results\dataset.json", encoding="utf-8"))
cases = dataset["cases"]
print(f"Dataset: {len(cases)} cases")

# Load saved results
saved = json.load(open(r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII\benchmarks\results\sweep_results.json", encoding="utf-8"))
saved_correct = sum(1 for r in saved["results"] if r["decision_correct"])
print(f"Saved results: {saved_correct}/{len(saved['results'])} correct = {saved_correct/len(saved['results'])*100:.1f}%")

# Fresh run with a NEW cortex instance
cortex = ReasoningCortex(enable_ml=False)
fresh_correct = 0
fresh_results = []
mismatches = 0
start = time.time()

for i, case in enumerate(cases):
    result = cortex.reason(case["query"], case["evidence"])
    correct = (result.decision == case["expected_decision"])
    fresh_correct += 1 if correct else 0

    # Compare with saved result
    saved_match = saved["results"][i]["actual_decision"] == result.decision
    if not saved_match:
        mismatches += 1

    fresh_results.append({
        "id": case["id"],
        "decision": result.decision,
        "confidence": result.confidence,
        "correct": correct,
    })

elapsed = time.time() - start
fresh_acc = fresh_correct / len(cases) * 100

print(f"\nFresh run: {fresh_correct}/{len(cases)} correct = {fresh_acc:.1f}%")
print(f"Time: {elapsed:.1f}s ({len(cases)/elapsed:.1f} cases/sec)")
print(f"Decision mismatches with saved: {mismatches}/{len(cases)}")

# Per-category verification
cats = {}
for case, fresh in zip(cases, fresh_results):
    cat = case["category"]
    if cat not in cats:
        cats[cat] = {"total": 0, "correct": 0}
    cats[cat]["total"] += 1
    if fresh["correct"]:
        cats[cat]["correct"] += 1

print(f"\n{'Category':<20s} {'Fresh':>8s} {'Saved':>8s}")
print("-" * 40)

# Load saved per-category
saved_cats = {}
for r, case in zip(saved["results"], cases):
    cat = case["category"]
    if cat not in saved_cats:
        saved_cats[cat] = {"total": 0, "correct": 0}
    saved_cats[cat]["total"] += 1
    if r["decision_correct"]:
        saved_cats[cat]["correct"] += 1

for cat in sorted(cats.keys()):
    fresh_pct = cats[cat]["correct"] / cats[cat]["total"] * 100
    saved_pct = saved_cats[cat]["correct"] / saved_cats[cat]["total"] * 100
    match = "OK" if abs(fresh_pct - saved_pct) < 0.1 else "MISMATCH"
    print(f"  {cat:<18s} {fresh_pct:>6.1f}% {saved_pct:>6.1f}%  {match}")

print(f"\nOverall: fresh={fresh_acc:.1f}% saved={saved_correct/len(cases)*100:.1f}%")
if abs(fresh_acc - saved_correct/len(cases)*100) < 0.5:
    print("VERIFIED: Results are genuine and reproducible.")
else:
    print("WARNING: Results differ - possible caching or randomness issue!")
