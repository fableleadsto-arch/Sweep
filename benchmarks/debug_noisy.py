"""
Analyze noisy_input failures.
"""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")
from sweep_neural_mesh.neurons.cortex import ReasoningCortex

cortex = ReasoningCortex(enable_ml=False)
d = json.load(open(r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII\benchmarks\results\dataset.json", encoding="utf-8"))["cases"]

cases = [c for c in d if c["category"] == "noisy_input"]
wrong = 0
for c in cases:
    r = cortex.reason(c["query"], c["evidence"])
    if r.decision != c["expected_decision"]:
        wrong += 1
        if wrong <= 15:
            print(f"  Q: {c['query'][:80]}")
            print(f"  Ev: {c['evidence'][0][:80]}")
            print(f"  Expected: {c['expected_decision']}  Got: {r.decision} ({r.confidence:.2f})")
            print()

print(f"noisy_input: {len(cases)} cases, {wrong} wrong, {100-wrong/len(cases)*100:.0f}% acc")
