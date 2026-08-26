import json
import sys
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")
from sweep_neural_mesh.neurons.cortex import ReasoningCortex
from sweep_neural_mesh.neurons.centers import EvidenceGatherer

cortex = ReasoningCortex(enable_ml=False)
eg = EvidenceGatherer()
d = json.load(open("benchmarks/results/dataset.json"))["cases"]

for cat in ["spatial", "ambiguity", "noisy_input", "temporal"]:
    cases = [c for c in d if c["category"] == cat]
    wrong_cases = []
    for c in cases:
        r = cortex.reason(c["query"], c["evidence"])
        if r.decision != c["expected_decision"]:
            dirs = [eg._detect_support_direction(ev, c["query"]) for ev in c["evidence"]]
            wrong_cases.append((c["query"], c["expected_decision"], r.decision, dirs, c["evidence"]))

    print(f"=== {cat} ({len(wrong_cases)}/{len(cases)} wrong) ===")
    for q, exp, got, dirs, evs in wrong_cases[:5]:
        print(f"  Q: {q}")
        print(f"  Expected: {exp}, Got: {got}")
        for i, (ev, d) in enumerate(zip(evs, dirs)):
            print(f"  Ev[{i}] dir={d}: {ev[:80]}")
        print()
