import json
import sys
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")

from sweep_neural_mesh.neurons.cortex import ReasoningCortex

cortex = ReasoningCortex(enable_ml=False)

with open(r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII\benchmarks\results\dataset.json") as f:
    dataset = json.load(f)["cases"]

for category in ["basic_logic", "noisy_input", "novel_structures", "ambiguity"]:
    cases = [c for c in dataset if c["category"] == category]
    wrong = []
    for c in cases:
        result = cortex.reason(c["query"], c["evidence"])
        if result.decision != c["expected_answer"]:
            wrong.append((c["query"], c["evidence"][:2], c["expected_answer"], result.decision, result.confidence))
    print(f"\n=== {category} ({len(wrong)}/{len(cases)} wrong) ===")
    for q, ev, exp, got, conf in wrong[:8]:
        print(f"  Q: {q}")
        print(f"  Expected: {exp}, Got: {got} ({conf:.3f})")
        print(f"  Ev[0]: {ev[0][:100] if ev else 'NONE'}")
        print(f"  Ev[1]: {ev[1][:100] if len(ev) > 1 else 'NONE'}")
        print()
