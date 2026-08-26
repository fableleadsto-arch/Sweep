import json
import sys
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")
from sweep_neural_mesh.neurons.cortex import ReasoningCortex

cortex = ReasoningCortex(enable_ml=False)
d = json.load(open("benchmarks/results/dataset.json"))["cases"]

cases = [c for c in d if c["category"] == "ambiguity"]
wrong = []
for c in cases:
    r = cortex.reason(c["query"], c["evidence"])
    if r.decision != c["expected_decision"]:
        wrong.append((c["query"], c["evidence"], c["expected_decision"], r.decision, r.confidence))

print(f"Ambiguity: {len(cases) - len(wrong)}/{len(cases)} correct")
print(f"Wrong: {len(wrong)}")
print()

# Show 10 mixed->supported failures
mixed_to_sup = [(q, ev, exp, got, conf) for q, ev, exp, got, conf in wrong if exp == "mixed" and got == "supported"]
print(f"mixed -> supported: {len(mixed_to_sup)}")
for q, ev, exp, got, conf in mixed_to_sup[:5]:
    print(f"  Q: {q}")
    for i, e in enumerate(ev):
        print(f"    Ev[{i}]: {e[:80]}")
    print()

# Show 10 supported->insufficient failures
sup_to_ins = [(q, ev, exp, got, conf) for q, ev, exp, got, conf in wrong if exp == "supported" and got == "insufficient"]
print(f"supported -> insufficient: {len(sup_to_ins)}")
for q, ev, exp, got, conf in sup_to_ins[:5]:
    print(f"  Q: {q}")
    for i, e in enumerate(ev):
        print(f"    Ev[{i}]: {e[:80]}")
    print()
