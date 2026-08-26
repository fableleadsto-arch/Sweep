import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")
from sweep_neural_mesh.neurons.cortex import ReasoningCortex
cortex = ReasoningCortex(enable_ml=False)
d = json.load(open(r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII\benchmarks\results\dataset.json", encoding="utf-8"))["cases"]

for cat in ["basic_logic", "novel_structures", "ambiguity"]:
    cases = [c for c in d if c["category"] == cat]
    wrong = 0
    print(f"\n{'='*60}")
    print(f"  {cat.upper()} FAILURES")
    print(f"{'='*60}")
    for c in cases:
        r = cortex.reason(c["query"], c["evidence"])
        if r.decision != c["expected_decision"]:
            wrong += 1
            if wrong <= 10 and c["difficulty"] in ("hard", "medium"):
                print(f"\n  Q: {c['query'][:90]}")
                print(f"  Ev0: {c['evidence'][0][:100]}")
                print(f"  Expected: {c['expected_decision']}  Got: {r.decision} ({r.confidence:.2f})")
                if r.reasoning:
                    print(f"  Reason: {r.reasoning[:140]}")
    print(f"\n  {cat}: {len(cases)} cases, {wrong} wrong, {100-wrong/len(cases)*100:.0f}% acc")
