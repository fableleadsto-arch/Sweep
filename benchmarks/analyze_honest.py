import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")
from sweep_neural_mesh.neurons.cortex import ReasoningCortex

cortex = ReasoningCortex(enable_ml=False)
d = json.load(open("benchmarks/results/dataset.json", encoding="utf-8"))["cases"]

for cat in ["basic_logic", "novel_structures", "ambiguity"]:
    cases = [c for c in d if c["category"] == cat]
    print(f"\n{'='*60}")
    print(f"  {cat.upper()} -- what GPT-4o gets right that we don't")
    print(f"{'='*60}")

    for c in cases:
        r = cortex.reason(c["query"], c["evidence"])
        if r.decision != c["expected_decision"]:
            if c["difficulty"] == "hard":
                print(f"\n  Q: {c['query']}")
                print(f"  Expected: {c['expected_decision']}")
                print(f"  Got: {r.decision} ({r.confidence:.2f})")
                for i, e in enumerate(c["evidence"][:2]):
                    print(f"  Ev[{i}]: {e[:100]}")
