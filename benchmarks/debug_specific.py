import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")
from sweep_neural_mesh.neurons.cortex import ReasoningCortex

cortex = ReasoningCortex(enable_ml=False)

# Test specific failing cases
tests = [
    ("Do birds can talk?", ["Birds can talk according to biological classification"], "refuted"),
    ("Do cars are made of metal?", ["Cars are made of metal according to biological classification"], "refuted"),
    ("Is the sun a star?", ["The sun is a G-type main-sequence star at the center of our solar system"], "supported"),
    ("Is sound faster than light in air?", ["Light travels at 300000 km/s while sound travels at 0.34 km/s in air"], "refuted"),
    ("Do fish are heavy?", ["Fish are heavy according to biological classification"], "mixed"),
    ("Do cats conduct electricity?", ["Cats conduct electricity according to biological classification"], "mixed"),
]
for q, ev, expected in tests:
    r = cortex.reason(q, ev)
    ok = "OK" if r.decision == expected else "WRONG"
    print(f"[{ok}] Q: {q}")
    print(f"     Expected: {expected}  Got: {r.decision} ({r.confidence:.2f})")
    if r.reasoning:
        print(f"     Reasoning: {r.reasoning[:120]}")
    print()
