import sys
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")
from sweep_neural_mesh.neurons.cortex import ReasoningCortex

cortex = ReasoningCortex(enable_ml=False)

tests = [
    ("Is a tomato a fruit?", ["Botanically, a tomato is a fruit", "Culinarily, a tomato is treated as a vegetable"]),
    ("Is sugar addictive?", ["Sugar activates reward pathways in the brain", "Sugar does not meet clinical criteria for addiction"]),
    ("Can you see the sun at night?", ["The sun is always shining but is only visible during daytime on earth", "During a lunar eclipse the suns shadow is visible"]),
    ("Is Mount Everest the tallest mountain from base to peak?", ["Mauna Kea is taller from base to peak at over 10000 meters", "Mount Everest is 8849 meters from sea level"]),
    ("Can humans breathe underwater?", ["Humans have lungs and cannot extract oxygen from water"]),
]

for q, evs in tests:
    r = cortex.reason(q, evs)
    trace = r.trace
    consensus_data = {}
    for f in trace.factors:
        print(f"  Factor: {f['name']} = {f['score']} ({f['detail']})")
    print(f"Q: {q}")
    print(f"  Decision: {r.decision} (conf={r.confidence:.3f})")
    print(f"  Integration conf: {trace.integration_confidence:.3f}")
    print(f"  Reasoning: {r.reasoning}")
    print()
