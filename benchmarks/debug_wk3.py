import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")
from sweep_neural_mesh.neurons.world_knowledge import WorldKnowledge
wk = WorldKnowledge()

tests = [
    "Is 1 a prime number",
    "1 is a prime number",
    "Are cats made of metal?",
    "Cats are made of metal according to biological classification",
    "There are estimated to be about 7.5 times 10^18 grains of sand on earth",
    "Is the concept of justice real?",
    "Do buildings are alive?",
    "Do cars are made of metal",
]
for t in tests:
    c = wk.check_claim(t)
    print(f"{t[:60]:60s} -> plausible={c.plausible} conf={c.confidence:.2f}")
    if c.contradictions:
        print(f"    contradictions: {c.contradictions[:3]}")
