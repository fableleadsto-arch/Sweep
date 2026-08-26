import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")
from sweep_neural_mesh.neurons.world_knowledge import WorldKnowledge
wk = WorldKnowledge()
print("Entity 1 exists:", "1" in wk._entities)
print("1 isdigit:", "1".isdigit())
text = "is 1 a prime number"
m = re.search(r"\b" + re.escape("1") + r"\b", text.lower())
print("Regex match:", m)
for name in sorted(wk._entities, key=len, reverse=True)[:5]:
    print(f"  Entity: {name!r} isdigit={name.isdigit()}")
