import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")
from sweep_neural_mesh.neurons.world_knowledge import WorldKnowledge

wk = WorldKnowledge()
print(f"Entities: {wk.entity_count}, Relations: {wk.relation_count}")

# Test the gate
tests = [
    "Birds can talk according to biological classification",
    "Fish are made of metal according to biological classification",
    "Buildings are alive according to biological classification",
    "Cats can fly according to biological classification",
    "Dogs can talk according to biological classification",
]
for t in tests:
    check = wk.check_claim(t)
    print(f"\n  Claim: {t[:60]}")
    print(f"    Plausible: {check.plausible}  Conf: {check.confidence:.2f}")
    print(f"    Contradictions: {check.contradictions[:2]}")
    print(f"    Reasoning: {check.reasoning[:80]}")

# Test entities found
print("\n\nEntities found in 'Birds can talk according to biological classification':")
entities = wk.find_entities_in_text("Birds can talk according to biological classification")
for e in entities:
    print(f"  {e.name}: NOT_abilities={e.NOT_abilities[:3]}")
