import sys
sys.path.insert(0, r"C:\Users\mansh\OneDrive\Desktop\MyStuff\Relay.AII")
from sweep_neural_mesh.neurons.centers import EvidenceGatherer
from sweep_neural_mesh.neurons.signal import Signal, SignalType

eg = EvidenceGatherer()

# Test: Is water wet?
raw = Signal(
    data={"query": "Is water wet?", "evidence": ["Water is a liquid that wets surfaces it touches"]},
    signal_type=SignalType.RAW, confidence=1.0,
)
result = eg.process([raw])
for s in result:
    d = s.data
    print(f"Text: {d.get('evidence_text', '')}")
    print(f"Direction: {d.get('support_direction', '?')}")
    print(f"Confidence: {s.confidence:.3f}")
print()

# Test: Do plants need sunlight?
raw2 = Signal(
    data={"query": "Do plants need sunlight?", "evidence": ["Plants use photosynthesis which requires sunlight"]},
    signal_type=SignalType.RAW, confidence=1.0,
)
result2 = eg.process([raw2])
for s in result2:
    d = s.data
    print(f"Text: {d.get('evidence_text', '')}")
    print(f"Direction: {d.get('support_direction', '?')}")
    print(f"Confidence: {s.confidence:.3f}")
print()

# Test direction detection directly
tests = [
    ("Is water wet?", "Water is a liquid that wets surfaces it touches"),
    ("Do plants need sunlight?", "Plants use photosynthesis which requires sunlight"),
    ("Can fish fly?", "Fish swim in water using fins and gills"),
    ("Is the earth flat?", "The earth is an oblate spheroid as confirmed by satellite imagery"),
]
for q, ev in tests:
    direction = eg._detect_support_direction(ev, q)
    print(f"Q: {q} | E: {ev[:50]} | Direction: {direction}")
