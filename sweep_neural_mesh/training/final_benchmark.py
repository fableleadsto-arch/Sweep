"""Final benchmark with trained Relay Transformer + pretrained MiniLM."""
import sys, os, json, time, logging
from pathlib import Path

_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("final_bench")

print("=" * 70)
print("SWEEP NEURAL ENGINE — FINAL BENCHMARK")
print("=" * 70)

import torch
from companion.neural.training.checkpointing import load_model, load_tokenizer

# ══════════════════════════════════════════════════════════════════
# Load trained models
# ══════════════════════════════════════════════════════════════════

relay_dir = str(_sweep_dir / "training" / "relay_small_trained")
print(f"\n[1/4] Loading trained Relay Transformer...")
relay_model = load_model(relay_dir)
relay_tokenizer = load_tokenizer(relay_dir)
relay_model.eval()
print(f"   Model: {relay_model.param_count():,} params")

# Load MiniLM
print(f"\n[2/4] Loading pretrained MiniLM embeddings...")
from neurons.semantic_embeddings import SemanticEmbedder
embedder = SemanticEmbedder()
print(f"   Backend: {embedder.backend}, dim: {embedder.embedding_dim}")

# ══════════════════════════════════════════════════════════════════
# Test 1: Relay Transformer — token prediction
# ══════════════════════════════════════════════════════════════════

print(f"\n[3/4] Testing Relay Transformer...")

relay_tasks = [
    ("What is the capital of France?", ["Paris", "paris"]),
    ("What is the chemical formula for water?", ["H2O", "h2o"]),
    ("How many planets are in the solar system?", ["8", "eight"]),
    ("What is the speed of light?", ["299792458", "299,792,458"]),
    ("What is the largest ocean?", ["Pacific", "pacific"]),
    ("What year did World War II end?", ["1945"]),
    ("What is photosynthesis?", ["energy", "sunlight", "light"]),
    ("What is DNA?", ["genetic", "genetic information"]),
    ("What is 15% of 200?", ["30"]),
    ("What is the boiling point of water?", ["100"]),
]

relay_correct = 0
for q, expected_keywords in relay_tasks:
    prompt = f"Question: {q} Answer:"
    tokens = relay_tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        logits, _ = relay_model(input_ids)
    top_tokens = torch.topk(logits[0, -1, :], k=15).indices.tolist()
    predictions = [relay_tokenizer.decode([t]).lower() for t in top_tokens]
    found = any(any(kw.lower() in p for p in predictions) for kw in expected_keywords)
    if found:
        relay_correct += 1
    logger.info(f"  {'PASS' if found else 'FAIL'}: {q[:45]}... top={[p[:12] for p in predictions[:5]]}")

relay_acc = relay_correct / len(relay_tasks)
print(f"   Relay Transformer: {relay_correct}/{len(relay_tasks)} ({relay_acc:.1%})")

# ══════════════════════════════════════════════════════════════════
# Test 2: MiniLM — semantic similarity
# ══════════════════════════════════════════════════════════════════

print(f"\n[4/4] Testing MiniLM semantic similarity...")

similarity_tests = [
    ("The drug reduces symptoms", "The medication alleviates signs", 0.7, True),
    ("It is raining outside", "The weather is sunny and dry", 0.3, False),
    ("Exercise improves health", "Physical activity benefits wellness", 0.7, True),
    ("The Earth orbits the Sun", "Planets revolve around stars", 0.5, True),
    ("Cats are mammals", "Dogs are reptiles", 0.3, False),
]

sim_correct = 0
for text_a, text_b, threshold, expected_related in similarity_tests:
    result = embedder.similarity(text_a, text_b)
    is_related = result.score >= threshold
    found = is_related == expected_related
    if found:
        sim_correct += 1
    logger.info(f"  {'PASS' if found else 'FAIL'}: sim={result.score:.3f} (threshold={threshold}, expected_related={expected_related})")

sim_acc = sim_correct / len(similarity_tests)
print(f"   Semantic similarity: {sim_correct}/{len(similarity_tests)} ({sim_acc:.1%})")

# ══════════════════════════════════════════════════════════════════
# Test 3: Cortex integration
# ══════════════════════════════════════════════════════════════════

print(f"\n[3/4] Testing Cortex integration...")

from neurons.cortex import ReasoningCortex
cortex = ReasoningCortex(enable_ml=True)

cortex_tests = [
    ("What is the capital of France?", [], "supported"),
    ("Is exercise good for health?", ["Exercise reduces heart disease risk"], "supported"),
    ("What is 2 + 2?", [], "supported"),
    ("What is the boiling point of water?", [], "supported"),
    ("Is the drug effective?", ["The drug reduces symptoms by 40%", "The drug shows no significant effect"], "mixed"),
    ("What is the population of the Moon?", [], "insufficient"),
]

cortex_correct = 0
for q, evidence, expected in cortex_tests:
    t0 = time.perf_counter()
    result = cortex.reason(query=q, evidence=evidence)
    latency = (time.perf_counter() - t0) * 1000
    found = result.decision == expected
    if found:
        cortex_correct += 1
    logger.info(f"  {'PASS' if found else 'FAIL'}: {q[:40]}... -> {result.decision} (expected: {expected}, {latency:.0f}ms)")

cortex_acc = cortex_correct / len(cortex_tests)
print(f"   Cortex integration: {cortex_correct}/{len(cortex_tests)} ({cortex_acc:.1%})")

# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"  Relay Transformer (trained):  {relay_correct}/{len(relay_tasks)} ({relay_acc:.1%})")
print(f"  MiniLM embeddings:            {sim_correct}/{len(similarity_tests)} ({sim_acc:.1%})")
print(f"  Cortex integration:           {cortex_correct}/{len(cortex_tests)} ({cortex_acc:.1%})")
print(f"  Total:                        {relay_correct + sim_correct + cortex_correct}/{len(relay_tasks) + len(similarity_tests) + len(cortex_tests)}")
print("=" * 70)

# Save results
results = {
    "relay_transformer": {"correct": relay_correct, "total": len(relay_tasks), "accuracy": relay_acc},
    "minilm_embeddings": {"correct": sim_correct, "total": len(similarity_tests), "accuracy": sim_acc},
    "cortex_integration": {"correct": cortex_correct, "total": len(cortex_tests), "accuracy": cortex_acc},
    "model_parameters": relay_model.param_count(),
    "training_steps": 200,
}
results_path = str(_sweep_dir / "training" / "final_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")
