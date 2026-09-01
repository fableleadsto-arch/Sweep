"""Run RAG + web search + expanded classifiers (no seq2seq)."""
import sys
import os
import json
import time
import random
import logging
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
SWEEP_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(SWEEP_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sweep_rag_classifiers")


def test_rag():
    """Test the RAG pipeline with live Wikipedia."""
    logger.info("=" * 60)
    logger.info("TESTING RAG WITH LIVE WIKIPEDIA RETRIEVAL")
    logger.info("=" * 60)

    try:
        from rag_pipeline import RAGPipeline
        rag = RAGPipeline()
        rag.initialize()

        tests = [
            "What is the capital of France?",
            "What is the boiling point of water?",
            "What is DNA?",
            "When was WWII?",
            "What is the speed of light?",
        ]
        for q in tests:
            result = rag.query(q)
            logger.info(f"  Q: {q}")
            logger.info(f"  A: {result.answer[:120]}...")
            logger.info(f"  Sources: {result.sources[:3]} | Conf: {result.confidence:.2f} | {result.latency_ms:.0f}ms")
            logger.info("")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"RAG test failed: {e}")
        return {"status": "error", "error": str(e)}


def test_web_search():
    """Test the cortex with RAG integration."""
    logger.info("=" * 60)
    logger.info("TESTING WEB SEARCH INTEGRATION")
    logger.info("=" * 60)

    try:
        sys.path.insert(0, str(SWEEP_DIR))
        from cortex_integration import SweepInferencePipeline
        pipeline = SweepInferencePipeline()
        pipeline.initialize()

        tests = [
            "What is the capital of France?",
            "What is the boiling point of water?",
            "Is exercise good for health?",
            "What is DNA?",
            "What is the largest planet?",
            "All cats are animals. Is a cat a living thing?",
            "What year did WWII end?",
            "What is the speed of light?",
        ]
        for q in tests:
            result = pipeline.infer(q)
            logger.info(f"  Q: {q}")
            logger.info(f"  A: {result.answer[:100]}")
            logger.info(f"  Method: {result.method} | Conf: {result.confidence:.2f} | {result.latency_ms:.0f}ms")
            logger.info("")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Web search test failed: {e}")
        return {"status": "error", "error": str(e)}


def train_classifiers():
    """Train expanded classifiers on 2000+ samples per task."""
    logger.info("=" * 60)
    logger.info("TRAINING EXPANDED CLASSIFIERS (2000+ samples)")
    logger.info("=" * 60)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from neurons.semantic_embeddings import SemanticEmbedder

    embedder = SemanticEmbedder()
    # Determine actual vector dimension from a test embed
    test_vec = embedder.embed("test").vector
    embedder_dim = len(test_vec)
    logger.info(f"Embedder dim: {embedder_dim}")

    class Classifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )
        def forward(self, x):
            return self.net(x)

    checkpoint_dir = str(EXPERIMENT_DIR / "checkpoints_expanded")
    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    results = {}

    # === Logic classifier ===
    logger.info("\n--- Logic Classifier ---")
    fillers = ["it rains", "the ground is wet", "cats are animals", "animals are living",
               "A > B", "B > C", "P implies Q", "Q implies R", "X is true", "Y is false",
               "it snows", "roads are icy", "dogs are mammals", "mammals are warm-blooded"]
    valid_templates = [
        "If {a} then {b}. {a} happened.", "All {a} are {b}. This is {a}.",
        "{a} is greater than {b}. {b} is greater than {c}. {a} is greater than {c}.",
        "If {a} implies {b}, and {b} implies {c}, then {a} implies {c}.",
    ]
    invalid_templates = [
        "If {a} then {b}. {b} happened.", "All {a} are {b}. This is {b}.",
        "{a} is greater than {b}. Therefore {b} is greater than {a}.",
        "If {a} then {b}. Therefore if {b} then {a}.",
    ]

    logic_data = []
    for _ in range(1000):
        a, b = random.sample(fillers, 2)
        c = random.choice(fillers)
        for tmpl in valid_templates:
            q = tmpl.format(a=a, b=b, c=c)
            vec = embedder.embed(q).vector
            logic_data.append((vec, 0))
        for tmpl in invalid_templates:
            q = tmpl.format(a=a, b=b, c=c)
            vec = embedder.embed(q).vector
            logic_data.append((vec, 1))

    X = torch.tensor(np.array([d[0] for d in logic_data]), dtype=torch.float32)
    y = torch.tensor([d[1] for d in logic_data], dtype=torch.long)
    perm = torch.randperm(len(X))
    split = int(len(X) * 0.8)

    model_logic = Classifier(embedder_dim, 2)
    optimizer = optim.Adam(model_logic.parameters(), lr=1e-3)

    for epoch in range(30):
        model_logic.train()
        out = model_logic(X[perm[:split]])
        loss = criterion(out, y[perm[:split]])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model_logic.eval()
    with torch.no_grad():
        pred = model_logic(X[perm[split:]]).argmax(dim=1)
        acc = (pred == y[perm[split:]]).float().mean().item()
    logger.info(f"  Accuracy: {acc:.4f} | Samples: {len(logic_data)}")
    torch.save(model_logic.state_dict(), os.path.join(checkpoint_dir, "logic.pt"))
    results["logic"] = {"accuracy": acc, "samples": len(logic_data)}

    # === Math classifier ===
    logger.info("\n--- Math Classifier ---")
    math_data = []
    for _ in range(1500):
        a, b = random.randint(1, 100), random.randint(1, 100)
        op = random.choice(['+', '-', '*'])
        q = f"What is {a} {op} {b}?"
        vec = embedder.embed(q).vector
        math_data.append((vec, 0))
    for _ in range(500):
        nonsense = random.choice(["blue minus happiness", "undefined calculation", "infinity times zero"])
        vec = embedder.embed(f"Calculate {nonsense}.").vector
        math_data.append((vec, 1))

    X_m = torch.tensor(np.array([d[0] for d in math_data]), dtype=torch.float32)
    y_m = torch.tensor([d[1] for d in math_data], dtype=torch.long)
    perm_m = torch.randperm(len(X_m))
    split_m = int(len(X_m) * 0.8)

    model_math = Classifier(embedder_dim, 2)
    optimizer = optim.Adam(model_math.parameters(), lr=1e-3)

    for epoch in range(30):
        model_math.train()
        out = model_math(X_m[perm_m[:split_m]])
        loss = criterion(out, y_m[perm_m[:split_m]])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model_math.eval()
    with torch.no_grad():
        pred = model_math(X_m[perm_m[split_m:]]).argmax(dim=1)
        acc = (pred == y_m[perm_m[split_m:]]).float().mean().item()
    logger.info(f"  Accuracy: {acc:.4f} | Samples: {len(math_data)}")
    torch.save(model_math.state_dict(), os.path.join(checkpoint_dir, "math.pt"))
    results["math"] = {"accuracy": acc, "samples": len(math_data)}

    # === Evidence classifier ===
    logger.info("\n--- Evidence Classifier ---")
    topics = ["exercise", "education", "sleep", "nutrition", "meditation", "reading", "music", "social interaction"]
    outcomes = ["health", "performance", "well-being", "cognitive function", "productivity"]

    ev_data = []
    for _ in range(600):
        t, o = random.choice(topics), random.choice(outcomes)
        vec = embedder.embed(f"Studies show that {t} improves {o}. The evidence supports this claim.").vector
        ev_data.append((vec, 0))  # SUPPORTS
    for _ in range(600):
        t, o = random.choice(topics), random.choice(outcomes)
        vec = embedder.embed(f"Studies show that {t} does NOT improve {o}. Evidence contradicts this claim.").vector
        ev_data.append((vec, 1))  # REFUTES
    for _ in range(600):
        t, o = random.choice(topics), random.choice(outcomes)
        vec = embedder.embed(f"The effect of {t} on {o} is uncertain. More research is needed.").vector
        ev_data.append((vec, 2))  # NEUTRAL

    X_e = torch.tensor(np.array([d[0] for d in ev_data]), dtype=torch.float32)
    y_e = torch.tensor([d[1] for d in ev_data], dtype=torch.long)
    perm_e = torch.randperm(len(X_e))
    split_e = int(len(X_e) * 0.8)

    model_ev = Classifier(embedder_dim, 3)
    optimizer = optim.Adam(model_ev.parameters(), lr=1e-3)

    for epoch in range(30):
        model_ev.train()
        out = model_ev(X_e[perm_e[:split_e]])
        loss = criterion(out, y_e[perm_e[:split_e]])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model_ev.eval()
    with torch.no_grad():
        pred = model_ev(X_e[perm_e[split_e:]]).argmax(dim=1)
        acc = (pred == y_e[perm_e[split_e:]]).float().mean().item()
    logger.info(f"  Accuracy: {acc:.4f} | Samples: {len(ev_data)}")
    torch.save(model_ev.state_dict(), os.path.join(checkpoint_dir, "evidence.pt"))
    results["evidence"] = {"accuracy": acc, "samples": len(ev_data)}

    # === Recognition classifier ===
    logger.info("\n--- Recognition Classifier ---")
    rec_data = []
    entities = ["John Smith", "Paris", "Microsoft", "January 15, 2024", "42 Main Street",
                "john@example.com", "+1-555-0123", "https://example.com"]
    non_entities = ["the weather is nice", "I think therefore I am", "red blue green",
                    "quick brown fox", "hello world test"]

    for _ in range(800):
        ent = random.choice(entities)
        vec = embedder.embed(f"Extract entities from: {ent} visited the office.").vector
        rec_data.append((vec, 0))  # HAS_ENTITY
    for _ in range(800):
        ne = random.choice(non_entities)
        vec = embedder.embed(f"Extract entities from: {ne}.").vector
        rec_data.append((vec, 1))  # NO_ENTITY

    X_r = torch.tensor(np.array([d[0] for d in rec_data]), dtype=torch.float32)
    y_r = torch.tensor([d[1] for d in rec_data], dtype=torch.long)
    perm_r = torch.randperm(len(X_r))
    split_r = int(len(X_r) * 0.8)

    model_rec = Classifier(embedder_dim, 2)
    optimizer = optim.Adam(model_rec.parameters(), lr=1e-3)

    for epoch in range(30):
        model_rec.train()
        out = model_rec(X_r[perm_r[:split_r]])
        loss = criterion(out, y_r[perm_r[:split_r]])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model_rec.eval()
    with torch.no_grad():
        pred = model_rec(X_r[perm_r[split_r:]]).argmax(dim=1)
        acc = (pred == y_r[perm_r[split_r:]]).float().mean().item()
    logger.info(f"  Accuracy: {acc:.4f} | Samples: {len(rec_data)}")
    torch.save(model_rec.state_dict(), os.path.join(checkpoint_dir, "recognition.pt"))
    results["recognition"] = {"accuracy": acc, "samples": len(rec_data)}

    return results


def main():
    logger.info("=" * 70)
    logger.info("SWEEP RAG + CLASSIFIERS TRAINING")
    logger.info("=" * 70)

    all_results = {}
    all_results["rag"] = test_rag()
    all_results["web_search"] = test_web_search()
    all_results["classifiers"] = train_classifiers()

    with open(str(EXPERIMENT_DIR / "rag_classifier_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    for task, data in all_results["classifiers"].items():
        logger.info(f"  {task}: accuracy={data['accuracy']:.4f}, samples={data['samples']}")

    return all_results


if __name__ == "__main__":
    main()
