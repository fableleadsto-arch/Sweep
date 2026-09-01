"""
Continue Training Relay Transformer on new capability domains.

Adds training data for the 7 new capabilities and continues
training the existing relay-small model.
"""
import sys
import os
import json
import time
import logging
import random
from pathlib import Path

_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("relay_continue")


def generate_new_domain_data() -> list[str]:
    """Generate training data for the 7 new capability domains."""
    texts = []
    rng = random.Random(42)

    # ═══ #7 Recursive Investigation ═══
    people = ["John Smith", "Alice Chen", "Bob Wilson", "Carol Davis", "Eve Johnson"]
    orgs = ["TechCorp", "DataInc", "ResearchLab", "MediaGroup", "FinServ"]
    cities = ["Delhi", "London", "Tokyo", "New York", "Berlin", "Paris"]

    for _ in range(300):
        p = rng.choice(people)
        o = rng.choice(orgs)
        c = rng.choice(cities)
        texts.append(f"Investigation: {p} works at {o}. {o} is based in {c}. Entities discovered: {p}, {o}, {c}.")
        texts.append(f"Recursive investigation starting from {p}. Found connection to {o}. {o} located in {c}.")
        texts.append(f"Investigation graph: {p} -> works_at -> {o} -> located_in -> {c}.")
        texts.append(f"Entity {p} is associated with {o}. {o} has employees in {c}.")

    # ═══ #13 Evidence Graph ═══
    for _ in range(300):
        ev_a = rng.choice(["Study confirms X", "Data supports claim Y", "Evidence shows Z"])
        ev_b = rng.choice(["Research agrees with X", "Analysis confirms Y", "Findings support Z"])
        ev_c = rng.choice(["Study contradicts X", "Data refutes claim Y", "Evidence denies Z"])

        texts.append(f"Evidence graph: {ev_a}. {ev_b}. Correlation: CORROBORATES. These evidence nodes form a cluster.")
        texts.append(f"Evidence graph: {ev_a}. {ev_c}. Correlation: CONTRADICTS. These evidence nodes are in conflict.")
        texts.append(f"Building evidence graph from 3 sources. Adding nodes and correlation edges. Graph shows {rng.choice(['corroboration', 'contradiction'])}.")
        texts.append(f"Evidence importance (PageRank): Source A=0.35, Source B=0.32, Source C=0.28. Most important: Source A.")

    # ═══ #21 Location Intelligence ═══
    for _ in range(300):
        loc1 = rng.choice(cities)
        loc2 = rng.choice(cities)
        while loc2 == loc1:
            loc2 = rng.choice(cities)
        texts.append(f"Location intelligence: {loc1} is in {rng.choice(['India', 'UK', 'Japan', 'USA', 'Germany'])}. Coordinates: {rng.uniform(-90,90):.1f}, {rng.uniform(-180,180):.1f}.")
        texts.append(f"Person traveled from {loc1} to {loc2}. Distance approximately {rng.randint(500, 15000)} km. Same region: {loc1 == loc2}.")
        texts.append(f"Geographic analysis: {loc1} and {loc2} are in different regions. Travel between them takes {rng.randint(2, 24)} hours by air.")

    # ═══ #22 Search Strategy ═══
    aspects = ["identity", "location", "affiliation", "timeline", "activities", "associates", "online_presence"]
    for _ in range(300):
        known = rng.sample(aspects, k=rng.randint(1, 4))
        unknown = [a for a in aspects if a not in known]
        texts.append(f"Search strategy: Known aspects: {', '.join(known)}. Unknown aspects: {', '.join(unknown)}. Priority: investigate {unknown[0] if unknown else 'none'}.")
        texts.append(f"Investigation round {rng.randint(1,5)}: {len(known)} aspects known, {len(unknown)} unknown. Confidence coverage: {len(known)/len(aspects):.0%}.")
        texts.append(f"Uncertainty reduction: {rng.choice(known)} confirmed with confidence 0.9. Next search target: {unknown[0] if unknown else 'completed'}.")

    # ═══ #24 Evidence Reports ═══
    for _ in range(300):
        n_sup = rng.randint(2, 8)
        n_con = rng.randint(0, 3)
        total = n_sup + n_con
        independent = rng.randint(1, min(n_sup, 5))
        if n_con == 0 and n_sup >= 4:
            level = "CONFIRMED"
        elif n_sup > n_con and n_sup >= 2:
            level = "LIKELY"
        elif n_sup >= 1:
            level = "POSSIBLE"
        else:
            level = "UNCERTAIN"

        texts.append(f"Investigation report: Target analyzed. Evidence: {total} items, {n_sup} supporting, {n_con} contradicting. Independent sources: {independent}. Confidence: {level}.")
        texts.append(f"Evidence summary: {n_sup} pieces of evidence support the claim. {n_con} contradict it. Overall assessment: {level}. {independent} independent sources consulted.")
        texts.append(f"Report generation: Compiling {total} evidence items. Supporting: {n_sup}. Contradicting: {n_con}. Final confidence level: {level}.")

    # ═══ #26 Deduplication ═══
    for _ in range(300):
        total = rng.randint(3, 10)
        duplicates = rng.randint(1, total - 1)
        unique = total - duplicates
        texts.append(f"Deduplication: {total} items analyzed. Found {duplicates} duplicates. Effective unique evidence: {unique}. Independence ratio: {unique/total:.2f}.")
        texts.append(f"Content dedup: {total} sources checked. SimHash found {duplicates} near-duplicates. Exact matches: {rng.randint(0, duplicates)}. Effective count: {unique}.")
        texts.append(f"Source dedup: {total} pages from {rng.randint(2, total)} domains. Syndicated content: {duplicates} items. Truly independent: {unique}.")

    # ═══ #27 Source Independence ═══
    for _ in range(300):
        n_derived = rng.randint(1, 5)
        origin = rng.choice(["Press Release", "Wire Service", "Government Report", "Company Statement"])
        texts.append(f"Source independence: {origin} is the origin. {n_derived} derived articles found. Effective independent sources: {max(1, 1 + int(n_derived * 0.2))}. Independence score: {max(0.1, 1.0 - n_derived * 0.15):.2f}.")
        texts.append(f"Provenance tracking: Article A derived from {origin}. Article B derived from {origin}. Article C independent. True independent count: 2 (not 3).")
        texts.append(f"Source analysis: {n_derived + 1} sources found. {n_derived} share same origin ({origin}). Independent confirmations: 1. The {n_derived} derived articles do not count as independent evidence.")

    logger.info(f"Generated {len(texts)} training examples for new capability domains")
    return texts


def continue_training():
    """Continue training the Relay Transformer with new domain data."""
    from companion.neural.training.checkpointing import load_model, load_tokenizer
    from companion.neural.training.trainer import train, TrainConfig
    from companion.neural.training.datasets import TextDataset

    checkpoint_dir = str(_sweep_dir / "training" / "relay_small_trained")

    logger.info("=" * 60)
    logger.info("CONTINUING RELAY TRAINING ON NEW DOMAINS")
    logger.info("=" * 60)

    # Load existing model
    logger.info("Loading existing Relay model...")
    model = load_model(checkpoint_dir)
    tokenizer = load_tokenizer(checkpoint_dir)
    logger.info(f"Loaded: {model.param_count():,} params")

    # Generate new data
    texts = generate_new_domain_data()

    # Build dataset
    logger.info("Building dataset...")
    dataset = TextDataset.from_texts(
        texts=texts, tokenizer=tokenizer,
        source="sweep_new_domains",
    )
    logger.info(f"Dataset: {dataset.n_tokens} tokens, {len(texts)} examples")

    # Training config (lower LR for continued training)
    train_config = TrainConfig(
        batch_size=4,
        seq_len=128,
        learning_rate=5e-5,
        warmup_steps=20,
        total_steps=200,
        max_grad_norm=1.0,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_every=50,
        log_every=25,
        seed=42,
        device="cpu",
        dtype="fp32",
    )

    def data_iter():
        while True:
            for batch in dataset.to_dataloader(train_config.batch_size, train_config.seq_len, seed=42):
                yield batch

    logger.info("=" * 60)
    logger.info(f"CONTINUING TRAINING: {train_config.total_steps} steps")
    logger.info(f"Dataset: {len(texts)} examples, {dataset.n_tokens} tokens")
    logger.info("=" * 60)

    losses = []
    def on_log(entry):
        losses.append(entry["loss"])

    t0 = time.perf_counter()
    result = train(
        model=model, tokenizer=tokenizer,
        data_iter=data_iter, config=train_config,
        checkpoint_dir=checkpoint_dir, on_log=on_log,
    )
    elapsed = time.perf_counter() - t0

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Steps: {result.steps_run}")
    logger.info(f"  Final loss: {result.final_loss:.4f}")
    logger.info(f"  Best loss: {result.best_loss:.4f}")
    logger.info(f"  Duration: {elapsed:.1f}s")
    logger.info("=" * 60)

    # Save summary
    summary = {
        "model": "relay-small",
        "parameters": model.param_count(),
        "training_steps": result.steps_run,
        "final_loss": result.final_loss,
        "best_loss": result.best_loss,
        "duration_seconds": elapsed,
        "training_data": len(texts),
        "new_domains": [
            "recursive_investigation",
            "evidence_graph",
            "location_intelligence",
            "search_strategy",
            "evidence_reporting",
            "deduplication",
            "source_independence",
        ],
        "status": "trained",
    }
    summary_path = str(_sweep_dir / "training" / "relay_new_domains_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    return model, result


if __name__ == "__main__":
    continue_training()
