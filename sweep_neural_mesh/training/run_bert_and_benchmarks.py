"""Fine-tune BERT and run benchmarks."""
import sys, os, json, time, logging, random
from pathlib import Path

_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bert_bench")

# ══════════════════════════════════════════════════════════════════
# STEP 1: Fine-tune BERT
# ══════════════════════════════════════════════════════════════════

def finetune_bert(output_dir: str):
    """Fine-tune BERT for evidence classification."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    os.makedirs(output_dir, exist_ok=True)

    from transformers import BertTokenizer, BertForSequenceClassification

    logger.info("Loading BERT...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    # Training data: (text, label) where 0=supports, 1=refutes, 2=neutral
    training_pairs = [
        # Supports (label=0)
        ("Studies confirm that exercise improves health", 0),
        ("Research shows vaccines are safe and effective", 0),
        ("Multiple trials demonstrate the drug reduces symptoms", 0),
        ("Data indicates the treatment is beneficial", 0),
        ("Evidence supports the claim that reading helps learning", 0),
        ("Scientists agree that climate change is real", 0),
        ("The study found significant positive effects", 0),
        ("Meta-analysis confirms the intervention works", 0),
        ("Peer-reviewed research validates the hypothesis", 0),
        ("Clinical trials show the medication is effective", 0),
        ("The experiment demonstrated clear benefits", 0),
        ("Statistical analysis confirms the improvement", 0),
        ("Expert consensus supports this conclusion", 0),
        ("The evidence overwhelmingly supports the claim", 0),
        ("Longitudinal studies confirm the relationship", 0),
        ("The drug was shown to be effective in 80 percent of patients", 0),
        ("Exercise was correlated with improved cardiovascular health", 0),
        ("The vaccine prevented infection in 95 percent of participants", 0),
        ("Meditation was shown to reduce anxiety levels", 0),
        ("The education program improved test scores by 20 percent", 0),
        ("Water is essential for human survival and health", 0),
        ("Sleep deprivation negatively impacts cognitive function", 0),
        ("Regular physical activity reduces the risk of heart disease", 0),
        ("Proper nutrition is important for growth and development", 0),
        ("Education increases employment opportunities", 0),

        # Refutes (label=1)
        ("The drug shows no significant effect compared to placebo", 1),
        ("Studies contradict the claim that the treatment works", 1),
        ("Research found no evidence supporting the hypothesis", 1),
        ("The experiment failed to demonstrate any benefit", 1),
        ("Data shows the intervention had no measurable impact", 1),
        ("Meta-analysis found no significant effect", 1),
        ("The treatment group showed no improvement over control", 1),
        ("Evidence does not support the claimed benefits", 1),
        ("Clinical trials showed the drug was ineffective", 1),
        ("The study found no correlation between the variables", 1),
        ("Results were not statistically significant", 1),
        ("The hypothesis was not supported by the data", 1),
        ("No measurable difference was found between groups", 1),
        ("The intervention showed no positive outcomes", 1),
        ("Research failed to replicate previous findings", 1),
        ("The vaccine did not reduce infection rates", 1),
        ("Exercise showed no improvement in cognitive function", 1),
        ("The supplement had no effect on performance", 1),
        ("Sleep deprivation did not affect the measured outcomes", 1),
        ("The diet showed no weight loss benefits", 1),
        ("The drug caused more harm than benefit", 1),
        ("The treatment worsened patient outcomes", 1),
        ("No improvement was observed in the intervention group", 1),
        ("The study concluded the treatment was not effective", 1),
        ("Results contradicted the initial hypothesis", 1),

        # Neutral (label=2)
        ("The study had mixed results across different populations", 2),
        ("Some participants improved while others did not", 2),
        ("The effect size was small and borderline significant", 2),
        ("Results varied depending on the dosage used", 2),
        ("The findings are preliminary and need more research", 2),
        ("Evidence is insufficient to draw a conclusion", 2),
        ("The results were inconclusive", 2),
        ("More research is needed to confirm these findings", 2),
        ("The effect was observed in some conditions but not others", 2),
        ("Results were inconsistent across studies", 2),
        ("The intervention showed variable effects", 2),
        ("Data quality limits the strength of conclusions", 2),
        ("The relationship is complex and not fully understood", 2),
        ("Findings depend on the specific population studied", 2),
        ("The evidence is mixed and inconclusive", 2),
        ("Results were contradictory across different studies", 2),
        ("The effect size varies considerably", 2),
        ("Some evidence supports while other evidence contradicts", 2),
        ("The findings are suggestive but not conclusive", 2),
        ("Additional studies are warranted", 2),
        ("The results are context-dependent", 2),
        ("More data is needed to reach a definitive conclusion", 2),
        ("The outcome depends on multiple factors", 2),
        ("Further investigation is required", 2),
        ("The evidence is equivocal", 2),
    ]

    logger.info(f"Fine-tuning BERT on {len(training_pairs)} examples...")
    texts = [p[0] for p in training_pairs]
    labels = [p[1] for p in training_pairs]

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels_tensor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    t0 = time.perf_counter()
    total_loss = 0
    steps = 0

    for epoch in range(5):
        epoch_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, batch_labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_loss += loss.item()
            steps += 1
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"  Epoch {epoch+1}/5: loss={avg_loss:.4f}")

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / steps

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "model": "bert-base-uncased-finetuned",
        "task": "evidence_classification",
        "labels": ["supports", "refutes", "neutral"],
        "training_examples": len(training_pairs),
        "epochs": 5,
        "final_loss": avg_loss,
        "duration_seconds": elapsed,
        "status": "fine-tuned",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"BERT fine-tuning complete: loss={avg_loss:.4f}, duration={elapsed:.1f}s")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════
# STEP 2: Run benchmarks
# ══════════════════════════════════════════════════════════════════

def run_benchmarks(bert_dir: str):
    """Run comprehensive benchmarks."""
    import torch
    from companion.neural.training.checkpointing import load_model, load_tokenizer

    logger.info("=" * 60)
    logger.info("RUNNING BENCHMARKS")
    logger.info("=" * 60)

    # Load trained models
    relay_dir = str(_sweep_dir / "training" / "relay_small_trained")
    relay_model = load_model(relay_dir)
    relay_tokenizer = load_tokenizer(relay_dir)
    relay_model.eval()

    # Load fine-tuned BERT
    from transformers import BertTokenizer, BertForSequenceClassification
    bert_tokenizer = BertTokenizer.from_pretrained(bert_dir)
    bert_model = BertForSequenceClassification.from_pretrained(bert_dir)
    bert_model.eval()
    logger.info("Loaded both models")

    # ── Test 1: Relay Transformer factual QA ──
    logger.info("\n--- Test 1: Relay Transformer Factual QA ---")
    relay_tasks = [
        ("What is the capital of France?", "Paris"),
        ("What is the chemical formula for water?", "H2O"),
        ("How many planets are in the solar system?", "8"),
        ("What is the speed of light?", "299792458"),
        ("What is the largest ocean?", "Pacific"),
        ("What year did World War II end?", "1945"),
    ]
    relay_correct = 0
    for q, expected in relay_tasks:
        prompt = f"Question: {q} Answer:"
        tokens = relay_tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            logits, _ = relay_model(input_ids)
        top_tokens = torch.topk(logits[0, -1, :], k=10).indices.tolist()
        predictions = [relay_tokenizer.decode([t]).lower() for t in top_tokens]
        found = any(expected.lower() in p for p in predictions)
        if found:
            relay_correct += 1
        logger.info(f"  {'PASS' if found else 'FAIL'}: {q[:40]}... expected={expected}, top={predictions[:3]}")
    logger.info(f"Relay Transformer: {relay_correct}/{len(relay_tasks)} ({relay_correct/len(relay_tasks):.1%})")

    # ── Test 2: Fine-tuned BERT evidence classification ──
    logger.info("\n--- Test 2: Fine-tuned BERT Evidence Classification ---")
    label_map = {0: "supports", 1: "refutes", 2: "neutral"}
    bert_test = [
        ("Studies confirm the drug is effective", "supports"),
        ("The drug shows no significant effect", "refutes"),
        ("Results were mixed across populations", "neutral"),
        ("Research supports the hypothesis", "supports"),
        ("The experiment failed to show benefit", "refutes"),
        ("More research is needed", "neutral"),
        ("The evidence overwhelmingly supports the claim", "supports"),
        ("No measurable difference was found", "refutes"),
        ("Findings are preliminary and need more research", "neutral"),
        ("Clinical trials demonstrate positive outcomes", "supports"),
        ("The treatment was ineffective in all trials", "refutes"),
        ("Results varied depending on the population", "neutral"),
        ("Multiple studies confirm the relationship", "supports"),
        ("The intervention showed no improvement", "refutes"),
        ("Additional research is warranted", "neutral"),
    ]
    bert_correct = 0
    for text, expected in bert_test:
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        pred_label = label_map[outputs.logits.argmax(dim=1).item()]
        found = pred_label == expected
        if found:
            bert_correct += 1
        logger.info(f"  {'PASS' if found else 'FAIL'}: '{text[:50]}...' -> {pred_label} (expected: {expected})")
    logger.info(f"Fine-tuned BERT: {bert_correct}/{len(bert_test)} ({bert_correct/len(bert_test):.1%})")

    # ── Test 3: Cortex integration ──
    logger.info("\n--- Test 3: Cortex with Pretrained Models ---")
    from neurons.cortex import ReasoningCortex
    cortex = ReasoningCortex(enable_ml=True)

    cortex_tests = [
        ("What is the capital of France?", [], "supported"),
        ("Is exercise good for health?", ["Exercise reduces heart disease risk"], "supported"),
        ("Is the drug effective?", ["The drug reduces symptoms by 40%", "The drug shows no significant effect"], "mixed"),
        ("What is the population of the Moon?", [], "insufficient"),
    ]
    cortex_correct = 0
    for q, evidence, expected_decision in cortex_tests:
        t0 = time.perf_counter()
        result = cortex.reason(query=q, evidence=evidence)
        latency = (time.perf_counter() - t0) * 1000
        found = result.decision == expected_decision
        if found:
            cortex_correct += 1
        logger.info(f"  {'PASS' if found else 'FAIL'}: '{q[:40]}...' -> {result.decision} (expected: {expected_decision}, {latency:.0f}ms)")
    logger.info(f"Cortex integration: {cortex_correct}/{len(cortex_tests)} ({cortex_correct/len(cortex_tests):.1%})")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Relay Transformer:     {relay_correct}/{len(relay_tasks)} ({relay_correct/len(relay_tasks):.1%})")
    logger.info(f"  Fine-tuned BERT:       {bert_correct}/{len(bert_test)} ({bert_correct/len(bert_test):.1%})")
    logger.info(f"  Cortex integration:    {cortex_correct}/{len(cortex_tests)} ({cortex_correct/len(cortex_tests):.1%})")
    logger.info("=" * 60)

    return {
        "relay": {"correct": relay_correct, "total": len(relay_tasks)},
        "bert": {"correct": bert_correct, "total": len(bert_test)},
        "cortex": {"correct": cortex_correct, "total": len(cortex_tests)},
    }


if __name__ == "__main__":
    bert_dir = str(_sweep_dir / "training" / "bert_evidence_finetuned")

    logger.info("=" * 70)
    logger.info("SWEEP — BERT FINE-TUNING + BENCHMARKS")
    logger.info("=" * 70)

    # Fine-tune BERT
    finetune_bert(bert_dir)

    # Run benchmarks
    results = run_benchmarks(bert_dir)

    # Save
    summary_path = str(_sweep_dir / "training" / "v2_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {summary_path}")
