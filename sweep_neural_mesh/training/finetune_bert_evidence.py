"""
BERT Fine-Tuning for Evidence Classification.

Trains BERT to classify evidence as:
  0 = SUPPORTS (confirms the claim)
  1 = REFUTES (contradicts the claim)
  2 = NEUTRAL (mixed or insufficient)

Uses Sweep's investigation-specific training data.
"""
import sys
import os
import json
import time
import logging
from pathlib import Path

_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bert_finetune")

OUTPUT_DIR = str(_sweep_dir / "training" / "bert_evidence_finetuned")


def create_training_data() -> list[tuple[str, int]]:
    """Create Sweep-specific evidence classification training data."""
    data = []

    # ── SUPPORTS (label=0) ──
    supports = [
        ("Studies confirm that exercise improves cardiovascular health", 0),
        ("Research shows vaccines are safe and effective in preventing disease", 0),
        ("Multiple clinical trials demonstrate the drug reduces symptoms by 40%", 0),
        ("Data indicates the treatment is beneficial for patients", 0),
        ("Evidence supports the claim that reading helps learning", 0),
        ("Scientists agree that climate change is real and human-caused", 0),
        ("The study found significant positive effects on test scores", 0),
        ("Meta-analysis confirms the intervention works across populations", 0),
        ("Peer-reviewed research validates the hypothesis", 0),
        ("Clinical trials show the medication is effective in 80% of patients", 0),
        ("The experiment demonstrated clear benefits for participants", 0),
        ("Statistical analysis confirms the improvement is significant", 0),
        ("Expert consensus supports this conclusion based on decades of research", 0),
        ("The evidence overwhelmingly supports the claim", 0),
        ("Longitudinal studies confirm the relationship over 10 years", 0),
        ("Exercise was correlated with improved cardiovascular health", 0),
        ("The vaccine prevented infection in 95% of participants", 0),
        ("Meditation was shown to reduce anxiety levels significantly", 0),
        ("The education program improved test scores by 20%", 0),
        ("Official records confirm the person was present at the event", 0),
        ("DNA evidence matches the suspect with 99.9% confidence", 0),
        ("Witness testimony corroborates the video evidence", 0),
        ("Multiple independent sources confirm the same facts", 0),
        ("The timeline is consistent across all evidence", 0),
        ("Forensic analysis supports the conclusion", 0),
        ("Photographic evidence shows the person at the location", 0),
        ("Satellite imagery confirms the building exists", 0),
        ("Financial records support the claim of profitability", 0),
        ("Employee records confirm the person worked at the company", 0),
        ("Travel records show the person was in the city on that date", 0),
        ("Social media posts confirm the person attended the event", 0),
        ("Public documents verify the organization's registration", 0),
        ("Academic transcripts confirm the degree was earned", 0),
        ("Medical records support the diagnosis", 0),
        ("Police reports confirm the incident occurred", 0),
        ("Tax records verify the income claim", 0),
        ("Property records confirm ownership", 0),
        ("Court documents support the legal claim", 0),
        ("Patent records confirm the invention", 0),
        ("Insurance records support the damage assessment", 0),
    ]

    # ── REFUTES (label=1) ──
    refutes = [
        ("The drug shows no significant effect compared to placebo", 1),
        ("Studies contradict the claim that the treatment works", 1),
        ("Research found no evidence supporting the hypothesis", 1),
        ("The experiment failed to demonstrate any benefit", 1),
        ("Data shows the intervention had no measurable impact", 1),
        ("Meta-analysis found no significant effect across studies", 1),
        ("The treatment group showed no improvement over control", 1),
        ("Evidence does not support the claimed benefits", 1),
        ("Clinical trials showed the drug was ineffective", 1),
        ("The study found no correlation between the variables", 1),
        ("Results were not statistically significant", 1),
        ("The hypothesis was not supported by the data", 1),
        ("No measurable difference was found between groups", 1),
        ("The intervention showed no positive outcomes", 1),
        ("Research failed to replicate previous findings", 1),
        ("The vaccine did not reduce infection rates in the trial", 1),
        ("Exercise showed no improvement in cognitive function", 1),
        ("The supplement had no effect on performance", 1),
        ("Sleep deprivation did not affect the measured outcomes", 1),
        ("The diet showed no weight loss benefits", 1),
        ("Travel records show the person was in a different city", 1),
        ("The meeting time stated contradicts the schedule", 1),
        ("Revenue decreased 15% contradicting the growth claim", 1),
        ("The Earth is flat contradicts scientific consensus", 1),
        ("Sound travels faster than light is physically impossible", 1),
        ("The person was confirmed to be elsewhere at the time", 1),
        ("Document forgery was detected in the evidence", 1),
        ("The timestamp on the photo was manipulated", 1),
        ("Witness alibi contradicts the accusation", 1),
        ("Financial audit reveals the company reported losses", 1),
        ("The claim is refuted by official government data", 1),
        ("Satellite imagery shows the building was demolished", 1),
        ("The signature was determined to be forged", 1),
        ("DNA evidence does not match the suspect", 1),
        ("The person was proven to be in a different country", 1),
        ("Carbon dating shows the artifact is modern, not ancient", 1),
        ("The video was deepfake generated by AI", 1),
        ("Audio analysis reveals the recording was edited", 1),
        ("The document metadata shows it was created after the claimed date", 1),
        ("Cross-referencing sources reveals the story is fabricated", 1),
    ]

    # ── NEUTRAL (label=2) ──
    neutral = [
        ("The study had mixed results across different populations", 2),
        ("Some participants improved while others did not", 2),
        ("The effect size was small and borderline significant", 2),
        ("Results varied depending on the dosage used", 2),
        ("The findings are preliminary and need more research", 2),
        ("Evidence is insufficient to draw a definitive conclusion", 2),
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
        ("The effect size varies considerably between trials", 2),
        ("Some evidence supports while other evidence contradicts", 2),
        ("The findings are suggestive but not conclusive", 2),
        ("Additional studies are warranted before conclusions", 2),
        ("The person's location is uncertain based on available evidence", 2),
        ("Timeline cannot be established with current information", 2),
        ("The claim is plausible but unverified", 2),
        ("Insufficient data to determine the relationship", 2),
        ("The evidence is circumstantial", 2),
        ("Results are pending peer review", 2),
        ("The sample size was too small for definitive conclusions", 2),
        ("Confounding variables may have influenced the results", 2),
        ("The study design has limitations that affect interpretation", 2),
        ("Long-term effects are unknown", 2),
        ("The mechanism of action is not yet understood", 2),
        ("Different measuring methods produced different results", 2),
        ("The claim is possible but not confirmed", 2),
        ("Available evidence is contradictory and needs resolution", 2),
        ("The findings are preliminary from a pilot study", 2),
        ("Replication studies have not yet been conducted", 2),
        ("The effect may be context-dependent", 2),
        ("Confidence interval includes zero", 2),
        ("The result could be due to chance", 2),
        ("Further investigation is required", 2),
    ]

    data.extend(supports)
    data.extend(refutes)
    data.extend(neutral)

    logger.info(f"Created {len(data)} training examples: "
                f"{sum(1 for _, l in data if l == 0)} supports, "
                f"{sum(1 for _, l in data if l == 1)} refutes, "
                f"{sum(1 for _, l in data if l == 2)} neutral")

    return data


def train():
    """Fine-tune BERT for evidence classification."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    try:
        from transformers import BertTokenizer, BertForSequenceClassification
    except ImportError:
        logger.error("transformers not available. Install with: pip install transformers")
        return

    logger.info("=" * 60)
    logger.info("BERT FINE-TUNING FOR EVIDENCE CLASSIFICATION")
    logger.info("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create training data
    train_data = create_training_data()

    # Load tokenizer and model
    logger.info("Loading BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    )

    # Tokenize
    texts = [d[0] for d in train_data]
    labels = [d[1] for d in train_data]

    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(
        encodings["input_ids"], encodings["attention_mask"], labels_tensor
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    t0 = time.perf_counter()
    total_loss = 0
    steps = 0

    logger.info(f"Training on {len(train_data)} examples for 10 epochs...")

    for epoch in range(10):
        epoch_loss = 0
        correct = 0
        total = 0

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

            preds = outputs.logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)

        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total
        logger.info(f"  Epoch {epoch+1}/10: loss={avg_loss:.4f}, accuracy={accuracy:.1%}")

    elapsed = time.perf_counter() - t0
    avg_loss = total_loss / steps

    # Save model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save metadata
    metadata = {
        "model": "bert-base-uncased-finetuned",
        "task": "evidence_classification",
        "labels": ["supports", "refutes", "neutral"],
        "training_examples": len(train_data),
        "epochs": 10,
        "final_loss": avg_loss,
        "final_accuracy": accuracy,
        "duration_seconds": elapsed,
        "status": "fine-tuned",
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nTraining complete: loss={avg_loss:.4f}, accuracy={accuracy:.1%}")
    logger.info(f"Duration: {elapsed:.1f}s")
    logger.info(f"Model saved to {OUTPUT_DIR}")

    # Run evaluation
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    model.eval()
    label_map = {0: "supports", 1: "refutes", 2: "neutral"}

    test_cases = [
        ("Studies confirm the drug is effective", "supports"),
        ("The drug shows no significant effect", "refutes"),
        ("Results were mixed across populations", "neutral"),
        ("Research supports the hypothesis", "supports"),
        ("The experiment failed to show benefit", "refutes"),
        ("More research is needed", "neutral"),
        ("The evidence overwhelmingly supports the claim", "supports"),
        ("No measurable difference was found", "refutes"),
        ("Findings are preliminary and need more research", "neutral"),
        ("Travel records show the person was in a different city", "refutes"),
        ("DNA evidence matches the suspect", "supports"),
        ("The timeline cannot be established", "neutral"),
        ("Official documents confirm the registration", "supports"),
        ("The signature was forged", "refutes"),
        ("The claim is plausible but unverified", "neutral"),
    ]

    correct = 0
    for text, expected in test_cases:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)
        pred_label = label_map[outputs.logits.argmax(dim=1).item()]
        match = pred_label == expected
        if match:
            correct += 1
        status = "PASS" if match else "FAIL"
        logger.info(f"  {status}: '{text[:50]}...' -> {pred_label} (expected: {expected})")

    eval_accuracy = correct / len(test_cases)
    logger.info(f"\nEvaluation: {correct}/{len(test_cases)} ({eval_accuracy:.1%})")

    return model, tokenizer


if __name__ == "__main__":
    train()
