"""
Real Neural Training — Train the Relay Transformer on actual data.

This script:
1. Creates a training dataset from Sweep's knowledge base
2. Trains the Relay Transformer (forward/backward/gradient update)
3. Saves a checkpoint
4. Provides the trained model for inference

This is NOT fake training. Every step involves real gradient computation.
"""
import sys
import os
import json
import time
import logging
from pathlib import Path

# Setup paths — run from sweep/ root so companion.neural imports work
_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("real_training")

# ══════════════════════════════════════════════════════════════════
# STEP 1: Create training dataset from Sweep's knowledge
# ══════════════════════════════════════════════════════════════════

def create_training_data() -> list[str]:
    """
    Create a training corpus from Sweep's knowledge bases.
    Each line is a text passage the model learns to predict.
    """
    texts = []

    # ── Factual QA pairs ──
    factual_qa = [
        "Question: What is the capital of France? Answer: Paris is the capital of France.",
        "Question: What is the chemical formula for water? Answer: The chemical formula for water is H2O.",
        "Question: How many planets are in the solar system? Answer: There are 8 planets in the solar system.",
        "Question: What is the speed of light? Answer: The speed of light is approximately 299,792,458 meters per second.",
        "Question: What is the largest ocean? Answer: The Pacific Ocean is the largest ocean on Earth.",
        "Question: What year did World War II end? Answer: World War II ended in 1945.",
        "Question: What is photosynthesis? Answer: Photosynthesis is the process by which plants convert sunlight into energy.",
        "Question: What is DNA? Answer: DNA is deoxyribonucleic acid, which carries genetic information.",
        "Question: What is the boiling point of water? Answer: Water boils at 100 degrees Celsius at sea level.",
        "Question: What is gravity? Answer: Gravity is the force that attracts objects toward each other.",
        "Question: What is the largest planet? Answer: Jupiter is the largest planet in our solar system.",
        "Question: What is the closest star to Earth? Answer: The Sun is the closest star to Earth.",
        "Question: What is mitochondria? Answer: Mitochondria are organelles that produce energy in cells.",
        "Question: What is the atmosphere made of? Answer: The atmosphere is mostly nitrogen and oxygen.",
        "Question: What causes earthquakes? Answer: Earthquakes are caused by tectonic plate movements.",
    ]
    texts.extend(factual_qa)

    # ── Reasoning chains ──
    reasoning = [
        "Premise: All cats are animals. Premise: All animals are living things. Conclusion: All cats are living things. This is valid syllogistic reasoning.",
        "Premise: If it rains, the ground gets wet. Premise: It is raining. Conclusion: The ground is wet. This is modus ponens.",
        "Premise: If it rains, the ground gets wet. Premise: The ground is not wet. Conclusion: It is not raining. This is modus tollens.",
        "Premise: A is taller than B. Premise: B is taller than C. Conclusion: A is taller than C. This is transitivity.",
        "Premise: All birds have wings. Premise: A penguin is a bird. Conclusion: A penguin has wings. This is universal instantiation.",
        "Observation: The ground is wet. Best explanation: It rained. This is abductive reasoning.",
        "If all mammals are warm-blooded, and a whale is a mammal, then a whale is warm-blooded. Deductive reasoning from general to specific.",
        "The sun rises in the east. The sun has risen. Therefore it is likely morning. Inductive reasoning from pattern to instance.",
        "If all metals conduct electricity, and copper is a metal, then copper conducts electricity. This is a valid deduction.",
        "Evidence: Snow on the ground. Evidence: Footprints in the snow. Inference: Someone walked here after it snowed. Multi-step reasoning.",
    ]
    texts.extend(reasoning)

    # ── Evidence evaluation ──
    evidence = [
        "Evidence 1: Study shows exercise reduces heart disease. Evidence 2: Exercise improves mental health. Assessment: The evidence supports the claim that exercise is beneficial. Two independent sources corroborate.",
        "Evidence 1: The drug reduces symptoms. Evidence 2: The drug shows no effect. Assessment: The evidence is contradictory. We cannot draw a definitive conclusion without more data.",
        "Evidence 1: Water boils at 100C. Evidence 2: Water freezes at 0C. Assessment: These are established scientific facts with strong evidence.",
        "Source A: Official government report. Source B: Anonymous blog post. Assessment: Source A is more reliable than Source B. Official documents carry more weight.",
        "Claim: The Earth is flat. Evidence against: Photos from space, ship hulls disappearing over horizon, circumnavigation. Assessment: The claim is contradicted by overwhelming evidence.",
        "Claim: Vaccines cause autism. Evidence: Multiple large studies show no link. Assessment: The claim is not supported by scientific evidence.",
        "Evidence: Company revenue increased 15%. Evidence: Company laid off 10% of staff. Assessment: Revenue growth despite layoffs suggests efficiency improvements.",
    ]
    texts.extend(evidence)

    # ── Contradiction detection ──
    contradictions = [
        "Statement A: The meeting is at 3 PM. Statement B: The meeting is at 4 PM. These statements contradict each other about the meeting time.",
        "Statement A: The drug is effective. Statement B: The drug is ineffective. These statements present opposing views on the drug's efficacy.",
        "Statement A: All students passed the exam. Statement B: Some students failed the exam. These statements contradict each other.",
        "Statement A: It is raining outside. Statement B: It is sunny and dry outside. These statements describe opposite weather conditions.",
        "Statement A: The company is profitable. Statement B: The company reported losses. These statements contradict the financial status.",
    ]
    texts.extend(contradictions)

    # ── Uncertainty and abstention ──
    uncertainty = [
        "Question: What will the stock market do tomorrow? Answer: It is impossible to predict with certainty. The answer is uncertain.",
        "Question: Who will win the next election? Answer: This cannot be determined with current information. The answer is unknown.",
        "Question: What happened yesterday at 3:42 PM in New York? Answer: Without specific information, this cannot be answered. Insufficient evidence.",
        "Question: Is this investment safe? Answer: It depends on many factors that are not fully known. The answer requires more analysis.",
        "Question: Will it rain next week? Answer: Weather predictions beyond 5 days are unreliable. The answer is uncertain.",
    ]
    texts.extend(uncertainty)

    # ── Entity resolution ──
    entity_resolution = [
        "Record 1: John Smith, age 30, New York. Record 2: J. Smith, age 30, NYC. Assessment: These likely refer to the same person. Similar name, same age, same city.",
        "Record 1: Apple Inc., technology company. Record 2: Apple Corps, music company. Assessment: These are different entities. Despite similar names, they operate in different industries.",
        "Record 1: Cambridge University, UK. Record 2: Cambridge University, MA, USA. Assessment: These are different institutions in different countries.",
    ]
    texts.extend(entity_resolution)

    # ── Math reasoning ──
    math_reasoning = [
        "Problem: What is 15% of 200? Solution: 15% of 200 = 0.15 * 200 = 30. The answer is 30.",
        "Problem: If a car travels 60 mph for 2.5 hours, how far does it go? Solution: Distance = speed * time = 60 * 2.5 = 150 miles.",
        "Problem: What is the area of a rectangle with length 8 and width 5? Solution: Area = length * width = 8 * 5 = 40 square units.",
        "Problem: Convert 100 degrees Fahrenheit to Celsius. Solution: C = (F - 32) * 5/9 = (100 - 32) * 5/9 = 68 * 5/9 = 37.8 degrees Celsius.",
        "Problem: What is 2 raised to the power of 10? Solution: 2^10 = 1024.",
    ]
    texts.extend(math_reasoning)

    # ── Causal reasoning ──
    causal = [
        "Cause: Heating ice. Effect: Ice melts into water. Causal relationship: heat causes phase change from solid to liquid.",
        "Cause: Deforestation. Effect: Soil erosion. Causal relationship: removing tree roots destabilizes soil.",
        "Cause: Greenhouse gas emissions. Effect: Global warming. Causal relationship: greenhouse gases trap heat in atmosphere.",
        "Cause: Exercise. Effect: Improved cardiovascular health. Causal relationship: regular exercise strengthens the heart.",
        "Cause: Lack of sleep. Effect: Reduced cognitive performance. Causal relationship: sleep is necessary for brain function.",
    ]
    texts.extend(causal)

    # ── Temporal reasoning ──
    temporal = [
        "Event A: The meeting started at 9 AM. Event B: The meeting ended at 11 AM. Event B occurred after Event A. Duration was 2 hours.",
        "First: The company was founded in 1990. Then: It went public in 2005. Finally: It was acquired in 2020. Timeline spans 30 years.",
        "Before: The building was constructed in 1950. During: It was renovated in 2000. After: It was demolished in 2020.",
    ]
    texts.extend(temporal)

    # ── Source reliability ──
    source_reliability = [
        "Source: Peer-reviewed journal article. Reliability: High. Peer review ensures quality control.",
        "Source: Wikipedia article. Reliability: Medium. Community-edited but can contain errors.",
        "Source: Anonymous social media post. Reliability: Low. No accountability or verification.",
        "Source: Official government report. Reliability: High. Government data is typically reliable.",
        "Source: Personal blog. Reliability: Low to medium. Depends on the author's expertise.",
    ]
    texts.extend(source_reliability)

    logger.info(f"Created training dataset with {len(texts)} examples")
    return texts


# ══════════════════════════════════════════════════════════════════
# STEP 2: Build tokenizer and dataset
# ══════════════════════════════════════════════════════════════════

def build_tokenizer_and_dataset(texts: list[str], checkpoint_dir: str):
    """Build a tokenizer from the training texts and create a dataset."""
    from companion.neural.tokenizer import RelayBpeTokenizer
    from companion.neural.training.datasets import TextDataset

    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")

    # Train a BPE tokenizer from the training texts
    logger.info("Building tokenizer...")
    from companion.neural.tokenizer import train_tokenizer
    tokenizer = train_tokenizer(texts, vocab_size=4096)
    tokenizer.save(tokenizer_path)
    logger.info(f"Tokenizer saved to {tokenizer_path} (vocab_size={tokenizer.vocab_size})")

    # Build dataset
    logger.info("Building tokenized dataset...")
    dataset = TextDataset.from_texts(
        texts=texts,
        tokenizer=tokenizer,
        source="sweep_knowledge_base",
        tokenizer_path=tokenizer_path,
    )
    logger.info(f"Dataset: {dataset.n_tokens} tokens")
    return tokenizer, dataset


# ══════════════════════════════════════════════════════════════════
# STEP 3: Train the Relay Transformer
# ══════════════════════════════════════════════════════════════════

def train_model(checkpoint_dir: str):
    """Run actual training with real gradient updates."""
    from companion.neural.architecture.config import ModelConfig
    from companion.neural.architecture.transformer import RelayTransformer
    from companion.neural.training.trainer import train, TrainConfig
    from companion.neural.training.checkpointing import save_checkpoint, load_model

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create data
    texts = create_training_data()
    tokenizer, dataset = build_tokenizer_and_dataset(texts, checkpoint_dir)

    # Build model (nano scale: ~2M params)
    logger.info("Building Relay Transformer (nano scale)...")
    config = ModelConfig(
        name="relay-nano",
        version="0.1.0",
        vocab_size=4096,
        hidden_size=128,
        intermediate_size=512,
        num_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_context_length=512,
        normalization="rmsnorm",
        positional_encoding="rope",
        tie_word_embeddings=False,
        status="training",
    )
    model = RelayTransformer(config)
    params = model.param_count()
    trainable = model.param_count_trainable()
    logger.info(f"Model: {params:,} total params, {trainable:,} trainable")
    logger.info(f"Breakdown: {model.param_breakdown()}")

    # Training config
    train_config = TrainConfig(
        batch_size=4,
        seq_len=128,
        learning_rate=3e-4,
        warmup_steps=50,
        total_steps=500,
        max_grad_norm=1.0,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_every=100,
        log_every=10,
        seed=42,
        device="cpu",
        dtype="fp32",
    )

    # Data iterator — yields (xs, ys) blocks repeatedly
    def data_iter():
        while True:
            for batch in dataset.to_dataloader(train_config.batch_size, train_config.seq_len, seed=42):
                yield batch

    # Log initial model state
    logger.info("=" * 60)
    logger.info("STARTING REAL TRAINING")
    logger.info(f"  Steps: {train_config.total_steps}")
    logger.info(f"  Batch size: {train_config.batch_size}")
    logger.info(f"  Sequence length: {train_config.seq_len}")
    logger.info(f"  Learning rate: {train_config.learning_rate}")
    logger.info(f"  Device: cpu")
    logger.info(f"  Parameters: {params:,}")
    logger.info("=" * 60)

    losses = []
    def on_log(entry):
        losses.append(entry["loss"])

    t0 = time.perf_counter()
    result = train(
        model=model,
        tokenizer=tokenizer,
        data_iter=data_iter,
        config=train_config,
        checkpoint_dir=checkpoint_dir,
        on_log=on_log,
    )
    elapsed = time.perf_counter() - t0

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Steps run: {result.steps_run}")
    logger.info(f"  Final loss: {result.final_loss:.4f}")
    logger.info(f"  Best loss: {result.best_loss:.4f}")
    logger.info(f"  Duration: {elapsed:.1f}s")
    logger.info(f"  Checkpoint: {result.checkpoint_dir}")
    logger.info(f"  Losses: {losses[:5]}...{losses[-5:]}")
    logger.info("=" * 60)

    return model, tokenizer, result


# ══════════════════════════════════════════════════════════════════
# STEP 4: Verify the trained model
# ══════════════════════════════════════════════════════════════════

def verify_model(checkpoint_dir: str):
    """Load the trained model and run a forward pass to verify it works."""
    from companion.neural.training.checkpointing import load_model, load_tokenizer
    from companion.neural.architecture.transformer import RelayTransformer

    logger.info("Loading trained model from checkpoint...")
    model = load_model(checkpoint_dir)
    tokenizer = load_tokenizer(checkpoint_dir)

    # Forward pass
    test_text = "Question: What is the capital of France? Answer:"
    tokens = tokenizer.encode(test_text)
    logger.info(f"Input: '{test_text}'")
    logger.info(f"Tokens: {tokens}")

    import torch
    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        logits, hidden = model(input_ids)

    logger.info(f"Output shape: logits={logits.shape}, hidden={hidden.shape}")
    logger.info(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    logger.info(f"Model loaded and produces output: YES")

    # Check checkpoint files
    checkpoint_files = list(Path(checkpoint_dir).glob("*"))
    logger.info(f"Checkpoint files: {[f.name for f in checkpoint_files]}")

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    checkpoint_dir = str(_sweep_dir / "training" / "relay_nano_trained")

    logger.info("=" * 70)
    logger.info("SWEEP NEURAL ENGINE — REAL TRAINING RUN")
    logger.info("=" * 70)

    # Train
    model, tokenizer, result = train_model(checkpoint_dir)

    # Verify
    verify_model(checkpoint_dir)

    # Save summary
    summary = {
        "model": "relay-nano",
        "parameters": model.param_count(),
        "trainable_parameters": model.param_count_trainable(),
        "training_steps": result.steps_run,
        "final_loss": result.final_loss,
        "best_loss": result.best_loss,
        "duration_seconds": result.duration_seconds,
        "checkpoint_dir": result.checkpoint_dir,
        "status": "trained",
        "training_performed": True,
    }
    summary_path = os.path.join(checkpoint_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training summary saved to {summary_path}")
    logger.info("TRAINING COMPLETE — MODEL IS NOW TRAINED")
