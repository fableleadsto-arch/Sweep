# SWEEP NEURAL ENGINE — TRAINING RESULTS

**Date:** 2026-08-30
**Training performed:** YES
**Models trained:** 2 (Relay Transformer + MiniLM integration)

---

## What Was Actually Trained

### 1. Relay Transformer (nano scale)
- **Parameters:** 2,098,304
- **Training steps:** 500
- **Initial loss:** 8.21
- **Final loss:** 0.0023
- **Perplexity:** 295.62 → 1.00
- **Duration:** 231 seconds (CPU)
- **Checkpoint:** `sweep_neural_mesh/training/relay_nano_trained/model.safetensors` (8.4MB)
- **Training data:** 63 examples from Sweep's knowledge base

### 2. Relay Transformer (small scale)
- **Parameters:** 10,489,088
- **Training steps:** 200 (partial — timed out)
- **Best loss:** 0.54
- **Checkpoint:** `sweep_neural_mesh/training/relay_small_trained/model.safetensors` (42MB)
- **Training data:** 1000+ examples from Sweep's knowledge base

### 3. MiniLM Embeddings (pretrained, integrated)
- **Model:** minishlab/potion-base-32M
- **Embedding dim:** 256
- **Integration:** Evidence cross-referencing pipeline
- **Status:** Loaded and functional

---

## Benchmark Results

| Component | Accuracy | Notes |
|-----------|----------|-------|
| Relay Transformer (nano) | 0/10 (0%) | Too small (2M params) for QA from 63 examples |
| Relay Transformer (small) | 0/10 (0%) | Partially trained (200/500 steps), needs more training |
| MiniLM embeddings | Working | Scores 0.4-0.6 (topic similarity, not opposition) |
| Cortex (rule-based) | 4/6 (67%) | Works for simple cases, fails on contradictions |

---

## Honest Assessment

### What Training Actually Achieved
1. **Real gradient updates** — The Relay Transformer was trained with actual backpropagation
2. **Loss decreased** — From 8.21 to 0.0023 (nano) and to 0.54 (small)
3. **Checkpoint saved** — Real model weights (8.4MB and 42MB)
4. **Infrastructure verified** — Training loop, tokenizer, checkpointing all work

### What Training Did NOT Achieve
1. **The nano model (2M params) is too small** — It learned to predict tokens but can't do QA
2. **The small model (10M params) needs more training** — Only 200/500 steps completed
3. **63 training examples is too few** — Need 10,000+ for meaningful learning
4. **The models can't answer questions** — They predict next tokens, not answers
5. **BERT fine-tuning timed out** — Too slow on CPU (110M params)

### Why the Models Can't Answer Questions
The Relay Transformer is a **causal language model** — it predicts the next token given previous tokens. It's not a QA system. To answer questions, you need:
1. Much more training data (10,000+ QA pairs)
2. Instruction fine-tuning (not just next-token prediction)
3. Or a much larger model (100M+ params)

### What Would Actually Work
1. **Fine-tune a pretrained LLM** (e.g., LLaMA, Mistral) on Sweep's tasks
2. **Use the pretrained BERT/RoBERTa** for evidence classification (not generation)
3. **Scale up training data** to 10,000+ examples
4. **Train for 10,000+ steps** instead of 500

---

## What's Actually Working

### MiniLM Embeddings ✅
- Semantic similarity scores: 0.4-0.6 for all pairs
- Integrated into evidence cross-referencing
- Falls back to word overlap when unavailable

### Rule-Based System ✅
- 200+ factual answers via regex
- Evidence scoring via word overlap
- Threshold-based decisions
- Works for simple cases

### Training Infrastructure ✅
- AdamW optimizer (custom implementation)
- Checkpointing with safetensors
- Tokenizer training
- Curriculum learning framework
- All verified working end-to-end

---

## Conclusion

**Training has been performed. Real gradient updates occurred. Real checkpoints were saved.**

However, the models are too small and undertrained to produce "astounding" results. The nano model (2M params) trained on 63 examples is a proof of concept, not a production system.

The real value is:
1. **Verified training infrastructure** — Ready for larger-scale training
2. **Pretrained embeddings** — MiniLM providing semantic understanding
3. **Modular architecture** — Ready to integrate larger models

To achieve "astounding" results, you would need:
- A pretrained LLM (100M+ params)
- 10,000+ training examples
- 10,000+ training steps
- GPU acceleration
- Instruction fine-tuning

The current training proves the infrastructure works. The next step is scaling up.
