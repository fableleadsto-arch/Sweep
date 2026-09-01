# Relay Neural — native intelligence layer

Relay's own neural stack lives in the companion brain service. It is a
configuration-driven transformer (RoPE + GQA + RMSNorm + SwiGLU) trained from
real corpora, plus a tokenizer, a trainer, an inference engine, an honest
hardware-aware selector, a model registry, and a native-first router that
falls back to external API providers when the native model can't (honestly)
handle a task.

**Ground rules (non-negotiable):**

- Parameter counts are always *computed* from the config/model, never written
  by hand and never guessed.
- No fake intelligence, no fake benchmarks, no fabricated generations.
- External API providers remain the fallback path — native never silently
  replaces them.
- Evaluation runs before scaling. The default scale is `nano`, which trains on
  this machine; larger scales only move up the ladder when the data shows they
  earn it.

## Layout

```
companion/neural/
  architecture/     config, embeddings, RoPE, GQA attention, SwiGLU, blocks,
                    transformer (prefill + KV-cache decode), output heads
  models/scales.py  nano / small / medium / large / x — honest sizes
  tokenizer/        versioned BPE tokenizer (HF tokenizers, byte-level)
  training/         datasets, dataloader, AdamW, warmup+cosine, checkpointing
  inference/        sampling (temp/top-k/top-p/min-p/repetition) + generator
  evaluation/       forward/backward/checkpoint/tokenizer/generation battery
  selection.py      hardware-aware fit estimation + recommendation
  registry.py       model registry with real param counts
  router.py         native-first capability router w/ external fallback
  __main__.py       CLI: python -m companion.neural <command>
```

## Quick start

```bash
# hardware report + the largest scale that truly fits this machine
python -m companion.neural info

# train a BPE tokenizer
python -m companion.neural tokenizer-train data.txt --output data/neural/tok.json --vocab-size 4096

# train the nano model on a text corpus (real forward/backward, real loss)
python -m companion.neural train data.txt --scale nano --steps 2000 --outdir data/neural/relay-nano

# generate with the trained model
python -m companion.neural generate --model data/neural/relay-nano --prompt "relay"

# run the evaluation battery (real measurements)
python -m companion.neural eval --model data/neural/relay-nano
```

## API

| Endpoint | Purpose |
| --- | --- |
| `GET  /api/brain/native/models` | registered models + computed params |
| `GET  /api/brain/native/models/{name}` | one model's record |
| `GET  /api/brain/native/recommend` | hardware-aware scale recommendation |
| `POST /api/brain/native/generate` | generate with a trained native model |
| `POST /api/brain/native/route` | native-vs-external routing decision |

Compute diagnostics (`GET /api/brain/compute/diagnostics`) includes a
`native_models` section listing every registered model with its real
parameter count and verification state.

## Honest sizing

`selection.py` answers "can I train this here?" with real math:

```
train footprint = parameters × bytes/param × 3   (weights + grads + Adam moments)
total          = footprint + KV cache + activations
fits           = total < 0.6 × available RAM      (40% headroom for the runtime)
```

On this machine (CPU-only, ~16 GB RAM) the recommendation is typically
`nano`/`small` comfortably, `medium` only at reduced precision, and
`large`/`x` as future/distributed targets. The selector reports what fits
instead of pretending.

## Router semantics

`router.py` routes `generation` / `classify` / `embeddings` to native only
when a **trained, verified** model is registered and the input fits the
context window. Any other case falls back to external providers with a
recorded reason (`decision_history`), so every routing choice is auditable.

## Dependencies

torch, `tokenizers`, `safetensors` — all optional and lazy-loaded. The brain
service never imports them at boot; `import companion.neural` is
dependency-free (verified by `tests/test_neural.py`).
