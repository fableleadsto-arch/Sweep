"""Autoregressive generation with KV cache.

Flow: encode prompt → prefill once → decode token-by-token, sampling from real
logits. Returns the generated text, the token count, and honest performance
stats (tokens/sec). No simulated output, ever.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..architecture import RelayTransformer
from ..tokenizer import RelayBpeTokenizer
from .sampling import sample_token


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    eos_token_id: int = 3  # convention: 0=pad,1=unk,2=bos,3=eos
    stop_sequences: list[str] = field(default_factory=list)
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    text: str
    prompt: str
    input_tokens: int
    generated_tokens: int
    token_ids: list[int]
    tokens_per_second: float
    duration_s: float


def generate(
    model: RelayTransformer,
    tokenizer: RelayBpeTokenizer,
    prompt: str,
    config: GenerationConfig = GenerationConfig(),
    device: torch.device | str = "cpu",
) -> GenerationResult:
    """Generate text from a prompt using the given (real, trained) model."""
    device = torch.device(device)
    model = model.to(device).eval()

    rng = torch.Generator(device="cpu")
    if config.seed is not None:
        rng.manual_seed(config.seed)

    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cache = model.new_cache()

    start = time.time()
    with torch.no_grad():
        logits, _ = model(input_ids, cache=cache)  # prefill
        generated: list[int] = []
        stop_at = len(prompt_ids) + config.max_new_tokens

        for _ in range(config.max_new_tokens):
            next_logits = logits[:, -1, :].reshape(-1)
            past = torch.tensor(prompt_ids + generated, dtype=torch.long, device="cpu")
            next_id = sample_token(
                next_logits,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                min_p=config.min_p,
                repetition_penalty=config.repetition_penalty,
                past_tokens=past,
                rng=rng,
            )
            generated.append(next_id)
            if next_id == config.eos_token_id:
                break

            nxt = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, _ = model(nxt, start_pos=len(prompt_ids) + len(generated) - 1, cache=cache)

            # Stop-sequence check via the growing decoded suffix.
            if config.stop_sequences:
                text_so_far = tokenizer.decode(prompt_ids + generated)
                if any(seq in text_so_far for seq in config.stop_sequences):
                    break

            if len(prompt_ids) + len(generated) >= stop_at:
                break

    duration = time.time() - start
    generated_text = tokenizer.decode(generated)
    n_tokens = len(generated)
    tps = n_tokens / duration if duration > 0 else 0.0

    return GenerationResult(
        text=generated_text,
        prompt=prompt,
        input_tokens=len(prompt_ids),
        generated_tokens=n_tokens,
        token_ids=generated,
        tokens_per_second=tps,
        duration_s=duration,
    )


__all__ = ["GenerationConfig", "GenerationResult", "generate"]
