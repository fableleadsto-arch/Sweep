"""Token sampling strategies. Operate on real model logits only."""

from __future__ import annotations

import torch


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    past_tokens: torch.Tensor | None = None,
    rng: torch.Generator | None = None,
) -> int:
    """Sample one token id from a (vocab,) logits vector.

    All settings are applied for real:
    - temperature: scale logits before softmax,
    - top_k: keep only the k most probable tokens,
    - top_p: nucleus filtering,
    - min_p: minimum probability floor (disabled at 0),
    - repetition_penalty: penalize tokens already generated.
    """
    logits = logits.float()

    if repetition_penalty != 1.0 and past_tokens is not None and past_tokens.numel() > 0:
        for tok in torch.unique(past_tokens):
            score = logits[tok]
            logits[tok] = score / repetition_penalty if score > 0 else score * repetition_penalty

    if temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_k > 0:
        k = min(top_k, probs.numel())
        values, indices = torch.topk(probs, k)
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[indices] = True
        probs = probs.masked_fill(~mask, 0.0)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum - sorted_probs <= top_p
        keep[0] = True  # always keep the single most probable token
        filtered = torch.zeros_like(probs)
        filtered[sorted_idx[keep]] = sorted_probs[keep]
        probs = filtered

    if min_p > 0:
        max_prob = probs.max()
        probs = probs.masked_fill(probs < min_p * max_prob, 0.0)

    if probs.sum() == 0:
        return int(torch.argmax(logits))

    probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=rng).item())


__all__ = ["sample_token"]
