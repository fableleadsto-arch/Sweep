"""Token and positional input embeddings."""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Standard learned token embedding table."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=hidden_size**-0.5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]


class LearnedPositionalEmbedding(nn.Module):
    """Optional learned positional embedding (used only when RoPE is disabled)."""

    def __init__(self, max_context_length: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(max_context_length, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=hidden_size**-0.5)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        if positions is None:
            positions = torch.arange(x.shape[1], device=x.device)
        return x + self.weight[positions]


def build_embeddings(config) -> tuple[TokenEmbedding, nn.Module | None]:
    """Return (token_embedding, optional positional_embedding)."""
    token_emb = TokenEmbedding(config.vocab_size, config.hidden_size)
    pos_emb = None
    if config.positional_encoding == "none":
        pos_emb = LearnedPositionalEmbedding(config.max_context_length, config.hidden_size)
    return token_emb, pos_emb


__all__ = ["TokenEmbedding", "LearnedPositionalEmbedding", "build_embeddings"]
