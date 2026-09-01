"""Task-specific output heads that reuse the transformer's hidden states.

The generative head is the (optionally weight-tied) LM head producing logits
over the vocabulary. Capability heads sit on top of hidden states for
downstream tasks: intent classification, semantic similarity embeddings, and
per-task logit heads. Heads are always evaluated for real (never pre-claimed).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LmHead(nn.Module):
    """Vocabulary logits. Optionally tied to the token embedding weights."""

    def __init__(self, hidden_size: int, vocab_size: int, tie_weights: torch.nn.Parameter | None = None) -> None:
        super().__init__()
        if tie_weights is not None:
            self.weight = tie_weights  # shared Parameter, not a fresh one
        else:
            self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
            nn.init.normal_(self.weight, mean=0.0, std=hidden_size**-0.5)
        self.tied = tie_weights is not None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)


class ClassificationHead(nn.Module):
    """Single-label classification head with temperature scaling."""

    def __init__(self, hidden_size: int, num_labels: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_labels)
        self.temperature = temperature

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states) / self.temperature


class SimilarityHead(nn.Module):
    """Projects hidden states into a fixed-size similarity embedding space."""

    def __init__(self, hidden_size: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, embedding_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(hidden_states), dim=-1)


__all__ = ["LmHead", "ClassificationHead", "SimilarityHead"]
