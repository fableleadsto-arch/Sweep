"""Feed-forward networks (SwiGLU MLP)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """Gated linear unit with SiLU activation (the modern standard).

    ``intermediate_size`` is used for both the gated and up projections; the
    output projection brings the result back to ``hidden_size``. To match a
    plain MLP of effective size 4*hidden, set intermediate_size ≈ 8/3*hidden.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MLP(nn.Module):
    """Plain two-layer MLP (used when an architecture disables SwiGLU)."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


def build_feed_forward(config) -> nn.Module:
    if config.extra.get("ffn_kind", "swiglu") == "mlp":
        return MLP(config.hidden_size, config.intermediate_size, config.bias)
    return SwiGLU(config.hidden_size, config.intermediate_size, config.bias)


__all__ = ["SwiGLU", "MLP", "build_feed_forward"]
