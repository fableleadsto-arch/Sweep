"""Normalization modules (RMSNorm / LayerNorm)."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no bias, optional epsilon).

    Unlike LayerNorm there is no mean-subtraction, which makes it cheaper and
    is the standard choice for modern autoregressive transformers.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


def build_norm(kind: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """Factory: ``rmsnorm`` or ``layernorm``."""
    if kind == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    raise ValueError(f"unknown normalization kind '{kind}'")
