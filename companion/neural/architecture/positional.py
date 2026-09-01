"""Rotary positional embeddings (RoPE).

Position information is injected by rotating pairs of head-dimension features
by an angle proportional to the token's absolute position. ``max_context_length``
frequencies are precomputed once and reused (and cached across KV-cache decode
steps), so generation never recomputes them per token.
"""

from __future__ import annotations

import torch
from torch import nn


def precompute_rope_freqs(
    max_context_length: int,
    head_dim: int,
    theta: float = 10_000.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) each of shape (max_context_length, head_dim)."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(max_context_length, dtype=torch.float32, device=device)
    angles = torch.outer(positions, inv_freq)  # (seq, head_dim // 2)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)  # (seq, head_dim)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape (batch, seq, num_heads, head_dim).

    ``cos``/``sin`` come from :func:`precompute_rope_freqs` (seq, head_dim);
    ``positions`` holds absolute token positions and is indexed into cos/sin.
    """
    x = x.to(cos.dtype) if x.dtype != cos.dtype else x
    b, s, h, d = x.shape
    half = d // 2
    cos_sel = cos[positions][..., :half]  # (s, half)
    sin_sel = sin[positions][..., :half]  # (s, half)

    x1 = x[..., :half]
    x2 = x[..., half:]
    cos_ = cos_sel.unsqueeze(1).unsqueeze(0)  # (1, s, 1, half)
    sin_ = sin_sel.unsqueeze(1).unsqueeze(0)

    rotated = torch.cat([x1 * cos_ - x2 * sin_, x2 * cos_ + x1 * sin_], dim=-1)
    return rotated


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE buffers, registered so they move device with the model."""

    def __init__(self, config) -> None:
        super().__init__()
        self.max_context_length = config.max_context_length
        self.head_dim = config.head_dim
        cos, sin = precompute_rope_freqs(config.max_context_length, config.head_dim, config.rope_theta)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return apply_rotary_emb(x, self.cos_cached, self.sin_cached, positions)


def make_positions(seq_len: int, start_pos: int, device: torch.device) -> torch.Tensor:
    return torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=device)


__all__ = ["RotaryEmbedding", "precompute_rope_freqs", "apply_rotary_emb", "make_positions"]
