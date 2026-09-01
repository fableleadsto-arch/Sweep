"""Pre-norm residual transformer block."""

from __future__ import annotations

import torch
from torch import nn

from .attention import GroupedQueryAttention
from .feed_forward import build_feed_forward
from .normalization import build_norm


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = build_norm(config.normalization, config.hidden_size)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = build_norm(config.normalization, config.hidden_size)
        self.mlp = build_feed_forward(config)
        self.dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cache_k: torch.Tensor | None,
        cache_v: torch.Tensor | None,
        positions: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_layernorm(x)
        attn_out, new_k, new_v = self.self_attn(h, cache_k, cache_v, positions, cos, sin)
        x = x + self.dropout(attn_out)
        h = self.post_attention_layernorm(x)
        x = x + self.dropout(self.mlp(h))
        return x, new_k, new_v


__all__ = ["TransformerBlock"]
