"""Grouped-Query Attention with RoPE and incremental KV caching.

Supports:
- full causal attention (prefill / training), and
- single-token decode with a cached (K, V) per layer,
so generation is O(1) per new token instead of O(seq).

``num_key_value_heads < num_attention_heads`` gives Grouped-Query Attention
(MQA when kv_heads == 1), which is what lets larger scales fit in RAM.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GroupedQueryAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.bias = config.bias
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.bias)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,
        cache_k: torch.Tensor | None,
        cache_v: torch.Tensor | None,
        positions: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x``: (batch, seq, hidden). Returns (out, new_k, new_v).

        ``cache_k``/``cache_v`` hold all keys/values up to ``start_pos``
        (including), or are None on the very first token.
        """
        from .positional import apply_rotary_emb

        b, s, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)  # (b, h, s, d)
        k = k.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE is applied in (b, s, h, d) layout for clarity.
        q = apply_rotary_emb(q.transpose(1, 2), cos, sin, positions).transpose(1, 2)
        k = apply_rotary_emb(k.transpose(1, 2), cos, sin, positions).transpose(1, 2)

        if cache_k is not None:
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)

        new_k, new_v = k, v
        if self.num_kv_heads < self.num_heads:
            # Repeat KV heads up to the query head count.
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        attn_weights = (q * self.scale) @ k.transpose(-1, -2)  # (b, h, s, s+k)
        total = attn_weights.shape[-1]
        if total > 1:
            # Causal mask: query row i (absolute start_pos+i) may attend to key
            # columns j <= start_pos + i. For a single-token decode the window
            # is one column wide, so nothing needs masking.
            start_pos = int(positions[0])
            mask = torch.triu(
                torch.ones(attn_weights.shape[-2], total, device=x.device, dtype=torch.bool),
                diagonal=1 + start_pos,
            )
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        if self.attention_dropout and self.training:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        out = attn_weights @ v  # (b, h, s, d)
        out = out.transpose(1, 2).contiguous().view(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out), new_k, new_v


__all__ = ["GroupedQueryAttention"]
