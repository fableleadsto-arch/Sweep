"""The Relay Transformer: token embedding → N residual blocks → final norm.

Interface is split for efficient generation:
- ``forward`` runs a full block over ``input_ids`` (prefill or training),
- ``decode_step`` runs a single new token against an existing KV cache,
so inference cost stays flat per generated token.
"""

from __future__ import annotations

import torch
from torch import nn

from .block import TransformerBlock
from .config import ModelConfig
from .embeddings import TokenEmbedding, build_embeddings
from .normalization import build_norm
from .output_heads import LmHead
from .positional import RotaryEmbedding, make_positions


class RelayTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers

        self.token_embedding, self.pos_embedding = build_embeddings(config)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_id=i) for i in range(config.num_layers)]
        )
        self.final_norm = build_norm(config.normalization, config.hidden_size) if config.final_norm else nn.Identity()
        self.lm_head = LmHead(config.hidden_size, config.vocab_size)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
            self.lm_head.tied = True

        self._rotary = None
        if config.positional_encoding == "rope":
            self._rotary = RotaryEmbedding(config)

    # ── forward passes ──────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        cache: "KVCache | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, hidden_states).

        ``input_ids``: (batch, seq). When ``cache`` is provided and
        ``start_pos > 0``, ``input_ids`` should contain only the newly
        generated tokens; previously cached positions are reused.
        """
        b, s = input_ids.shape
        if s == 0:
            raise ValueError("input_ids must not be empty")

        h = self.token_embedding(input_ids)  # (b, s, hidden)
        if self._rotary is not None:
            positions = make_positions(s, start_pos, input_ids.device)
        else:
            positions = make_positions(s, start_pos, input_ids.device)
            if self.pos_embedding is not None:
                h = self.pos_embedding(h, positions)

        cos = sin = None
        if self._rotary is not None:
            cos = self._rotary.cos_cached
            sin = self._rotary.sin_cached

        for i, layer in enumerate(self.layers):
            cache_k = cache.layers_k[i] if cache is not None and i < len(cache.layers_k) else None
            cache_v = cache.layers_v[i] if cache is not None and i < len(cache.layers_v) else None
            h, k, v = layer(h, cache_k, cache_v, positions, cos, sin)
            if cache is not None:
                cache.layers_k[i] = k
                cache.layers_v[i] = v

        if self.config.final_norm:
            h = self.final_norm(h)
        logits = self.lm_head(h)
        if cache is not None:
            cache.size = (cache.layers_k[0].shape[2] if cache.layers_k[0] is not None else 0)
        return logits, h

    def decode_step(
        self,
        input_ids: torch.Tensor,
        cache: "KVCache",
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate logits for the last token only, appending to ``cache``."""
        logits, _ = self.forward(input_ids, start_pos=start_pos, cache=cache)
        return logits[:, -1], logits

    # ── parameter bookkeeping ───────────────────────────────────────

    def param_count(self) -> int:
        """Real, honest total parameter count from the instantiated tensors."""
        return sum(p.numel() for p in self.parameters())

    def param_count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self) -> dict[str, int]:
        """Parameter count grouped by component type (e.g. ``attention``)."""
        counts: dict[str, int] = {"embedding": 0, "attention": 0, "feed_forward": 0, "norm": 0, "lm_head": 0, "other": 0}
        mapping: dict[type, str] = {}
        from .attention import GroupedQueryAttention

        for name, p in self.named_parameters():
            parts = name.split(".")
            kind = "other"
            if "token_embedding" in parts or "pos_embedding" in parts:
                kind = "embedding"
            elif "self_attn" in parts or "q_proj" in parts or "k_proj" in parts or "v_proj" in parts or "o_proj" in parts:
                kind = "attention"
            elif "mlp" in parts or "gate_proj" in parts or "up_proj" in parts or "down_proj" in parts:
                kind = "feed_forward"
            elif "norm" in parts or "input_layernorm" in parts or "post_attention_layernorm" in parts or "final_norm" in parts:
                kind = "norm"
            elif "lm_head" in parts:
                kind = "lm_head"
            counts[kind] += p.numel()
        return counts

    # ── cache helpers ───────────────────────────────────────────────

    def new_cache(self) -> "KVCache":
        return KVCache(num_layers=self.num_layers)

    @classmethod
    def from_config_dict(cls, data: dict) -> "RelayTransformer":
        return cls(ModelConfig.from_dict(data))

    def estimate_kv_cache_bytes(self, seq_len: int, batch: int = 1, bytes_per_elem: int = 4) -> int:
        per_layer = (
            batch * seq_len * self.config.num_key_value_heads * self.config.head_dim * 2
        )
        return per_layer * self.num_layers * bytes_per_elem

    def total_bytes(self, bytes_per_elem: int = 4) -> int:
        return self.param_count() * bytes_per_elem


class KVCache:
    """Holds (k, v) per layer; used across decode steps.

    Preallocated on CPU/GPU as float32 by default. ``len`` equals the number of
    cached positions.
    """

    def __init__(self, num_layers: int) -> None:
        self.layers_k: list[torch.Tensor | None] = [None] * num_layers
        self.layers_v: list[torch.Tensor | None] = [None] * num_layers
        self.size = 0

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.layers_k[layer] = k
        self.layers_v[layer] = v
        self.size = k.shape[2]

    def __len__(self) -> int:
        return self.size


__all__ = ["RelayTransformer", "KVCache"]
