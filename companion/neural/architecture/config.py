"""Configuration-driven model architecture.

One dataclass drives the whole stack: embeddings, RoPE, GQA attention, RMSNorm,
SwiGLU MLPs, residual blocks, and the output heads. Nothing is hard-coded to a
single tiny model — sizes come from a config (JSON/YAML-able dict), so the same
code trains Relay Nano on a laptop and (eventually) Relay X on a cluster.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    """All knobs for a Relay Transformer.

    Values are validated on construction so a bad config fails loudly instead
    of producing a silently-misshapen model.
    """

    # ── identity ────────────────────────────────────────────────────
    name: str = "relay-nano"
    version: str = "0.1.0"
    architecture: str = "transformer"
    framework: str = "pytorch"

    # ── dimensions ──────────────────────────────────────────────────
    vocab_size: int = 4096
    hidden_size: int = 128
    intermediate_size: int = 512
    num_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    max_context_length: int = 1024

    # ── architecture choices ────────────────────────────────────────
    normalization: str = "rmsnorm"          # rmsnorm | layernorm
    positional_encoding: str = "rope"       # rope | none
    rope_theta: float = 10_000.0
    bias: bool = False
    tie_word_embeddings: bool = False
    dropout: float = 0.0
    attention_dropout: float = 0.0
    final_norm: bool = True

    # ── training metadata ───────────────────────────────────────────
    training_dataset: str = ""
    status: str = "experimental"
    precision: str = "fp32"                 # fp32 | fp16 | bf16
    created_at: str = ""
    hardware: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if self.hidden_size <= 0 or self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be positive and divisible by num_attention_heads"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.max_context_length <= 0:
            raise ValueError("max_context_length must be positive")
        if self.normalization not in ("rmsnorm", "layernorm"):
            raise ValueError("normalization must be 'rmsnorm' or 'layernorm'")
        if self.positional_encoding not in ("rope", "none"):
            raise ValueError("positional_encoding must be 'rope' or 'none'")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(**data)


def merge_with_scale(scale_name: str, overrides: Optional[dict[str, Any]] = None) -> ModelConfig:
    """Build a config from a named scale (nano..x) plus explicit overrides."""
    from ..models.scales import scale_config

    cfg = scale_config(scale_name)
    merged = dict(cfg)
    if overrides:
        merged.update(overrides)
    return ModelConfig(**merged)
