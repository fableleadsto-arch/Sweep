"""Architecture building blocks (config, embeddings, attention, blocks).

Importing this package is torch-free; ``RelayTransformer`` is resolved lazily.
"""

from __future__ import annotations

from .config import ModelConfig, merge_with_scale

__all__ = ["ModelConfig", "merge_with_scale", "RelayTransformer", "KVCache"]


def __getattr__(name: str):
    if name in ("RelayTransformer", "KVCache"):
        from .transformer import KVCache, RelayTransformer

        return {"RelayTransformer": RelayTransformer, "KVCache": KVCache}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
