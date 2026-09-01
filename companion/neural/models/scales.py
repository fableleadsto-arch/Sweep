"""Named model scales with honest sizing for CPU/GPU.

Sizes are chosen so every scale trains and runs on the companion's hardware:
- ``nano`` / ``small``  → trainable on this machine (CPU, ~15 GB RAM).
- ``medium``            → CPU-inference viable only with reduced precision.
- ``large`` / ``x``     → future / distributed targets, never silently loaded.

Parameter counts are computed analytically from the config (see
``companion/neural/selection.py``); these are design targets, not claims.
"""

from __future__ import annotations

from typing import Any, Dict

SCALES: Dict[str, Dict[str, Any]] = {
    "nano": {
        "name": "relay-nano",
        "version": "0.1.0",
        "vocab_size": 4096,
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_context_length": 1024,
        "training_dataset": "relay-untrained",
        "status": "experimental",
        "precision": "fp32",
    },
    "small": {
        "name": "relay-small",
        "version": "0.1.0",
        "vocab_size": 8192,
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "max_context_length": 2048,
        "training_dataset": "relay-untrained",
        "status": "experimental",
        "precision": "fp32",
    },
    "medium": {
        "name": "relay-medium",
        "version": "0.1.0",
        "vocab_size": 16384,
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_context_length": 4096,
        "training_dataset": "relay-untrained",
        "status": "experimental",
        "precision": "bf16",
    },
    "large": {
        "name": "relay-large",
        "version": "0.1.0",
        "vocab_size": 32768,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "max_context_length": 8192,
        "training_dataset": "relay-untrained",
        "status": "experimental",
        "precision": "bf16",
    },
    "x": {
        "name": "relay-x",
        "version": "0.1.0",
        "vocab_size": 65536,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_context_length": 16384,
        "training_dataset": "relay-untrained",
        "status": "planned",
        "precision": "bf16",
    },
}


def scale_config(scale_name: str) -> Dict[str, Any]:
    """Return the raw config dict for a named scale."""
    if scale_name not in SCALES:
        raise KeyError(f"unknown scale '{scale_name}' (available: {', '.join(SCALES)})")
    return dict(SCALES[scale_name])
