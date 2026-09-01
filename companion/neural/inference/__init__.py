"""Inference: sampling + KV-cache generation."""

from __future__ import annotations

from .generator import GenerationConfig, GenerationResult, generate
from .sampling import sample_token

__all__ = ["GenerationConfig", "GenerationResult", "generate", "sample_token"]
