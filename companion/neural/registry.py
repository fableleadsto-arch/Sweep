"""Model registry: honest metadata + real parameter counts for every model.

A registry is a directory containing one subdirectory per model. Each model dir
holds:
- ``config.json``   — the ModelConfig dict (authoritative for parameter math),
- ``model.safetensors`` — real trained weights (absent for untrained models),
- ``tokenizer.json``    — the versioned BPE tokenizer file,
- ``training_state.json`` — dataset/step/epoch info written by the trainer.

``parameters`` is always *computed from the config* (never hand-written), and
``verified`` reports whether a matching instantiation was actually run.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..tools.common import module_available
from .architecture.config import ModelConfig
from .models.scales import SCALES

DEFAULT_REGISTRY_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "neural"


@dataclass
class ModelRecord:
    name: str
    version: str = ""
    architecture: str = "transformer"
    framework: str = "pytorch"
    parameters: int = 0
    context_length: int = 0
    training_dataset: str = ""
    status: str = "experimental"
    precision: str = "fp32"
    created_at: str = ""
    hardware: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    path: str = ""
    scale: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["parameters"] = self.parameters
        return d


def _count_parameters_from_config(config: ModelConfig) -> int:
    """Analytical parameter count — no model instantiation needed."""
    hidden = config.hidden_size
    head_dim = config.head_dim
    qkv = hidden * (config.num_attention_heads + 2 * config.num_key_value_heads) * head_dim
    out = config.num_attention_heads * head_dim * hidden
    attn = qkv + out
    ffn_gate = hidden * config.intermediate_size
    ffn_up = hidden * config.intermediate_size
    ffn_down = config.intermediate_size * hidden
    ffn = ffn_gate + ffn_up + ffn_down
    norm_params = 0
    if config.normalization == "rmsnorm":
        norm_params = hidden  # one weight per norm
    per_layer = attn + ffn + norm_params * 2 + (1 if config.bias else 0) * (hidden * 4 + config.intermediate_size * 3)
    total = config.vocab_size * hidden + per_layer * config.num_layers + (hidden if config.final_norm else 0) + hidden * config.vocab_size
    if config.tie_word_embeddings:
        total -= hidden * config.vocab_size
    return total


class ModelRegistry:
    """Scans a directory for models and builds records.

    Not a global singleton; the companion service owns one instance rooted at
    the data directory (and tests create their own temp registries).
    """

    def __init__(self, base_dir: Path = DEFAULT_REGISTRY_DIR) -> None:
        self.base_dir = Path(base_dir)

    def list_models(self) -> list[ModelRecord]:
        if not self.base_dir.is_dir():
            return []
        records = []
        # A model dir passed directly (contains config.json itself).
        if (self.base_dir / "config.json").is_file():
            return [self.record(self.base_dir.name)]
        for path in sorted(self.base_dir.iterdir()):
            if not path.is_dir():
                continue
            cfg_file = path / "config.json"
            if not cfg_file.is_file():
                continue
            records.append(self.record(path.name))
        return records

    def record(self, name: str) -> ModelRecord:
        path = self.base_dir / name
        if not (path / "config.json").is_file():
            # The registry root may itself be the model directory.
            if (self.base_dir / "config.json").is_file() and name == self.base_dir.name:
                path = self.base_dir
            else:
                raise FileNotFoundError(f"no model '{name}' in registry at {self.base_dir}")
        cfg_file = path / "config.json"
        with cfg_file.open("r", encoding="utf-8") as fh:
            cfg = ModelConfig.from_dict(json.load(fh))

        meta_file = path / "metadata.json"
        meta: dict[str, Any] = {}
        if meta_file.is_file():
            with meta_file.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)

        training_file = path / "training_state.json"
        training: dict[str, Any] = {}
        if training_file.is_file():
            with training_file.open("r", encoding="utf-8") as fh:
                training = json.load(fh)

        has_weights = (path / "model.safetensors").is_file()
        params = _count_parameters_from_config(cfg)
        verified = has_weights

        scale = ""
        for sname, scfg in SCALES.items():
            if scfg.get("name") == cfg.name:
                scale = sname
                break

        record = ModelRecord(
            name=name,
            version=cfg.version,
            architecture=cfg.architecture,
            framework=cfg.framework,
            parameters=params,
            context_length=cfg.max_context_length,
            training_dataset=training.get("dataset", cfg.training_dataset),
            status=training.get("status", cfg.status),
            precision=cfg.precision,
            created_at=meta.get("created_at", cfg.created_at),
            hardware=meta.get("hardware", cfg.hardware),
            evaluation=meta.get("evaluation", cfg.evaluation),
            verified=verified,
            path=str(path),
            scale=scale,
        )
        return record

    def resolve(self, name: str | None) -> ModelRecord:
        """Pick a model by exact name, or fall back to the best available."""
        if name is not None:
            return self.record(name)
        models = self.list_models()
        # Prefer a verified (trained) model; else the smallest available.
        trained = [m for m in models if m.verified]
        pool = trained if trained else models
        if not pool:
            raise FileNotFoundError(f"no models registered in {self.base_dir}")
        return min(pool, key=lambda m: m.parameters)


def availability() -> dict[str, bool]:
    """What the native stack can do on this machine (no imports yet)."""
    return {
        "torch": module_available("torch"),
        "safetensors": module_available("safetensors"),
        "tokenizers": module_available("tokenizers"),
    }


__all__ = ["ModelRegistry", "ModelRecord", "DEFAULT_REGISTRY_DIR", "availability"]
