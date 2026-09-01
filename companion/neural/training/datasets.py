"""Datasets: load real text corpora and turn them into tokenized tensors.

Two shapes are supported:
- ``TextDataset`` — plain / JSONL text lines for causal LM pretraining, and
- ``InstructionDataset`` — {"input", "output"} pairs for supervised finetuning.

Tokenization happens once up-front and is cached to disk, so re-runs don't
re-tokenize. No synthetic/placeholder data is ever injected.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch

from ...tools.common import module_available
from ..tokenizer import RelayBpeTokenizer

_HAS_TORCH = module_available("torch")


def _iterate_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def _iterate_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass
class TextDataset:
    """A tokenized causal-LM dataset.

    ``data`` is a flat LongTensor of token ids. ``n_tokens`` is the real count.
    """

    data: torch.Tensor
    n_tokens: int
    source: str
    tokenizer_path: str

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        tokenizer: RelayBpeTokenizer,
        source: str = "inline",
        tokenizer_path: str = "",
        cache_path: Optional[str] = None,
        seed: int = 42,
    ) -> "TextDataset":
        if not _HAS_TORCH:
            raise RuntimeError("torch is required to build a TextDataset")
        if cache_path and Path(cache_path).is_file():
            return cls.from_cache(cache_path, source, tokenizer_path)
        ids: list[int] = []
        for text in texts:
            ids.extend(tokenizer.encode(text))
        ids.append(0)  # EOS-like boundary token (<pad> id 0 by convention)
        data = torch.tensor(ids, dtype=torch.long)
        ds = cls(data=data, n_tokens=data.numel(), source=source, tokenizer_path=tokenizer_path)
        if cache_path:
            torch.save({"data": data, "source": source, "tokenizer_path": tokenizer_path}, cache_path)
        return ds

    @classmethod
    def from_files(
        cls,
        files: list[str],
        tokenizer: RelayBpeTokenizer,
        source: str = "files",
        tokenizer_path: str = "",
        cache_path: Optional[str] = None,
        seed: int = 42,
    ) -> "TextDataset":
        def gen() -> Iterable[str]:
            for f in files:
                p = Path(f)
                if p.suffix == ".jsonl":
                    for row in _iterate_jsonl(p):
                        text = row.get("text") or row.get("input") or row.get("output")
                        if text:
                            yield str(text)
                else:
                    yield from _iterate_lines(p)

        return cls.from_texts(gen(), tokenizer, source=source, tokenizer_path=tokenizer_path, cache_path=cache_path, seed=seed)

    @classmethod
    def from_cache(cls, cache_path: str, source: str = "cache", tokenizer_path: str = "") -> "TextDataset":
        data = torch.load(cache_path, weights_only=True)
        return cls(data=data["data"], n_tokens=data["data"].numel(), source=data.get("source", source), tokenizer_path=data.get("tokenizer_path", tokenizer_path))

    def to_dataloader(self, batch_size: int, seq_len: int, seed: int = 42) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (input_ids, target_ids) blocks of shape (batch, seq_len)."""
        n = (self.n_tokens - 1) // (batch_size * seq_len) * (batch_size * seq_len)
        if n < 1:
            raise ValueError("dataset too small for batch_size*seq_len")
        buf = self.data[: n + 1]
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(n // (batch_size * seq_len), generator=g)
        for block_idx in perm:
            start = block_idx * batch_size * seq_len
            block = buf[start : start + batch_size * seq_len + 1]
            xs = block[:-1].view(batch_size, seq_len)
            ys = block[1:].view(batch_size, seq_len)
            yield xs, ys


@dataclass
class InstructionDataset:
    """Supervised instruction examples: (input, output) token sequences."""

    inputs: list[list[int]]
    outputs: list[list[int]]

    @classmethod
    def from_jsonl(cls, path: str, tokenizer: RelayBpeTokenizer) -> "InstructionDataset":
        inputs, outputs = [], []
        for row in _iterate_jsonl(Path(path)):
            if "input" in row and "output" in row:
                inputs.append(tokenizer.encode(str(row["input"])))
                outputs.append(tokenizer.encode(str(row["output"])))
        return cls(inputs=inputs, outputs=outputs)

    def __len__(self) -> int:
        return len(self.inputs)

    def to_batches(self, batch_size: int) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, len(self.inputs), batch_size):
            xs, ys = [], []
            for i in range(start, min(start + batch_size, len(self.inputs))):
                xs.append(torch.tensor(self.inputs[i], dtype=torch.long))
                ys.append(torch.tensor(self.outputs[i], dtype=torch.long))
            xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
            ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
            yield xs, ys


__all__ = ["TextDataset", "InstructionDataset"]
