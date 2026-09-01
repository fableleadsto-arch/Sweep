"""Versioned BPE tokenizer built on HuggingFace ``tokenizers``.

The tokenizer is trained from real corpus files and saved as ``tokenizer.json``
(versioned by the training dataset/version stamp). A pure-Python character
fallback exists only so unit tests can run without the ``tokenizers`` package;
production always uses BPE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from ...tools.common import module_available

_HAS_TOKENIZERS = module_available("tokenizers")


def tokenizer_available() -> bool:
    return _HAS_TOKENIZERS


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 4096,
    min_frequency: int = 1,
    special_tokens: Optional[list[str]] = None,
    seed: int = 42,
) -> "RelayBpeTokenizer":
    """Train a fresh BPE tokenizer on the given texts (real training, no fakes)."""
    if not _HAS_TOKENIZERS:
        return _CharFallbackTokenizer(vocab_size=vocab_size, special_tokens=special_tokens or [])

    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, trainers
    from tokenizers.models import BPE

    specials = special_tokens or ["<unk>", "<pad>", "<bos>", "<eos>"]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = None

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    # tokenizers.train_from_iterator expects a (possibly infinite) iterator of str.
    tokenizer.train_from_iterator(texts, trainer=trainer, length=None)

    return RelayBpeTokenizer(tokenizer=tokenizer, special_tokens=specials, vocab_size=vocab_size)


class _CharFallbackTokenizer:
    """Byte-level fallback used only when the ``tokenizers`` package is absent.

    NOT used in production; lets the tokenizer contract stay testable anywhere.
    """

    def __init__(self, vocab_size: int = 4096, special_tokens: Optional[list[str]] = None) -> None:
        specials = special_tokens or ["<unk>", "<pad>", "<bos>", "<eos>"]
        self.special_tokens = list(specials)
        self._char_to_id = {s: i for i, s in enumerate(specials)}
        # Reserve space for real bytes; vocab_size is a hint only here.
        self.vocab_size = vocab_size
        self.saved_path: Optional[str] = None

    def encode(self, text: str) -> list[int]:
        ids = []
        for ch in text:
            if ch not in self._char_to_id:
                self._char_to_id[ch] = len(self._char_to_id)
            ids.append(self._char_to_id[ch])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        rev = {v: k for k, v in self._char_to_id.items()}
        return "".join(rev[i] for i in ids if i in rev)

    def vocab(self) -> int:
        return len(self._char_to_id)

    def save(self, path: str) -> None:
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump({"fallback": True, "mapping": self._char_to_id, "special": self.special_tokens}, fh)
        self.saved_path = str(path)

    @classmethod
    def load(cls, path: str) -> "_CharFallbackTokenizer":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        tk = cls(special_tokens=data.get("special"))
        tk._char_to_id = {k: v for k, v in data["mapping"].items()}
        tk.saved_path = str(path)
        return tk


class RelayBpeTokenizer:
    """Wrapper over a HF tokenizer with encode/decode and save/load."""

    def __init__(
        self,
        tokenizer=None,
        special_tokens: Optional[list[str]] = None,
        vocab_size: int = 4096,
    ) -> None:
        self._tokenizer = tokenizer
        self._fallback = None
        self.special_tokens = special_tokens or ["<unk>", "<pad>", "<bos>", "<eos>"]
        self.vocab_size = vocab_size
        self.saved_path: Optional[str] = None
        if tokenizer is None and not _HAS_TOKENIZERS:
            self._fallback = _CharFallbackTokenizer(vocab_size=vocab_size, special_tokens=self.special_tokens)

    # ── primary API ─────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        if self._fallback is not None:
            return self._fallback.encode(text)
        return self._tokenizer.encode(text).ids

    def decode(self, ids: Iterable[int]) -> str:
        if self._fallback is not None:
            return self._fallback.decode(ids)
        return self._tokenizer.decode(list(ids))

    def vocab(self) -> int:
        if self._fallback is not None:
            return self._fallback.vocab()
        return self._tokenizer.get_vocab_size()

    @property
    def is_fallback(self) -> bool:
        return self._fallback is not None

    # ── persistence ─────────────────────────────────────────────────

    def save(self, path: str) -> None:
        path = str(path)
        if self._fallback is not None:
            self._fallback.save(path)
            self.saved_path = path
            return
        self._tokenizer.save(path)
        self.saved_path = path

    @classmethod
    def load(cls, path: str) -> "RelayBpeTokenizer":
        if not _HAS_TOKENIZERS:
            return cls(tokenizer=None, special_tokens=None)
        from tokenizers import Tokenizer as HFTokenizer

        tokenizer = HFTokenizer.from_file(path)
        tk = cls(tokenizer=tokenizer)
        tk.saved_path = str(path)
        return tk

    # ── metadata ────────────────────────────────────────────────────

    def to_manifest(self) -> dict:
        return {
            "type": "bpe" if not self.is_fallback else "char-fallback",
            "vocab_size": self.vocab(),
            "special_tokens": self.special_tokens,
            "saved_path": self.saved_path,
        }


__all__ = ["RelayBpeTokenizer", "train_tokenizer", "tokenizer_available"]
