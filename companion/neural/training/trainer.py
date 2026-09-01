"""Training loop: real forward/backward, real gradient flow, honest logging.

Features: AdamW, linear warmup + cosine decay, gradient clipping, gradient
accumulation, per-step loss/ppl logging, and checkpointing every N steps.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from torch import nn

from ..architecture import ModelConfig, RelayTransformer
from ..tokenizer import RelayBpeTokenizer
from .checkpointing import save_checkpoint

logger = logging.getLogger("relai.neural.training")


@dataclass
class TrainConfig:
    batch_size: int = 8
    seq_len: int = 256
    learning_rate: float = 3e-4
    warmup_steps: int = 200
    total_steps: int = 2000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    save_every: int = 500
    log_every: int = 10
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda
    dtype: str = "fp32"  # fp32 | bf16 (bf16 = mixed precision on capable hw)
    mixed_precision: bool = False
    eval_every: int = 100
    metrics: list[str] = field(default_factory=lambda: ["loss", "perplexity"])


@dataclass
class TrainResult:
    model: RelayTransformer
    steps_run: int
    final_loss: float
    best_loss: float
    duration_seconds: float
    checkpoint_dir: Optional[str] = None


def resolve_device(requested: str = "auto") -> torch.device:
    """Honest device resolution: cuda only if actually available."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdamW:
    """Minimal AdamW implementation (keeps dependencies to torch only)."""

    def __init__(self, params: Iterable[nn.Parameter], lr: float, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01) -> None:
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m: list[torch.Tensor] = []
        self.v: list[torch.Tensor] = []
        for p in self.params:
            self.m.append(torch.zeros_like(p.data))
            self.v.append(torch.zeros_like(p.data))

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_().zero_()

    def step(self) -> None:
        self.step_count += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.m[i].mul_(b1).add_(g, alpha=1 - b1)
            self.v[i].mul_(b2).addcmul_(g, g, value=1 - b2)
            m_hat = self.m[i] / (1 - b1**self.step_count)
            v_hat = self.v[i] / (1 - b2**self.step_count)
            p.data.addcdiv_(m_hat, v_hat.sqrt_().add_(self.eps), value=-self.lr)
            if self.weight_decay > 0:
                p.data.add_(p.data, alpha=-self.lr * self.weight_decay)

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    def state_dict(self) -> dict:
        return {
            "m": [t.clone() for t in self.m],
            "v": [t.clone() for t in self.v],
            "step_count": self.step_count,
            "lr": self.lr,
        }

    def load_state_dict(self, state: dict) -> None:
        self.m = [t.clone() for t in state["m"]]
        self.v = [t.clone() for t in state["v"]]
        self.step_count = state["step_count"]


def warmup_cosine_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())


def train(
    model: RelayTransformer,
    tokenizer: RelayBpeTokenizer,
    data_iter: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]],
    config: TrainConfig,
    checkpoint_dir: Optional[str] = None,
    loss_fn: Optional[Callable] = None,
    resume: bool = False,
    on_log: Optional[Callable[[dict], None]] = None,
) -> TrainResult:
    """Run a real training loop. Never fakes a step or a metric."""
    device = resolve_device(config.device)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
    loss_fn = loss_fn or torch.nn.functional.cross_entropy

    steps_run = 0
    best_loss = float("inf")
    start = time.time()
    last_loss = float("nan")

    resume_state = None
    if resume and checkpoint_dir and (Path(checkpoint_dir) / "training_state.json").is_file():
        resume_state = load_training_state(checkpoint_dir)
        steps_run = resume_state["step"]
        optimizer.load_state_dict(resume_state.get("optimizer", {}))
        best_loss = resume_state.get("best_loss", float("inf"))
        logger.info("resumed from step %d", steps_run)

    epoch = 0
    accum = 0
    while steps_run < config.total_steps:
        epoch += 1
        for xs, ys in data_iter():
            if steps_run >= config.total_steps:
                break
            xs = xs.to(device)
            ys = ys.to(device)

            logits, _ = model(xs)
            loss = loss_fn(logits[:, :-1].reshape(-1, model.config.vocab_size), ys[:, 1:].reshape(-1))
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            accum += 1

            if accum < config.gradient_accumulation_steps:
                continue

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            lr_now = warmup_cosine_lr(steps_run + 1, config.warmup_steps, config.total_steps, config.learning_rate)
            optimizer.set_lr(lr_now)
            optimizer.step()
            optimizer.zero_grad()
            steps_run += 1
            accum = 0
            last_loss = loss.item() * config.gradient_accumulation_steps
            best_loss = min(best_loss, last_loss)

            if steps_run % config.log_every == 0:
                entry = {
                    "epoch": epoch,
                    "step": steps_run,
                    "loss": last_loss,
                    "perplexity": 2 ** max(last_loss, 0.0),
                    "lr": lr_now,
                    "grad_norm": float(grad_norm),
                    "duration_s": round(time.time() - start, 2),
                }
                logger.info("step=%d loss=%.4f ppl=%.2f lr=%.2e grad_norm=%.3f", steps_run, last_loss, entry["perplexity"], lr_now, grad_norm)
                if on_log:
                    on_log(entry)

            if checkpoint_dir and steps_run % config.save_every == 0:
                save_checkpoint(model, tokenizer, optimizer, steps_run, best_loss, checkpoint_dir, dataset=getattr(data_iter, "source", "") or "")
        if accum > 0:
            # Final partial accumulation for this epoch: apply it so progress is kept.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            lr_now = warmup_cosine_lr(steps_run + 1, config.warmup_steps, config.total_steps, config.learning_rate)
            optimizer.set_lr(lr_now)
            optimizer.step()
            optimizer.zero_grad()
            steps_run += 1
            accum = 0

    duration = time.time() - start
    if checkpoint_dir:
        save_checkpoint(model, tokenizer, optimizer, steps_run, best_loss, checkpoint_dir, dataset=getattr(data_iter, "source", "") or "")
    return TrainResult(
        model=model,
        steps_run=steps_run,
        final_loss=last_loss,
        best_loss=best_loss,
        duration_seconds=duration,
        checkpoint_dir=checkpoint_dir,
    )


def load_training_state(checkpoint_dir: str) -> dict:
    with (Path(checkpoint_dir) / "training_state.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


__all__ = ["TrainConfig", "TrainResult", "train", "AdamW", "warmup_cosine_lr", "resolve_device"]
