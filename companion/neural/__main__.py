"""CLI for the Relay Neural stack: ``python -m companion.neural``.

Commands:
- ``info``            → detect hardware, recommend a scale
- ``list``            → registry contents with real param counts
- ``tokenizer-train`` → train a BPE tokenizer from a text file
- ``train``           → train a model from a text corpus
- ``generate``        → generate text with a trained model
- ``eval``            → run the evaluation battery on a checkpoint
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_info(args: argparse.Namespace) -> int:
    from .selection import detect_hardware, recommend_model

    hw = detect_hardware()
    rec = recommend_model(hw)
    print(json.dumps({"hardware": hw, "recommendation": rec.to_dict()}, indent=2))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from .registry import ModelRegistry

    registry = ModelRegistry(Path(args.dir))
    models = registry.list_models()
    if not models:
        print(f"no models registered in {args.dir}")
        return 0
    for m in models:
        print(f"{m.name:20s} params={m.parameters:>10,} ctx={m.context_length:>6} status={m.status:>12} verified={m.verified} scale={m.scale}")
    return 0


def _cmd_tokenizer_train(args: argparse.Namespace) -> int:
    from .tokenizer import train_tokenizer

    texts: list[str] = []
    for path in args.inputs:
        texts.extend(line.strip() for line in Path(path).open("r", encoding="utf-8", errors="replace") if line.strip())
    tok = train_tokenizer(texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency)
    tok.save(args.output)
    print(json.dumps(tok.to_manifest(), indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from ..tools.common import module_available

    if not module_available("torch"):
        print("error: torch is required for training", file=sys.stderr)
        return 1

    from .architecture import ModelConfig, RelayTransformer
    from .models.scales import scale_config
    from .registry import ModelRegistry
    from .tokenizer import train_tokenizer
    from .training import TextDataset, TrainConfig, train

    scale = args.scale or "nano"
    cfg = ModelConfig(**scale_config(scale))
    if args.vocab_size:
        cfg.vocab_size = args.vocab_size

    texts: list[str] = []
    for path in args.inputs:
        texts.extend(line.strip() for line in Path(path).open("r", encoding="utf-8", errors="replace") if line.strip())
    if not texts:
        print("error: no training texts provided", file=sys.stderr)
        return 1

    tok = train_tokenizer(texts, vocab_size=min(cfg.vocab_size, 4096))

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))

    model = RelayTransformer(cfg)
    cache = str(out_dir / ".token_cache.pt")
    ds = TextDataset.from_texts(texts, tok, source=",".join(args.inputs), tokenizer_path=str(out_dir / "tokenizer.json"), cache_path=cache)

    tc = TrainConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        total_steps=args.steps,
        warmup_steps=max(1, args.steps // 10),
        save_every=max(1, args.steps // 4),
        log_every=max(1, args.steps // 20),
        seed=args.seed,
        device=args.device,
    )
    result = train(model, tok, lambda: ds.to_dataloader(tc.batch_size, tc.seq_len, seed=args.seed), tc, checkpoint_dir=str(out_dir))
    print(json.dumps({"steps": result.steps_run, "final_loss": round(result.final_loss, 4), "best_loss": round(result.best_loss, 4), "duration_s": round(result.duration_seconds, 2)}, indent=2))
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from ..tools.common import module_available

    if not module_available("torch"):
        print("error: torch is required for generation", file=sys.stderr)
        return 1

    from .inference.generator import GenerationConfig, generate
    from .training import load_model, load_tokenizer
    from .training.trainer import resolve_device

    model = load_model(args.model)
    tok = load_tokenizer(args.model)
    device = resolve_device(args.device)
    gen = generate(model, tok, args.prompt, GenerationConfig(max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed), device=device)
    print(gen.text)
    print(f"[tokens={gen.generated_tokens} tps={gen.tokens_per_second:.1f} dur={gen.duration_s:.2f}s]", file=sys.stderr)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from ..tools.common import module_available

    if not module_available("torch"):
        print("error: torch is required for evaluation", file=sys.stderr)
        return 1

    from .evaluation import run_evaluation
    from .training import load_model, load_tokenizer
    from .training.trainer import resolve_device

    model = load_model(args.model)
    tok = load_tokenizer(args.model)
    device = resolve_device(args.device)
    results = run_evaluation(model, tok, device=device, checkpoint_dir=args.model)
    print(json.dumps(results, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="companion.neural", description="Relay native neural stack")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="hardware detection + recommendation")

    p = sub.add_parser("list", help="list registered models")
    p.add_argument("--dir", default=str(Path(__file__).resolve().parent.parent.parent / "data" / "neural"))

    p = sub.add_parser("tokenizer-train", help="train a BPE tokenizer")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--min-frequency", type=int, default=1)

    p = sub.add_parser("train", help="train a model")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--scale", choices=["nano", "small", "medium", "large", "x"], default="nano")
    p.add_argument("--vocab-size", type=int, default=0)
    p.add_argument("--outdir", default=str(Path(__file__).resolve().parent.parent.parent / "data" / "neural" / "relay-nano"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")

    p = sub.add_parser("generate", help="generate text from a trained model")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto")

    p = sub.add_parser("eval", help="run the evaluation battery")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="auto")

    args = parser.parse_args(argv)
    handlers = {
        "info": _cmd_info,
        "list": _cmd_list,
        "tokenizer-train": _cmd_tokenizer_train,
        "train": _cmd_train,
        "generate": _cmd_generate,
        "eval": _cmd_eval,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
