"""Generate batched AlphaZero self-play data and optionally train on it."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quoridor.training.alphazero import (
    AlphaZeroExample,
    generate_alphazero_self_play_examples,
    save_alphazero_checkpoint,
    train_alphazero_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--action-limit", type=int, default=16)
    parser.add_argument("--wall-limit", type=int, default=8)
    parser.add_argument("--temperature-turns", type=int, default=12)
    parser.add_argument("--draw-value-mode", choices=("zero", "heuristic"), default="heuristic")
    parser.add_argument("--draw-value-scale", type=float, default=40.0)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-noise-fraction", type=float, default=0.25)
    parser.add_argument("--mcts-batch-size", type=int, default=8)
    parser.add_argument("--inference-cache-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--initial-checkpoint", default=None)
    parser.add_argument("--examples-output", default="experiments/results/alphazero_selfplay_examples.pt")
    parser.add_argument("--train-output", default=None)
    args = parser.parse_args()

    examples, self_play_stats = generate_alphazero_self_play_examples(
        games=args.games,
        simulations=args.simulations,
        max_turns=args.max_turns,
        hidden_size=args.hidden_size,
        action_limit=args.action_limit,
        wall_limit=args.wall_limit,
        temperature_turns=args.temperature_turns,
        draw_value_mode=args.draw_value_mode,
        draw_value_scale=args.draw_value_scale,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_noise_fraction=args.root_noise_fraction,
        mcts_batch_size=args.mcts_batch_size,
        inference_cache_size=args.inference_cache_size,
        seed=args.seed,
        device=args.device,
        initial_checkpoint=args.initial_checkpoint,
        workers=args.workers,
    )

    examples_output = Path(args.examples_output)
    examples_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "examples": examples,
            "self_play_stats": asdict(self_play_stats),
            "config": vars(args),
        },
        examples_output,
    )

    train_stats = None
    if args.train_output:
        model, train_stats = train_alphazero_examples(
            examples,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            initial_checkpoint=args.initial_checkpoint,
        )
        save_alphazero_checkpoint(
            model,
            args.train_output,
            hidden_size=args.hidden_size,
            metadata={
                "source_examples": str(examples_output),
                "self_play_stats": asdict(self_play_stats),
                "train_stats": asdict(train_stats),
                "config": vars(args),
            },
        )

    print(
        "batched alphazero self-play complete",
        {
            "examples_output": str(examples_output),
            "train_output": args.train_output,
            "self_play": asdict(self_play_stats),
            "train": None if train_stats is None else asdict(train_stats),
        },
    )


def load_examples(path: str | Path) -> list[AlphaZeroExample]:
    payload = torch.load(Path(path), map_location="cpu")
    return list(payload["examples"])


if __name__ == "__main__":
    main()
