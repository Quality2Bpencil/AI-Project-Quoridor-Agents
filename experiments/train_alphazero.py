"""Train an AlphaZero-style policy/value checkpoint from self-play."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quoridor.training.alphazero import save_alphazero_checkpoint, train_alphazero_self_play


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--action-limit", type=int, default=10)
    parser.add_argument("--wall-limit", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-per-game", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature-turns", type=int, default=8)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--output", default="experiments/results/alphazero_policy_value.pt")
    args = parser.parse_args()

    model, stats = train_alphazero_self_play(
        games=args.games,
        simulations=args.simulations,
        max_turns=args.max_turns,
        hidden_size=args.hidden_size,
        action_limit=args.action_limit,
        wall_limit=args.wall_limit,
        batch_size=args.batch_size,
        epochs_per_game=args.epochs_per_game,
        lr=args.lr,
        temperature_turns=args.temperature_turns,
        replay_capacity=args.replay_capacity,
        seed=args.seed,
        device=args.device,
        initial_checkpoint=args.resume_from,
    )
    save_alphazero_checkpoint(
        model,
        args.output,
        metadata={
            "games": stats.games,
            "examples": stats.examples,
            "updates": stats.updates,
            "wins": list(stats.wins),
            "draws": stats.draws,
            "device": stats.device,
            "elapsed_seconds": stats.elapsed_seconds,
            "simulations": args.simulations,
            "max_turns": args.max_turns,
            "action_limit": args.action_limit,
            "wall_limit": args.wall_limit,
            "replay_capacity": args.replay_capacity,
            "resume_from": args.resume_from,
        },
    )
    print(
        "trained alphazero",
        {
            "output": args.output,
            "games": stats.games,
            "examples": stats.examples,
            "updates": stats.updates,
            "wins": stats.wins,
            "draws": stats.draws,
            "device": stats.device,
            "elapsed_seconds": round(stats.elapsed_seconds, 3),
        },
    )


if __name__ == "__main__":
    main()
