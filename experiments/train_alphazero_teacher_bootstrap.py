"""Bootstrap AlphaZero from strong heuristic teachers before self-play."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quoridor.training.alphazero import save_alphazero_checkpoint, train_alphazero_examples
from quoridor.training.teacher_bootstrap import generate_teacher_bootstrap_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--teacher-profile", choices=("fast", "mixed_strong"), default="mixed_strong")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature-turns", type=int, default=12)
    parser.add_argument("--draw-value-scale", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--output", default="experiments/results/alphazero_teacher_bootstrap.pt")
    args = parser.parse_args()

    examples, teacher_stats = generate_teacher_bootstrap_examples(
        games=args.games,
        max_turns=args.max_turns,
        seed=args.seed,
        workers=args.workers,
        teacher_profile=args.teacher_profile,
        draw_value_scale=args.draw_value_scale,
        temperature_turns=args.temperature_turns,
    )
    model, train_stats = train_alphazero_examples(
        examples,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        initial_checkpoint=args.resume_from,
    )
    metadata = {
        "teacher": asdict(teacher_stats),
        "train": asdict(train_stats),
        "args": vars(args),
    }
    save_alphazero_checkpoint(model, args.output, metadata=metadata)
    print(
        "trained alphazero teacher bootstrap",
        {
            "output": args.output,
            "teacher": asdict(teacher_stats),
            "train": asdict(train_stats),
        },
    )


if __name__ == "__main__":
    main()
