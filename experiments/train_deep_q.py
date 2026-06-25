"""Train a GPU-capable Deep Q-Network Quoridor policy."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quoridor.training.deep_q import save_deep_q_checkpoint, train_deep_q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.6)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--target-update-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/deep_q_policy.pt"))
    args = parser.parse_args()

    start = time.perf_counter()
    model, stats = train_deep_q(
        episodes=args.episodes,
        max_turns=args.max_turns,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        gamma=args.gamma,
        lr=args.lr,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        target_update_interval=args.target_update_interval,
        seed=args.seed,
        device=args.device,
    )
    elapsed = time.perf_counter() - start
    stats = type(stats)(
        episodes=stats.episodes,
        wins=stats.wins,
        draws=stats.draws,
        updates=stats.updates,
        final_epsilon=stats.final_epsilon,
        device=stats.device,
        elapsed_seconds=elapsed,
    )
    save_deep_q_checkpoint(model, args.output, stats, hidden_size=args.hidden_size, metadata={"elapsed_seconds": elapsed})
    print(
        f"episodes={stats.episodes} wins={stats.wins} draws={stats.draws} updates={stats.updates} "
        f"final_epsilon={stats.final_epsilon:.3f} device={stats.device} elapsed={elapsed:.2f}s "
        f"wrote={args.output}"
    )


if __name__ == "__main__":
    main()
