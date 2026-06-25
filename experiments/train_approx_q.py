"""Train the linear approximate Q-learning Quoridor policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quoridor.training.approx_q_learning import save_trained_weights, train_approx_q_learning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/approx_q_policy.json"))
    args = parser.parse_args()

    weights, stats = train_approx_q_learning(
        episodes=args.episodes,
        max_turns=args.max_turns,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        seed=args.seed,
    )
    save_trained_weights(weights, stats, str(args.output))
    print(
        f"episodes={stats.episodes} wins={stats.wins} draws={stats.draws} "
        f"nonzero_weights={stats.nonzero_weights} final_epsilon={stats.final_epsilon:.3f} "
        f"wrote={args.output}"
    )


if __name__ == "__main__":
    main()
