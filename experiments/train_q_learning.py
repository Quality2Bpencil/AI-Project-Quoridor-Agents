"""Train a lightweight tabular Q-learning Quoridor policy."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quoridor.training.q_learning import save_trained_q_table, train_q_learning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=120)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.35)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/q_learning_policy.json"))
    args = parser.parse_args()

    start = time.perf_counter()
    q_table, stats = train_q_learning(
        episodes=args.episodes,
        max_turns=args.max_turns,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - start
    save_trained_q_table(q_table, stats, str(args.output))
    print(
        f"trained episodes={stats.episodes} wins={stats.wins} draws={stats.draws} "
        f"q_states={stats.q_states} q_entries={stats.q_entries} "
        f"final_epsilon={stats.final_epsilon:.3f} elapsed={elapsed:.2f}s"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
