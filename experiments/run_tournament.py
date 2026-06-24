"""Run a small Quoridor agent tournament from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quoridor.agents import GreedyBFSAgent, MCTSAgent, MinimaxAgent, PathLureAgent, RandomAgent
from quoridor.evaluation import AgentSpec, run_round_robin


def build_specs(preset: str) -> list[AgentSpec]:
    if preset == "smoke":
        return [
            AgentSpec("random", lambda: RandomAgent(seed=0)),
            AgentSpec("greedy_bfs", lambda: GreedyBFSAgent(seed=1)),
            AgentSpec(
                "path_lure",
                lambda: PathLureAgent(seed=4, action_limit=4, wall_limit=2, victim_action_limit=4),
            ),
        ]

    if preset == "full":
        return [
            AgentSpec("random", lambda: RandomAgent(seed=0)),
            AgentSpec("greedy_bfs", lambda: GreedyBFSAgent(seed=1, action_limit=16, wall_limit=8)),
            AgentSpec("minimax_d1", lambda: MinimaxAgent(depth=1, action_limit=8, wall_limit=4, seed=2)),
            AgentSpec("mcts_5", lambda: MCTSAgent(iterations=5, rollout_depth=4, action_limit=6, wall_limit=3, seed=3)),
            AgentSpec(
                "path_lure",
                lambda: PathLureAgent(seed=4, action_limit=4, wall_limit=2, victim_action_limit=4),
            ),
        ]

    if preset != "research":
        raise ValueError("preset must be 'smoke', 'full', or 'research'")

    return [
        AgentSpec("random", lambda: RandomAgent(seed=0)),
        AgentSpec("greedy_bfs", lambda: GreedyBFSAgent(seed=1)),
        AgentSpec("minimax_d1", lambda: MinimaxAgent(depth=1, action_limit=12, wall_limit=8, seed=2)),
        AgentSpec("mcts_30", lambda: MCTSAgent(iterations=30, rollout_depth=10, action_limit=12, wall_limit=8, seed=3)),
        AgentSpec("path_lure", lambda: PathLureAgent(seed=4)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "full", "research"], default="smoke")
    parser.add_argument("--games-per-pair", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/tournament_games.csv"))
    args = parser.parse_args()

    result = run_round_robin(build_specs(args.preset), games_per_pair=args.games_per_pair, max_turns=args.max_turns)
    result.write_games_csv(args.output)

    for row in result.standings():
        print(
            f"{row['agent']:12s} games={row['games']:3d} "
            f"wins={row['wins']:3d} draws={row['draws']:3d} "
            f"win_rate={row['win_rate']:.2f} elo={row['elo']:.1f}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
