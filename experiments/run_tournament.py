"""Run a small Quoridor agent tournament from the command line."""

from __future__ import annotations

import argparse
from functools import partial
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quoridor.agents import (
    ApproxQLearningAgent,
    ArgmaxQTrapAgent,
    CounterfactualTrapAgent,
    DeepQAgent,
    DepthTrapAgent,
    GreedyBFSAgent,
    MCTSAgent,
    MinimaxAgent,
    PathLureAgent,
    PUCTAgent,
    QLearningAgent,
    RandomAgent,
    RolloutPoisonAgent,
)
from quoridor.evaluation import (
    AgentSpec,
    arena_rows_to_result,
    build_round_robin_tasks,
    play_arena_task,
    read_arena_rows,
    run_arena,
    run_round_robin,
    write_matchup_matrix_csv,
    write_score_matrix_csv,
)


def build_specs(preset: str, seed_offset: int = 0) -> list[AgentSpec]:
    def seed(value: int) -> int:
        return value + seed_offset

    if preset == "smoke":
        return [
            AgentSpec("random", lambda: RandomAgent(seed=seed(0))),
            AgentSpec("greedy_bfs", lambda: GreedyBFSAgent(seed=seed(1))),
            AgentSpec(
                "path_lure",
                lambda: PathLureAgent(seed=seed(4), action_limit=4, wall_limit=2, victim_action_limit=4),
            ),
        ]

    if preset == "full":
        return [
            AgentSpec("random", lambda: RandomAgent(seed=seed(0))),
            AgentSpec("greedy_bfs", lambda: GreedyBFSAgent(seed=seed(1), action_limit=16, wall_limit=8)),
            AgentSpec("minimax_d1", lambda: MinimaxAgent(depth=1, action_limit=8, wall_limit=4, seed=seed(2))),
            AgentSpec("mcts_5", lambda: MCTSAgent(iterations=5, rollout_depth=4, action_limit=6, wall_limit=3, seed=seed(3))),
            AgentSpec("puct_4", lambda: PUCTAgent(simulations=4, action_limit=6, wall_limit=3)),
            AgentSpec(
                "path_lure",
                lambda: PathLureAgent(seed=seed(4), action_limit=4, wall_limit=2, victim_action_limit=4),
            ),
        ]

    if preset == "adversarial":
        return [
            AgentSpec("greedy_bfs", lambda: GreedyBFSAgent(seed=seed(1), action_limit=16, wall_limit=8)),
            AgentSpec("minimax_d1", lambda: MinimaxAgent(depth=1, action_limit=8, wall_limit=4, seed=seed(2))),
            AgentSpec("mcts_5", lambda: MCTSAgent(iterations=5, rollout_depth=4, action_limit=6, wall_limit=3, seed=seed(3))),
            AgentSpec("puct_4", lambda: PUCTAgent(simulations=4, action_limit=6, wall_limit=3)),
            AgentSpec(
                "path_lure",
                lambda: PathLureAgent(seed=seed(4), action_limit=8, wall_limit=4, victim_action_limit=6),
            ),
            AgentSpec(
                "depth_trap",
                lambda: DepthTrapAgent(seed=seed(5), action_limit=4, wall_limit=2, victim_action_limit=4, followup_limit=4),
            ),
            AgentSpec(
                "rollout_poison",
                lambda: RolloutPoisonAgent(
                    seed=seed(6),
                    action_limit=4,
                    wall_limit=2,
                    victim_action_limit=3,
                    rollout_depth=1,
                ),
            ),
            AgentSpec(
                "counter_trap",
                lambda: CounterfactualTrapAgent(
                    seed=seed(7),
                    action_limit=4,
                    wall_limit=2,
                    victim_action_limit=3,
                    response_width=2,
                    followup_limit=3,
                ),
            ),
            AgentSpec(
                "argmax_q_trap",
                lambda: ArgmaxQTrapAgent(
                    seed=seed(8),
                    action_limit=6,
                    wall_limit=4,
                    victim_action_limit=4,
                    response_width=1,
                    followup_limit=4,
                ),
            ),
        ]

    if preset == "trap_eval":
        return [
            AgentSpec("greedy_bfs", lambda: GreedyBFSAgent(seed=seed(1), action_limit=16, wall_limit=8)),
            AgentSpec("minimax_d1", lambda: MinimaxAgent(depth=1, action_limit=8, wall_limit=4, seed=seed(2))),
            AgentSpec("mcts_5", lambda: MCTSAgent(iterations=5, rollout_depth=4, action_limit=6, wall_limit=3, seed=seed(3))),
            AgentSpec("q_learning", lambda: QLearningAgent(seed=seed(8), table_path=Path("experiments/results/q_learning_policy.json"))),
            AgentSpec(
                "path_lure",
                lambda: PathLureAgent(seed=seed(4), action_limit=10, wall_limit=6, victim_action_limit=8),
            ),
            AgentSpec(
                "depth_trap",
                lambda: DepthTrapAgent(seed=seed(5), action_limit=4, wall_limit=2, victim_action_limit=4, followup_limit=4),
            ),
            AgentSpec(
                "rollout_poison",
                lambda: RolloutPoisonAgent(
                    seed=seed(6),
                    action_limit=4,
                    wall_limit=2,
                    victim_action_limit=3,
                    rollout_depth=1,
                ),
            ),
            AgentSpec(
                "counter_trap",
                lambda: CounterfactualTrapAgent(
                    seed=seed(7),
                    action_limit=4,
                    wall_limit=2,
                    victim_action_limit=3,
                    response_width=2,
                    followup_limit=3,
                ),
            ),
            AgentSpec(
                "argmax_q_trap",
                lambda: ArgmaxQTrapAgent(
                    seed=seed(8),
                    action_limit=8,
                    wall_limit=5,
                    victim_action_limit=5,
                    response_width=1,
                    followup_limit=5,
                ),
            ),
        ]

    if preset != "research":
        raise ValueError("preset must be 'smoke', 'full', 'adversarial', 'trap_eval', or 'research'")

    return [
        AgentSpec("random", lambda: RandomAgent(seed=seed(0))),
        AgentSpec("greedy_bfs", lambda: GreedyBFSAgent(seed=seed(1))),
        AgentSpec("minimax_d1", lambda: MinimaxAgent(depth=1, action_limit=12, wall_limit=8, seed=seed(2))),
        AgentSpec("mcts_30", lambda: MCTSAgent(iterations=30, rollout_depth=10, action_limit=12, wall_limit=8, seed=seed(3))),
        AgentSpec("puct_24", lambda: PUCTAgent(simulations=24, action_limit=12, wall_limit=8)),
        AgentSpec("q_learning", lambda: QLearningAgent(seed=seed(8), table_path=Path("experiments/results/q_learning_policy.json"))),
        AgentSpec(
            "approx_q",
            lambda: ApproxQLearningAgent(seed=seed(9), weights_path=Path("experiments/results/approx_q_policy.json")),
        ),
        AgentSpec(
            "deep_q",
            lambda: DeepQAgent(seed=seed(11), checkpoint_path=Path("experiments/results/deep_q_policy.pt")),
        ),
        AgentSpec("path_lure", lambda: PathLureAgent(seed=seed(4))),
        AgentSpec("depth_trap", lambda: DepthTrapAgent(seed=seed(5))),
        AgentSpec("rollout_poison", lambda: RolloutPoisonAgent(seed=seed(6))),
        AgentSpec("counter_trap", lambda: CounterfactualTrapAgent(seed=seed(7))),
        AgentSpec("argmax_q_trap", lambda: ArgmaxQTrapAgent(seed=seed(10))),
    ]


def _play_preset_arena_task(preset: str, task) -> dict[str, object]:
    specs_by_name = {spec.name: spec for spec in build_specs(preset, seed_offset=task.seed)}
    return play_arena_task(task, specs_by_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "full", "adversarial", "trap_eval", "research"], default="smoke")
    parser.add_argument("--games-per-pair", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/tournament_games.csv"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--matrix-output", type=Path)
    parser.add_argument("--score-matrix-output", type=Path)
    args = parser.parse_args()

    specs = build_specs(args.preset)
    use_arena = args.workers > 1 or args.resume or args.matrix_output or args.score_matrix_output
    if use_arena:
        tasks = build_round_robin_tasks(specs, games_per_pair=args.games_per_pair, max_turns=args.max_turns)
        rows_written = run_arena(
            tasks,
            partial(_play_preset_arena_task, args.preset),
            output=args.output,
            workers=args.workers,
            resume=args.resume,
            progress_interval=args.progress_interval,
        )
        rows = read_arena_rows(args.output)
        result = arena_rows_to_result(rows)

        if args.matrix_output is not None:
            write_matchup_matrix_csv(rows, args.matrix_output)
        if args.score_matrix_output is not None:
            write_score_matrix_csv(rows, args.score_matrix_output)
        skipped = len(tasks) - len(rows_written)
        print(
            f"arena tasks={len(tasks)} completed_now={len(rows_written)} "
            f"skipped_or_existing={skipped} workers={args.workers} output={args.output}"
        )
    else:
        result = run_round_robin(specs, games_per_pair=args.games_per_pair, max_turns=args.max_turns)
        result.write_games_csv(args.output)

    for row in result.standings():
        print(
            f"{row['agent']:12s} games={row['games']:3d} "
            f"wins={row['wins']:3d} draws={row['draws']:3d} "
            f"win_rate={row['win_rate']:.2f} elo={row['elo']:.1f} "
            f"trap={row['avg_trap_events']:.2f} walls={row['avg_wall_actions']:.2f} "
            f"path_delta={row['avg_path_delta']:.2f}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
