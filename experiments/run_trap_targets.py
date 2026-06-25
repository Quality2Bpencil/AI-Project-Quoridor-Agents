"""Run focused trap-victim matchups for paper tables."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_tournament import _play_preset_arena_task, build_specs
from experiments.summarize_trap_effectiveness import DEFAULT_TARGETS, summarize_trap_effectiveness, write_summary
from quoridor.evaluation import (
    ArenaTask,
    read_arena_rows,
    run_arena,
    write_matchup_matrix_csv,
    write_score_matrix_csv,
)


def build_target_tasks(*, games_per_pair: int, max_turns: int) -> list[ArenaTask]:
    specs = {spec.name for spec in build_specs("trap_eval")}
    tasks: list[ArenaTask] = []
    pair_index = 0
    for trap_agent, target_agent in DEFAULT_TARGETS.items():
        if trap_agent not in specs or target_agent not in specs:
            raise ValueError(f"trap_eval preset is missing {trap_agent!r} or {target_agent!r}")
        pair_id = f"{trap_agent}__vs__{target_agent}"
        for game_index in range(games_per_pair):
            agent0, agent1 = (
                (trap_agent, target_agent)
                if game_index % 2 == 0
                else (target_agent, trap_agent)
            )
            tasks.append(
                ArenaTask(
                    game_id=f"target-{pair_index:04d}-{game_index:04d}",
                    pair_id=pair_id,
                    pair_index=pair_index,
                    game_index=game_index,
                    agent_a=trap_agent,
                    agent_b=target_agent,
                    agent0=agent0,
                    agent1=agent1,
                    max_turns=max_turns,
                    seed=pair_index * 100_000 + game_index,
                )
            )
        pair_index += 1
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-pair", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/trap_targets_games.csv"))
    parser.add_argument("--matrix-output", type=Path, default=Path("experiments/results/trap_targets_matchups.csv"))
    parser.add_argument("--score-matrix-output", type=Path, default=Path("experiments/results/trap_targets_scores.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("experiments/results/trap_effectiveness_targets.csv"))
    args = parser.parse_args()

    tasks = build_target_tasks(games_per_pair=args.games_per_pair, max_turns=args.max_turns)
    rows_written = run_arena(
        tasks,
        partial(_play_preset_arena_task, "trap_eval"),
        output=args.output,
        workers=args.workers,
        resume=args.resume,
        progress_interval=args.progress_interval,
    )
    rows = read_arena_rows(args.output)
    write_matchup_matrix_csv(rows, args.matrix_output)
    write_score_matrix_csv(rows, args.score_matrix_output)
    summary = summarize_trap_effectiveness(rows)
    write_summary(summary, args.summary_output)

    print(
        f"target tasks={len(tasks)} completed_now={len(rows_written)} "
        f"skipped_or_existing={len(tasks) - len(rows_written)} workers={args.workers}"
    )
    for row in summary:
        print(
            f"{row['trap_agent']:14s} target={row['target_agent']:10s} "
            f"games={row['games']:>3s} wins={row['wins']:>3s} draws={row['draws']:>3s} "
            f"score={row['score_rate']} traps={row['avg_trap_events']} "
            f"target_delta={row['avg_target_path_delta']}"
        )
    print(f"wrote {args.output}")
    print(f"wrote {args.summary_output}")


if __name__ == "__main__":
    main()
