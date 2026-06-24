"""Run focused adversarial ablations for report tables."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quoridor.agents import GreedyBFSAgent, PathLureAgent
from quoridor.evaluation import GameRecord, play_game


def parse_weights(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def run_path_lure_ablation(
    *,
    weights: list[float],
    games: int,
    max_turns: int,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | float | str | None | bool]] = []

    for weight in weights:
        condition = f"path_lure_weight_{weight:g}"
        for game_index in range(games):
            adversary_player = game_index % 2
            seed = game_index
            adversary = PathLureAgent(
                seed=seed,
                trap_weight=weight,
                action_limit=8,
                wall_limit=4,
                victim_action_limit=8,
            )
            victim = GreedyBFSAgent(seed=seed + 10, action_limit=16, wall_limit=8)

            if adversary_player == 0:
                record = play_game(
                    adversary,
                    victim,
                    agent0_name=condition,
                    agent1_name="greedy_bfs",
                    max_turns=max_turns,
                )
            else:
                record = play_game(
                    victim,
                    adversary,
                    agent0_name="greedy_bfs",
                    agent1_name=condition,
                    max_turns=max_turns,
                )

            rows.append(_record_to_ablation_row(condition, game_index, adversary_player, record))

    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in summarize_rows(rows):
        print(
            f"{row['condition']:22s} games={row['games']:3d} "
            f"win_rate={row['adversary_win_rate']:.2f} "
            f"trap={row['avg_trap_events']:.2f} "
            f"opp_min_div={row['avg_opponent_min_diversity']:.2f} "
            f"opp_path_delta={row['avg_opponent_path_delta']:.2f}"
        )
    print(f"wrote {output}")


def _record_to_ablation_row(
    condition: str,
    game_index: int,
    adversary_player: int,
    record: GameRecord,
) -> dict[str, int | float | str | None | bool]:
    opponent = 1 - adversary_player
    return {
        "condition": condition,
        "game_index": game_index,
        "adversary_player": adversary_player,
        "winner": record.winner,
        "adversary_won": record.winner == adversary_player,
        "draw": record.winner is None,
        "turns": record.turns,
        "trap_events": record.trap_events[adversary_player],
        "opponent_min_diversity": record.min_path_diversity[opponent],
        "opponent_final_diversity": record.final_path_diversity[opponent],
        "opponent_path_delta": record.final_path_lengths[opponent] - record.initial_path_lengths[opponent],
        "adversary_wall_actions": record.wall_actions[adversary_player],
        "opponent_wall_actions": record.wall_actions[opponent],
        "max_turns_reached": record.max_turns_reached,
        "disqualified_player": record.disqualified_player,
    }


def summarize_rows(rows: list[dict[str, int | float | str | None | bool]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, int | float | str | None | bool]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)

    summaries: list[dict[str, float | int | str]] = []
    for condition, condition_rows in sorted(grouped.items()):
        games = len(condition_rows)
        wins = sum(1 for row in condition_rows if row["adversary_won"])
        summaries.append(
            {
                "condition": condition,
                "games": games,
                "adversary_win_rate": wins / games if games else 0.0,
                "avg_trap_events": _average(condition_rows, "trap_events"),
                "avg_opponent_min_diversity": _average(condition_rows, "opponent_min_diversity"),
                "avg_opponent_path_delta": _average(condition_rows, "opponent_path_delta"),
            }
        )
    return summaries


def _average(rows: list[dict[str, int | float | str | None | bool]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="0,4,8")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/path_lure_ablation.csv"))
    args = parser.parse_args()

    run_path_lure_ablation(
        weights=parse_weights(args.weights),
        games=args.games,
        max_turns=args.max_turns,
        output=args.output,
    )


if __name__ == "__main__":
    main()
