"""Summarize trap-agent effectiveness from arena game CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Mapping

DEFAULT_TARGETS = {
    "path_lure": "greedy_bfs",
    "depth_trap": "minimax_d1",
    "rollout_poison": "mcts_5",
    "counter_trap": "mcts_5",
    "argmax_q_trap": "q_learning",
}

FIELDNAMES = [
    "trap_agent",
    "target_agent",
    "games",
    "wins",
    "losses",
    "draws",
    "score_rate",
    "score_ci95_low",
    "score_ci95_high",
    "win_rate",
    "win_ci95_low",
    "win_ci95_high",
    "avg_turns",
    "avg_trap_events",
    "avg_trap_wall_actions",
    "avg_target_path_delta",
    "avg_target_min_diversity",
]


def summarize_trap_effectiveness(
    rows: Iterable[Mapping[str, str]],
    targets: Mapping[str, str] = DEFAULT_TARGETS,
) -> list[dict[str, str]]:
    rows = list(rows)
    summary: list[dict[str, str]] = []
    for trap_agent, target_agent in targets.items():
        values = {
            "games": 0.0,
            "wins": 0.0,
            "draws": 0.0,
            "turns": 0.0,
            "trap_events": 0.0,
            "trap_wall_actions": 0.0,
            "target_path_delta": 0.0,
            "target_min_diversity": 0.0,
        }
        for row in rows:
            if row.get("status", "ok") not in {"ok", "factory_error"}:
                continue
            agents = {row["agent0"], row["agent1"]}
            if agents != {trap_agent, target_agent}:
                continue

            trap_seat = 0 if row["agent0"] == trap_agent else 1
            target_seat = 1 - trap_seat
            winner = _optional_int(row.get("winner", ""))
            values["games"] += 1.0
            values["turns"] += _int(row.get("turns", "0"))
            values["trap_events"] += _int(row.get(f"trap_events_{trap_seat}", "0"))
            values["trap_wall_actions"] += _int(row.get(f"wall_actions_{trap_seat}", "0"))
            values["target_path_delta"] += _int(row.get(f"final_path_delta_{target_seat}", "0"))
            values["target_min_diversity"] += _int(row.get(f"min_diversity_{target_seat}", "0"))
            if winner is None:
                values["draws"] += 1.0
            elif winner == trap_seat:
                values["wins"] += 1.0

        games = int(values["games"])
        wins = int(values["wins"])
        draws = int(values["draws"])
        losses = games - wins - draws
        score = wins + 0.5 * draws
        score_low, score_high = _wilson_interval(score, games)
        win_low, win_high = _wilson_interval(wins, games)
        summary.append(
            {
                "trap_agent": trap_agent,
                "target_agent": target_agent,
                "games": str(games),
                "wins": str(wins),
                "losses": str(losses),
                "draws": str(draws),
                "score_rate": _format_rate(score / games if games else 0.0),
                "score_ci95_low": _format_rate(score_low),
                "score_ci95_high": _format_rate(score_high),
                "win_rate": _format_rate(wins / games if games else 0.0),
                "win_ci95_low": _format_rate(win_low),
                "win_ci95_high": _format_rate(win_high),
                "avg_turns": _format_float(values["turns"] / games if games else 0.0),
                "avg_trap_events": _format_float(values["trap_events"] / games if games else 0.0),
                "avg_trap_wall_actions": _format_float(values["trap_wall_actions"] / games if games else 0.0),
                "avg_target_path_delta": _format_float(values["target_path_delta"] / games if games else 0.0),
                "avg_target_min_diversity": _format_float(values["target_min_diversity"] / games if games else 0.0),
            }
        )
    return summary


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_summary(rows: Iterable[Mapping[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize_trap_effectiveness(read_rows(args.games))
    write_summary(summary, args.output)
    for row in summary:
        print(
            f"{row['trap_agent']:14s} target={row['target_agent']:10s} "
            f"games={row['games']:>3s} wins={row['wins']:>3s} draws={row['draws']:>3s} "
            f"score={row['score_rate']} traps={row['avg_trap_events']} "
            f"target_delta={row['avg_target_path_delta']}"
        )
    print(f"wrote {args.output}")


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def _format_rate(value: float) -> str:
    return f"{value:.4f}"


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1.0 + z * z / total
    center = (p_hat + z * z / (2.0 * total)) / denom
    margin = z * ((p_hat * (1.0 - p_hat) / total + z * z / (4.0 * total * total)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


if __name__ == "__main__":
    main()
