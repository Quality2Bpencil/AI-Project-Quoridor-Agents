"""Round-robin tournament utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .elo import update_elo
from .metrics import GameRecord, play_game


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    factory: Callable[[], object]


@dataclass(slots=True)
class TournamentResult:
    records: list[GameRecord] = field(default_factory=list)

    def standings(self) -> list[dict[str, float | int | str]]:
        names = sorted({record.agent0 for record in self.records} | {record.agent1 for record in self.records})
        stats = {
            name: {
                "agent": name,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "elo": 1000.0,
                "trap_events": 0,
                "wall_actions": 0,
                "path_delta": 0,
            }
            for name in names
        }

        for record in self.records:
            a = stats[record.agent0]
            b = stats[record.agent1]
            a["games"] += 1
            b["games"] += 1
            a["trap_events"] += record.trap_events[0]
            b["trap_events"] += record.trap_events[1]
            a["wall_actions"] += record.wall_actions[0]
            b["wall_actions"] += record.wall_actions[1]
            a["path_delta"] += record.final_path_lengths[0] - record.initial_path_lengths[0]
            b["path_delta"] += record.final_path_lengths[1] - record.initial_path_lengths[1]

            if record.winner is None:
                a["draws"] += 1
                b["draws"] += 1
                score_a = 0.5
            elif record.winner == 0:
                a["wins"] += 1
                b["losses"] += 1
                score_a = 1.0
            else:
                b["wins"] += 1
                a["losses"] += 1
                score_a = 0.0

            a["elo"], b["elo"] = update_elo(float(a["elo"]), float(b["elo"]), score_a)

        rows = list(stats.values())
        for row in rows:
            games = int(row["games"])
            row["win_rate"] = 0.0 if games == 0 else float(row["wins"]) / games
            row["avg_trap_events"] = 0.0 if games == 0 else round(float(row["trap_events"]) / games, 3)
            row["avg_wall_actions"] = 0.0 if games == 0 else round(float(row["wall_actions"]) / games, 3)
            row["avg_path_delta"] = 0.0 if games == 0 else round(float(row["path_delta"]) / games, 3)
            row["elo"] = round(float(row["elo"]), 1)
        return sorted(rows, key=lambda row: (float(row["elo"]), float(row["win_rate"]), str(row["agent"])), reverse=True)

    def write_games_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "agent0",
                    "agent1",
                    "winner",
                    "winner_name",
                    "turns",
                    "max_turns_reached",
                    "disqualified_player",
                    "remaining_walls_0",
                    "remaining_walls_1",
                    "initial_path_0",
                    "initial_path_1",
                    "final_path_0",
                    "final_path_1",
                    "final_path_delta_0",
                    "final_path_delta_1",
                    "final_diversity_0",
                    "final_diversity_1",
                    "min_diversity_0",
                    "min_diversity_1",
                    "move_actions_0",
                    "move_actions_1",
                    "wall_actions_0",
                    "wall_actions_1",
                    "trap_events_0",
                    "trap_events_1",
                ],
            )
            writer.writeheader()
            for record in self.records:
                writer.writerow(
                    {
                        "agent0": record.agent0,
                        "agent1": record.agent1,
                        "winner": record.winner,
                        "winner_name": record.winner_name,
                        "turns": record.turns,
                        "max_turns_reached": record.max_turns_reached,
                        "disqualified_player": record.disqualified_player,
                        "remaining_walls_0": record.remaining_walls[0],
                        "remaining_walls_1": record.remaining_walls[1],
                        "initial_path_0": record.initial_path_lengths[0],
                        "initial_path_1": record.initial_path_lengths[1],
                        "final_path_0": record.final_path_lengths[0],
                        "final_path_1": record.final_path_lengths[1],
                        "final_path_delta_0": record.final_path_lengths[0] - record.initial_path_lengths[0],
                        "final_path_delta_1": record.final_path_lengths[1] - record.initial_path_lengths[1],
                        "final_diversity_0": record.final_path_diversity[0],
                        "final_diversity_1": record.final_path_diversity[1],
                        "min_diversity_0": record.min_path_diversity[0],
                        "min_diversity_1": record.min_path_diversity[1],
                        "move_actions_0": record.move_actions[0],
                        "move_actions_1": record.move_actions[1],
                        "wall_actions_0": record.wall_actions[0],
                        "wall_actions_1": record.wall_actions[1],
                        "trap_events_0": record.trap_events[0],
                        "trap_events_1": record.trap_events[1],
                    }
                )


def run_round_robin(
    specs: list[AgentSpec],
    *,
    games_per_pair: int = 2,
    max_turns: int = 200,
) -> TournamentResult:
    records: list[GameRecord] = []
    for i, spec_a in enumerate(specs):
        for spec_b in specs[i + 1 :]:
            for game_index in range(games_per_pair):
                if game_index % 2 == 0:
                    records.append(
                        play_game(
                            spec_a.factory(),
                            spec_b.factory(),
                            agent0_name=spec_a.name,
                            agent1_name=spec_b.name,
                            max_turns=max_turns,
                        )
                    )
                else:
                    records.append(
                        play_game(
                            spec_b.factory(),
                            spec_a.factory(),
                            agent0_name=spec_b.name,
                            agent1_name=spec_a.name,
                            max_turns=max_turns,
                        )
                    )
    return TournamentResult(records)
