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
            name: {"agent": name, "games": 0, "wins": 0, "losses": 0, "draws": 0, "elo": 1000.0}
            for name in names
        }

        for record in self.records:
            a = stats[record.agent0]
            b = stats[record.agent1]
            a["games"] += 1
            b["games"] += 1

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
