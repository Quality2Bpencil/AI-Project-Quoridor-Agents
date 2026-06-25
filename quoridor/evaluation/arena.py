"""Parallel, resumable arena runner for formal Quoridor tournaments."""

from __future__ import annotations

import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .metrics import GameRecord, play_game
from .tournament import AgentSpec, TournamentResult


@dataclass(frozen=True, slots=True)
class ArenaTask:
    game_id: str
    pair_id: str
    pair_index: int
    game_index: int
    agent_a: str
    agent_b: str
    agent0: str
    agent1: str
    max_turns: int
    seed: int = 0


ARENA_FIELDNAMES = [
    "game_id",
    "pair_id",
    "pair_index",
    "game_index",
    "agent_a",
    "agent_b",
    "agent0",
    "agent1",
    "seed",
    "status",
    "error",
    "elapsed_seconds",
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
]

MATCHUP_FIELDNAMES = [
    "agent",
    "opponent",
    "games",
    "wins",
    "losses",
    "draws",
    "score_rate",
    "win_rate",
    "avg_turns",
    "avg_trap_events",
    "avg_wall_actions",
    "avg_path_delta",
    "avg_elapsed_seconds",
]


def build_round_robin_tasks(
    specs: Sequence[AgentSpec],
    *,
    games_per_pair: int,
    max_turns: int,
) -> list[ArenaTask]:
    """Build deterministic per-game tasks with alternating seat assignment."""

    tasks: list[ArenaTask] = []
    pair_index = 0
    for i, spec_a in enumerate(specs):
        for spec_b in specs[i + 1 :]:
            pair_id = f"{spec_a.name}__vs__{spec_b.name}"
            for game_index in range(games_per_pair):
                agent0, agent1 = (
                    (spec_a.name, spec_b.name)
                    if game_index % 2 == 0
                    else (spec_b.name, spec_a.name)
                )
                tasks.append(
                    ArenaTask(
                        game_id=f"{pair_index:04d}-{game_index:04d}",
                        pair_id=pair_id,
                        pair_index=pair_index,
                        game_index=game_index,
                        agent_a=spec_a.name,
                        agent_b=spec_b.name,
                        agent0=agent0,
                        agent1=agent1,
                        max_turns=max_turns,
                        seed=pair_index * 100_000 + game_index,
                    )
                )
            pair_index += 1
    return tasks


def play_arena_task(task: ArenaTask, specs_by_name: Mapping[str, AgentSpec]) -> dict[str, object]:
    """Play one arena task and return a CSV-ready row."""

    start = time.perf_counter()
    try:
        spec0 = specs_by_name[task.agent0]
        spec1 = specs_by_name[task.agent1]
    except KeyError as exc:
        return _error_row(task, f"missing agent spec: {exc!r}", elapsed_seconds=time.perf_counter() - start)

    try:
        agent0 = spec0.factory()
    except Exception as exc:
        record = _disqualification_record(task, disqualified_player=0)
        return record_to_arena_row(
            task,
            record,
            status="factory_error",
            error=repr(exc),
            elapsed_seconds=time.perf_counter() - start,
        )

    try:
        agent1 = spec1.factory()
    except Exception as exc:
        record = _disqualification_record(task, disqualified_player=1)
        return record_to_arena_row(
            task,
            record,
            status="factory_error",
            error=repr(exc),
            elapsed_seconds=time.perf_counter() - start,
        )

    try:
        record = play_game(
            agent0,
            agent1,
            agent0_name=task.agent0,
            agent1_name=task.agent1,
            max_turns=task.max_turns,
        )
        return record_to_arena_row(task, record, elapsed_seconds=time.perf_counter() - start)
    except Exception as exc:
        return _error_row(task, repr(exc), elapsed_seconds=time.perf_counter() - start)


def run_arena(
    tasks: Sequence[ArenaTask],
    play_task: Callable[[ArenaTask], dict[str, object]],
    *,
    output: str | Path,
    workers: int = 1,
    resume: bool = False,
    progress_interval: int = 0,
) -> list[dict[str, str]]:
    """Run tasks, appending rows as games finish.

    When resume is enabled, game_ids already present in output are skipped.
    Writes are performed only by the parent process, so the output file remains
    append-safe even with process workers.
    """

    output = Path(output)
    completed_ids = _completed_game_ids(output) if resume else set()
    pending = [task for task in tasks if task.game_id not in completed_ids]
    output.parent.mkdir(parents=True, exist_ok=True)
    run_start = time.perf_counter()

    needs_header = not output.exists() or output.stat().st_size == 0
    rows_written: list[dict[str, str]] = []

    def report_progress() -> None:
        if progress_interval <= 0:
            return
        done = len(rows_written)
        if done == 0 or done % progress_interval != 0:
            return
        elapsed = max(1e-9, time.perf_counter() - run_start)
        rate = done / elapsed
        remaining = max(0, len(pending) - done)
        eta = remaining / rate if rate > 0 else 0.0
        print(
            f"arena progress {done}/{len(pending)} "
            f"rate={rate:.2f} games/s eta={eta:.1f}s"
        )

    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ARENA_FIELDNAMES, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
            handle.flush()

        if workers <= 1:
            for task in pending:
                row = _stringify_row(play_task(task))
                writer.writerow(row)
                handle.flush()
                rows_written.append(row)
                report_progress()
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                future_to_task = {pool.submit(play_task, task): task for task in pending}
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        row = _stringify_row(future.result())
                    except Exception as exc:
                        row = _stringify_row(_error_row(task, repr(exc)))
                    writer.writerow(row)
                    handle.flush()
                    rows_written.append(row)
                    report_progress()

    return rows_written


def read_arena_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def arena_rows_to_result(rows: Iterable[Mapping[str, str]]) -> TournamentResult:
    records = [arena_row_to_record(row) for row in rows if row.get("status", "ok") in {"ok", "factory_error"}]
    return TournamentResult(records)


def write_matchup_matrix_csv(rows: Iterable[Mapping[str, str]], path: str | Path) -> None:
    matrix_rows = matchup_matrix_rows(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCHUP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(matrix_rows)


def write_score_matrix_csv(rows: Iterable[Mapping[str, str]], path: str | Path) -> None:
    rows = list(rows)
    agents = sorted({row["agent0"] for row in rows} | {row["agent1"] for row in rows})
    lookup = {(row["agent"], row["opponent"]): row for row in matchup_matrix_rows(rows)}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["agent", *agents])
        writer.writeheader()
        for agent in agents:
            output_row = {"agent": agent}
            for opponent in agents:
                if agent == opponent:
                    output_row[opponent] = ""
                else:
                    matchup = lookup.get((agent, opponent))
                    output_row[opponent] = "" if matchup is None else matchup["score_rate"]
            writer.writerow(output_row)


def matchup_matrix_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    stats: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        if row.get("status", "ok") not in {"ok", "factory_error"}:
            continue
        agent0 = row["agent0"]
        agent1 = row["agent1"]
        winner = _optional_int(row.get("winner", ""))
        turns = _int(row.get("turns", "0"))

        _update_agent_matchup(stats, row, agent0, agent1, seat=0, winner=winner, turns=turns)
        _update_agent_matchup(stats, row, agent1, agent0, seat=1, winner=winner, turns=turns)

    output: list[dict[str, str]] = []
    for (agent, opponent), values in sorted(stats.items()):
        games = int(values["games"])
        wins = int(values["wins"])
        draws = int(values["draws"])
        losses = games - wins - draws
        score = wins + 0.5 * draws
        output.append(
            {
                "agent": agent,
                "opponent": opponent,
                "games": str(games),
                "wins": str(wins),
                "losses": str(losses),
                "draws": str(draws),
                "score_rate": _format_rate(score / games if games else 0.0),
                "win_rate": _format_rate(wins / games if games else 0.0),
                "avg_turns": _format_float(values["turns"] / games if games else 0.0),
                "avg_trap_events": _format_float(values["trap_events"] / games if games else 0.0),
                "avg_wall_actions": _format_float(values["wall_actions"] / games if games else 0.0),
                "avg_path_delta": _format_float(values["path_delta"] / games if games else 0.0),
                "avg_elapsed_seconds": _format_float(values["elapsed_seconds"] / games if games else 0.0),
            }
        )
    return output


def record_to_arena_row(
    task: ArenaTask,
    record: GameRecord,
    *,
    status: str = "ok",
    error: str = "",
    elapsed_seconds: float = 0.0,
) -> dict[str, object]:
    return {
        "game_id": task.game_id,
        "pair_id": task.pair_id,
        "pair_index": task.pair_index,
        "game_index": task.game_index,
        "agent_a": task.agent_a,
        "agent_b": task.agent_b,
        "agent0": record.agent0,
        "agent1": record.agent1,
        "seed": task.seed,
        "status": status,
        "error": error,
        "elapsed_seconds": f"{elapsed_seconds:.6f}",
        "winner": "" if record.winner is None else record.winner,
        "winner_name": record.winner_name,
        "turns": record.turns,
        "max_turns_reached": record.max_turns_reached,
        "disqualified_player": "" if record.disqualified_player is None else record.disqualified_player,
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


def arena_row_to_record(row: Mapping[str, str]) -> GameRecord:
    return GameRecord(
        agent0=row["agent0"],
        agent1=row["agent1"],
        winner=_optional_int(row.get("winner", "")),
        turns=_int(row["turns"]),
        max_turns_reached=_bool(row["max_turns_reached"]),
        disqualified_player=_optional_int(row.get("disqualified_player", "")),
        remaining_walls=(_int(row["remaining_walls_0"]), _int(row["remaining_walls_1"])),
        initial_path_lengths=(_int(row["initial_path_0"]), _int(row["initial_path_1"])),
        final_path_lengths=(_int(row["final_path_0"]), _int(row["final_path_1"])),
        final_path_diversity=(_int(row["final_diversity_0"]), _int(row["final_diversity_1"])),
        min_path_diversity=(_int(row["min_diversity_0"]), _int(row["min_diversity_1"])),
        move_actions=(_int(row["move_actions_0"]), _int(row["move_actions_1"])),
        wall_actions=(_int(row["wall_actions_0"]), _int(row["wall_actions_1"])),
        trap_events=(_int(row["trap_events_0"]), _int(row["trap_events_1"])),
    )


def _update_agent_matchup(
    stats: dict[tuple[str, str], dict[str, float]],
    row: Mapping[str, str],
    agent: str,
    opponent: str,
    *,
    seat: int,
    winner: int | None,
    turns: int,
) -> None:
    values = stats.setdefault(
        (agent, opponent),
        {
            "games": 0.0,
            "wins": 0.0,
            "draws": 0.0,
            "turns": 0.0,
            "trap_events": 0.0,
            "wall_actions": 0.0,
            "path_delta": 0.0,
            "elapsed_seconds": 0.0,
        },
    )
    values["games"] += 1.0
    values["turns"] += turns
    values["trap_events"] += _int(row.get(f"trap_events_{seat}", "0"))
    values["wall_actions"] += _int(row.get(f"wall_actions_{seat}", "0"))
    values["path_delta"] += _int(row.get(f"final_path_delta_{seat}", "0"))
    values["elapsed_seconds"] += _float(row.get("elapsed_seconds", "0")) / 2.0
    if winner is None:
        values["draws"] += 1.0
    elif winner == seat:
        values["wins"] += 1.0


def _completed_game_ids(path: Path) -> set[str]:
    return {row["game_id"] for row in read_arena_rows(path) if row.get("game_id")}


def _disqualification_record(task: ArenaTask, disqualified_player: int) -> GameRecord:
    winner = 1 - disqualified_player
    return GameRecord(
        agent0=task.agent0,
        agent1=task.agent1,
        winner=winner,
        turns=0,
        max_turns_reached=False,
        disqualified_player=disqualified_player,
    )


def _error_row(task: ArenaTask, error: str, *, elapsed_seconds: float = 0.0) -> dict[str, object]:
    record = GameRecord(
        agent0=task.agent0,
        agent1=task.agent1,
        winner=None,
        turns=0,
        max_turns_reached=False,
    )
    return record_to_arena_row(task, record, status="error", error=error, elapsed_seconds=elapsed_seconds)


def _stringify_row(row: Mapping[str, object]) -> dict[str, str]:
    return {field: str(row.get(field, "")) for field in ARENA_FIELDNAMES}


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def _float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() == "true"


def _format_rate(value: float) -> str:
    return f"{value:.4f}"


def _format_float(value: float) -> str:
    return f"{value:.3f}"
