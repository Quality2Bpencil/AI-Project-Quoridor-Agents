"""Evaluation helpers for Quoridor agents."""

from .arena import (
    ArenaTask,
    arena_rows_to_result,
    build_round_robin_tasks,
    matchup_matrix_rows,
    play_arena_task,
    read_arena_rows,
    run_arena,
    write_matchup_matrix_csv,
    write_score_matrix_csv,
)
from .elo import expected_score, update_elo
from .metrics import GameRecord, play_game
from .tournament import AgentSpec, TournamentResult, run_round_robin

__all__ = [
    "AgentSpec",
    "ArenaTask",
    "GameRecord",
    "TournamentResult",
    "arena_rows_to_result",
    "build_round_robin_tasks",
    "expected_score",
    "matchup_matrix_rows",
    "play_arena_task",
    "play_game",
    "read_arena_rows",
    "run_arena",
    "run_round_robin",
    "update_elo",
    "write_matchup_matrix_csv",
    "write_score_matrix_csv",
]
