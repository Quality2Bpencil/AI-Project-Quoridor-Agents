"""Shared heuristic helpers for search-based Quoridor agents."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from quoridor.core.actions import Action, MoveAction, Position, WallAction
from quoridor.core.rules import (
    adjacent_reachable,
    apply_action,
    legal_pawn_moves,
    legal_wall_action,
    shortest_path_length,
)
from quoridor.core.state import QuoridorState

WIN_SCORE = 100_000.0
UNREACHABLE_DISTANCE = 1_000


def action_sort_key(action: Action) -> tuple[str, int, int, str]:
    if isinstance(action, MoveAction):
        row, col = action.target
        return ("M", row, col, "")
    return ("W", action.row, action.col, action.orientation)


def path_distance(state: QuoridorState, player: int) -> int:
    return shortest_path_length(state, player) or UNREACHABLE_DISTANCE


def evaluate_state(state: QuoridorState, player: int) -> float:
    """Score a state from one player's perspective."""

    opponent = 1 - player
    if state.winner == player:
        return WIN_SCORE - state.turn_count
    if state.winner == opponent:
        return -WIN_SCORE + state.turn_count

    self_dist = path_distance(state, player)
    opp_dist = path_distance(state, opponent)
    wall_balance = state.remaining_walls[player] - state.remaining_walls[opponent]
    turn_bonus = 0.25 if state.current_player == player else -0.25
    return 10.0 * (opp_dist - self_dist) + 0.75 * wall_balance + turn_bonus


def evaluate_action(state: QuoridorState, action: Action, player: int | None = None) -> float:
    player = state.current_player if player is None else player
    try:
        next_state = apply_action(state, action)
    except ValueError:
        return -WIN_SCORE
    return evaluate_state(next_state, player)


def ranked_actions(
    state: QuoridorState,
    *,
    player: int | None = None,
    max_actions: int | None = None,
    wall_limit: int | None = None,
    wall_radius: int = 2,
) -> list[Action]:
    """Return legal actions ranked by a one-ply heuristic.

    Wall actions dominate the raw branching factor, so callers can keep all pawn
    moves while only taking the highest-scoring wall candidates.
    """

    if state.done:
        return []

    player = state.current_player if player is None else player
    pawn_actions: list[Action] = list(legal_pawn_moves(state, player))
    wall_actions: list[Action] = list(plausible_wall_actions(state, player, radius=wall_radius))

    if wall_limit is not None:
        wall_actions = sorted(
            wall_actions,
            key=lambda action: (evaluate_action(state, action, player), action_sort_key(action)),
            reverse=True,
        )[:wall_limit]

    actions = pawn_actions + wall_actions
    actions = sorted(
        actions,
        key=lambda action: (evaluate_action(state, action, player), action_sort_key(action)),
        reverse=True,
    )
    if max_actions is not None:
        return actions[:max_actions]
    return actions


def plausible_wall_actions(state: QuoridorState, player: int, radius: int = 2) -> list[WallAction]:
    """Generate legal wall actions near the current tactical area."""

    if state.done or state.remaining_walls[player] <= 0:
        return []

    max_anchor = state.board_size - 2
    anchors: set[tuple[int, int]] = set()
    for pawn_row, pawn_col in state.pawn_positions:
        for row in range(max(0, pawn_row - radius), min(max_anchor, pawn_row + radius) + 1):
            for col in range(max(0, pawn_col - radius), min(max_anchor, pawn_col + radius) + 1):
                anchors.add((row, col))

    actions: list[WallAction] = []
    for row, col in sorted(anchors):
        for orientation in ("H", "V"):
            action = WallAction(orientation, row, col)
            if legal_wall_action(state, action, player):
                actions.append(action)
    return actions


def first_step_options_on_shortest_paths(state: QuoridorState, player: int) -> set[Position]:
    """Approximate path diversity by counting shortest-path first steps."""

    start = state.pawn_positions[player]
    best = shortest_path_length(state, player)
    if best is None or best <= 0:
        return set()

    options: set[Position] = set()
    for nxt in adjacent_reachable(state, start):
        pawn_positions = list(state.pawn_positions)
        pawn_positions[player] = nxt
        candidate = replace(state, pawn_positions=(pawn_positions[0], pawn_positions[1]))
        if shortest_path_length(candidate, player) == best - 1:
            options.add(nxt)
    return options


def path_diversity(state: QuoridorState, player: int) -> int:
    return len(first_step_options_on_shortest_paths(state, player))


def choose_best(actions: Iterable[Action], state: QuoridorState, player: int) -> Action:
    return max(actions, key=lambda action: (evaluate_action(state, action, player), action_sort_key(action)))
