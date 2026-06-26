"""Shared heuristic helpers for search-based Quoridor agents."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
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
RACE_PROJECTION_WEIGHT = 28.0


def action_sort_key(action: Action) -> tuple[str, int, int, str]:
    if isinstance(action, MoveAction):
        row, col = action.target
        return ("M", row, col, "")
    return ("W", action.row, action.col, action.orientation)


def path_distance(state: QuoridorState, player: int) -> int:
    distance = shortest_path_length(state, player)
    return UNREACHABLE_DISTANCE if distance is None else distance


def evaluate_state(state: QuoridorState, player: int) -> float:
    """Score a state from one player's perspective."""

    return sum(evaluate_state_terms(state, player).values())


def evaluate_state_terms(state: QuoridorState, player: int) -> dict[str, float]:
    """Break a heuristic state evaluation into named, paper-friendly terms."""

    return dict(_evaluate_state_terms_cached(state, player))


@lru_cache(maxsize=100_000)
def _evaluate_state_terms_cached(state: QuoridorState, player: int) -> tuple[tuple[str, float], ...]:
    opponent = 1 - player
    if state.winner == player:
        return (("terminal", WIN_SCORE - state.turn_count),)
    if state.winner == opponent:
        return (("terminal", -WIN_SCORE + state.turn_count),)

    self_dist = path_distance(state, player)
    opp_dist = path_distance(state, opponent)
    wall_balance = state.remaining_walls[player] - state.remaining_walls[opponent]
    self_diversity = path_diversity(state, player)
    opp_diversity = path_diversity(state, opponent)
    self_mobility = pawn_mobility(state, player)
    opp_mobility = pawn_mobility(state, opponent)
    progress = goal_progress(state, player) - goal_progress(state, opponent)
    race_bonus = 0.0
    if _should_project_pawn_race(state, self_dist, opp_dist):
        race_bonus = RACE_PROJECTION_WEIGHT * pawn_race_score(state, player)
    terms = (
        ("path_distance", 10.0 * (opp_dist - self_dist)),
        ("wall_balance", 0.75 * wall_balance),
        ("path_diversity", 1.25 * (self_diversity - opp_diversity)),
        ("pawn_mobility", 0.75 * (self_mobility - opp_mobility)),
        ("goal_progress", 0.5 * progress),
        ("tempo", 0.25 if state.current_player == player else -0.25),
        ("pawn_race", race_bonus),
    )
    return tuple((name, float(value)) for name, value in terms)


def evaluate_action(state: QuoridorState, action: Action, player: int | None = None) -> float:
    player = state.current_player if player is None else player
    try:
        next_state = apply_action(state, action)
    except ValueError:
        return -WIN_SCORE
    value = evaluate_state(next_state, player)
    if isinstance(action, WallAction):
        value += wall_action_bonus(state, next_state, action, player)
    return value


def ranked_actions(
    state: QuoridorState,
    *,
    player: int | None = None,
    max_actions: int | None = None,
    wall_limit: int | None = None,
    wall_radius: int = 2,
    wall_penalty: float = 0.0,
) -> list[Action]:
    """Return legal actions ranked by a one-ply heuristic.

    Wall actions dominate the raw branching factor, so callers can keep all pawn
    moves while only taking the highest-scoring wall candidates. A small
    wall_penalty is useful for low-budget searches that otherwise over-spend
    walls because a one-ply path delta looks immediately attractive.
    """

    if state.done:
        return []

    player = state.current_player if player is None else player
    scores: dict[Action, float] = {}

    def score(action: Action) -> float:
        value = scores.get(action)
        if value is None:
            value = evaluate_action(state, action, player)
            if isinstance(action, WallAction):
                value -= wall_penalty
            scores[action] = value
        return value

    pawn_actions: list[Action] = list(legal_pawn_moves(state, player))
    wall_actions: list[Action] = list(plausible_wall_actions(state, player, radius=wall_radius))

    if wall_limit is not None:
        wall_actions = sorted(
            wall_actions,
            key=lambda action: (score(action), action_sort_key(action)),
            reverse=True,
        )[:wall_limit]

    actions = pawn_actions + wall_actions
    actions = sorted(
        actions,
        key=lambda action: (score(action), action_sort_key(action)),
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


def pawn_mobility(state: QuoridorState, player: int) -> int:
    return len(legal_pawn_moves(state, player))


def goal_progress(state: QuoridorState, player: int) -> int:
    row, _ = state.pawn_positions[player]
    if player == 0:
        return state.board_size - 1 - row
    return row


def wall_action_bonus(
    state: QuoridorState,
    next_state: QuoridorState,
    action: WallAction,
    player: int,
) -> float:
    opponent = 1 - player
    before_self = path_distance(state, player)
    before_opp = path_distance(state, opponent)
    after_self = path_distance(next_state, player)
    after_opp = path_distance(next_state, opponent)
    before_opp_diversity = path_diversity(state, opponent)
    after_opp_diversity = path_diversity(next_state, opponent)
    before_opp_mobility = pawn_mobility(state, opponent)
    after_opp_mobility = pawn_mobility(next_state, opponent)

    opp_slowdown = max(0, after_opp - before_opp)
    self_slowdown = max(0, after_self - before_self)
    path_bonus = 5.0 * opp_slowdown - 4.0 * self_slowdown
    diversity_drop = max(0, before_opp_diversity - after_opp_diversity)
    mobility_drop = max(0, before_opp_mobility - after_opp_mobility)
    constraint_bonus = 2.5 * diversity_drop + 1.5 * mobility_drop

    opp_row, opp_col = state.pawn_positions[opponent]
    front_row = opp_row if opponent == 1 else opp_row - 1
    front_row = max(0, min(state.board_size - 2, front_row))
    front_distance = abs(action.row - front_row) + abs(action.col - opp_col)
    front_bonus = max(0.0, 4.0 - front_distance) * 1.5

    center = state.board_size // 2
    center_bonus = -0.2 * abs(action.col - center)
    orientation_bonus = 2.5 if action.orientation == "H" else 0.0
    empty_tempo_penalty = 0.0
    if opp_slowdown == 0 and diversity_drop == 0 and mobility_drop == 0:
        empty_tempo_penalty = -1.0
    return path_bonus + constraint_bonus + front_bonus + center_bonus + orientation_bonus + empty_tempo_penalty


@lru_cache(maxsize=100_000)
def pawn_race_winner(state: QuoridorState, max_steps: int = 40) -> int | None:
    """Project a no-more-walls pawn race under shortest-path movement.

    Plain shortest-path distance misses the central Quoridor jump tactic: from
    the opening, two players that both run straight do not actually draw; the
    second player can jump and win the race. This cheap projection gives the
    one-ply and low-budget search agents enough tactical awareness to block
    obvious pawn races instead of blindly trusting static path lengths.
    """

    current = state
    step_limit = min(max_steps, max(8, path_distance(state, 0) + path_distance(state, 1) + 4))
    for _ in range(step_limit):
        if current.done:
            return current.winner

        player = current.current_player
        moves = legal_pawn_moves(current, player)
        if not moves:
            return None

        def move_key(action: MoveAction) -> tuple[int, tuple[str, int, int, str]]:
            next_state = apply_action(current, action)
            distance = shortest_path_length(next_state, player)
            return (UNREACHABLE_DISTANCE if distance is None else distance, action_sort_key(action))

        current = apply_action(current, min(moves, key=move_key))
    return None


def pawn_race_score(state: QuoridorState, player: int) -> float:
    winner = pawn_race_winner(state)
    if winner == player:
        return 1.0
    if winner == 1 - player:
        return -1.0
    return 0.0


def _should_project_pawn_race(state: QuoridorState, self_dist: int, opp_dist: int) -> bool:
    if state.turn_count > 1:
        return False
    if abs(self_dist - opp_dist) > 2:
        return False
    if self_dist >= UNREACHABLE_DISTANCE or opp_dist >= UNREACHABLE_DISTANCE:
        return False
    return True
