"""Rule implementation for Quoridor."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from functools import lru_cache
from typing import Iterable

from .actions import Action, MoveAction, Orientation, Position, WallAction
from .state import QuoridorState, Wall

Direction = tuple[int, int]

ORTHOGONAL_DIRECTIONS: tuple[Direction, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def in_bounds(state: QuoridorState, pos: Position) -> bool:
    row, col = pos
    return 0 <= row < state.board_size and 0 <= col < state.board_size


def wall_in_bounds(state: QuoridorState, action: WallAction) -> bool:
    return 0 <= action.row < state.board_size - 1 and 0 <= action.col < state.board_size - 1


def is_blocked(state: QuoridorState, a: Position, b: Position) -> bool:
    """Return True when a wall blocks movement between adjacent squares."""

    ar, ac = a
    br, bc = b
    dr, dc = br - ar, bc - ac
    if abs(dr) + abs(dc) != 1:
        raise ValueError("is_blocked expects adjacent squares")

    walls = state.walls
    if dr == 1:
        return ("H", ar, ac) in walls or ("H", ar, ac - 1) in walls
    if dr == -1:
        return ("H", br, bc) in walls or ("H", br, bc - 1) in walls
    if dc == 1:
        return ("V", ar, ac) in walls or ("V", ar - 1, ac) in walls
    return ("V", br, bc) in walls or ("V", br - 1, bc) in walls


def adjacent_reachable(state: QuoridorState, pos: Position) -> Iterable[Position]:
    for dr, dc in ORTHOGONAL_DIRECTIONS:
        nxt = (pos[0] + dr, pos[1] + dc)
        if in_bounds(state, nxt) and not is_blocked(state, pos, nxt):
            yield nxt


def legal_pawn_moves(state: QuoridorState, player: int | None = None) -> list[MoveAction]:
    """Generate legal pawn moves, including jumps and diagonal side-steps."""

    if state.done:
        return []

    player = state.current_player if player is None else player
    opponent = 1 - player
    current = state.pawn_positions[player]
    other = state.pawn_positions[opponent]
    moves: set[Position] = set()

    for nxt in adjacent_reachable(state, current):
        if nxt != other:
            moves.add(nxt)
            continue

        jump_dr = other[0] - current[0]
        jump_dc = other[1] - current[1]
        behind = (other[0] + jump_dr, other[1] + jump_dc)
        if in_bounds(state, behind) and not is_blocked(state, other, behind):
            moves.add(behind)
            continue

        if jump_dr != 0:
            side_dirs = ((0, -1), (0, 1))
        else:
            side_dirs = ((-1, 0), (1, 0))
        for side_dr, side_dc in side_dirs:
            side = (other[0] + side_dr, other[1] + side_dc)
            if in_bounds(state, side) and not is_blocked(state, other, side):
                moves.add(side)

    return [MoveAction(target=target) for target in sorted(moves)]


def wall_conflicts(state: QuoridorState, action: WallAction) -> bool:
    """Return True when a wall overlaps or crosses an existing wall."""

    wall: Wall = (action.orientation, action.row, action.col)
    if wall in state.walls:
        return True

    if action.orientation == "H":
        return (
            ("H", action.row, action.col - 1) in state.walls
            or ("H", action.row, action.col + 1) in state.walls
            or ("V", action.row, action.col) in state.walls
        )
    return (
        ("V", action.row - 1, action.col) in state.walls
        or ("V", action.row + 1, action.col) in state.walls
        or ("H", action.row, action.col) in state.walls
    )


@lru_cache(maxsize=100_000)
def has_path_to_goal(state: QuoridorState, player: int) -> bool:
    """Check whether a player still has at least one path to their goal row."""

    start = state.pawn_positions[player]
    goal = state.goal_row(player)
    visited = {start}
    queue: deque[Position] = deque([start])

    while queue:
        pos = queue.popleft()
        if pos[0] == goal:
            return True
        for nxt in adjacent_reachable(state, pos):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return False


@lru_cache(maxsize=100_000)
def shortest_path_length(state: QuoridorState, player: int) -> int | None:
    """Return the shortest wall-respecting path length to the goal row."""

    start = state.pawn_positions[player]
    goal = state.goal_row(player)
    visited = {start}
    queue: deque[tuple[Position, int]] = deque([(start, 0)])

    while queue:
        pos, dist = queue.popleft()
        if pos[0] == goal:
            return dist
        for nxt in adjacent_reachable(state, pos):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def legal_wall_action(state: QuoridorState, action: WallAction, player: int | None = None) -> bool:
    if state.done:
        return False

    player = state.current_player if player is None else player
    if state.remaining_walls[player] <= 0:
        return False
    if not wall_in_bounds(state, action):
        return False
    if wall_conflicts(state, action):
        return False

    candidate = replace(state, walls=state.walls | {(action.orientation, action.row, action.col)})
    return has_path_to_goal(candidate, 0) and has_path_to_goal(candidate, 1)


def legal_wall_actions(state: QuoridorState, player: int | None = None) -> list[WallAction]:
    if state.done:
        return []

    player = state.current_player if player is None else player
    if state.remaining_walls[player] <= 0:
        return []

    actions: list[WallAction] = []
    for orientation in ("H", "V"):
        for row in range(state.board_size - 1):
            for col in range(state.board_size - 1):
                action = WallAction(orientation, row, col)
                if legal_wall_action(state, action, player):
                    actions.append(action)
    return actions


def legal_actions(state: QuoridorState, player: int | None = None) -> list[Action]:
    return [*legal_pawn_moves(state, player), *legal_wall_actions(state, player)]


def is_legal_action(state: QuoridorState, action: Action, player: int | None = None) -> bool:
    """Return whether a single action is legal without enumerating all walls."""

    if isinstance(action, MoveAction):
        return action in legal_pawn_moves(state, player)
    if isinstance(action, WallAction):
        return legal_wall_action(state, action, player)
    return False


def apply_action(state: QuoridorState, action: Action) -> QuoridorState:
    """Apply a legal action and return the next immutable state."""

    if not is_legal_action(state, action):
        raise ValueError(f"illegal action for player {state.current_player}: {action}")

    player = state.current_player
    next_player = 1 - player
    pawn_positions = list(state.pawn_positions)
    remaining_walls = list(state.remaining_walls)
    walls = state.walls
    winner = None

    if isinstance(action, MoveAction):
        pawn_positions[player] = action.target
        if action.target[0] == state.goal_row(player):
            winner = player
    else:
        walls = walls | {(action.orientation, action.row, action.col)}
        remaining_walls[player] -= 1

    return QuoridorState(
        board_size=state.board_size,
        pawn_positions=(pawn_positions[0], pawn_positions[1]),
        walls=walls,
        remaining_walls=(remaining_walls[0], remaining_walls[1]),
        current_player=next_player,
        winner=winner,
        turn_count=state.turn_count + 1,
    )
