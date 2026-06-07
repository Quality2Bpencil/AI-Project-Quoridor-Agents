"""Immutable game state for Quoridor."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import Orientation, Position

Wall = tuple[Orientation, int, int]


@dataclass(frozen=True, slots=True)
class QuoridorState:
    """Complete public state of a Quoridor game."""

    board_size: int = 9
    pawn_positions: tuple[Position, Position] = ((8, 4), (0, 4))
    walls: frozenset[Wall] = frozenset()
    remaining_walls: tuple[int, int] = (10, 10)
    current_player: int = 0
    winner: int | None = None
    turn_count: int = 0

    @property
    def done(self) -> bool:
        return self.winner is not None

    def goal_row(self, player: int) -> int:
        return 0 if player == 0 else self.board_size - 1
