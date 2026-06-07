"""Action types used by the Quoridor engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Position = tuple[int, int]
Orientation = Literal["H", "V"]


@dataclass(frozen=True, slots=True)
class MoveAction:
    """Move the current player's pawn to a target square."""

    target: Position


@dataclass(frozen=True, slots=True)
class WallAction:
    """Place a horizontal or vertical 1x2 wall at an anchor coordinate."""

    orientation: Orientation
    row: int
    col: int

    def __post_init__(self) -> None:
        normalized = self.orientation.upper()
        if normalized not in {"H", "V"}:
            raise ValueError("wall orientation must be 'H' or 'V'")
        object.__setattr__(self, "orientation", normalized)


Action = MoveAction | WallAction
