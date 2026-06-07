"""ASCII rendering helper for quick debugging."""

from __future__ import annotations

from .state import QuoridorState


def render_ascii(state: QuoridorState) -> str:
    size = state.board_size
    lines: list[str] = []

    for row in range(size):
        cell_parts: list[str] = []
        for col in range(size):
            pos = (row, col)
            if pos == state.pawn_positions[0]:
                cell = "A"
            elif pos == state.pawn_positions[1]:
                cell = "B"
            else:
                cell = "."
            cell_parts.append(cell)
            if col < size - 1:
                blocked = ("V", row, col) in state.walls or ("V", row - 1, col) in state.walls
                cell_parts.append("|" if blocked else " ")
        lines.append("".join(cell_parts))

        if row < size - 1:
            wall_parts: list[str] = []
            for col in range(size):
                blocked = ("H", row, col) in state.walls or ("H", row, col - 1) in state.walls
                wall_parts.append("-" if blocked else " ")
                if col < size - 1:
                    wall_parts.append("+")
            lines.append("".join(wall_parts))

    return "\n".join(lines)
