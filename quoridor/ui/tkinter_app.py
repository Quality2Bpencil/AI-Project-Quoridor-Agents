"""Tkinter visual interface for two human players."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

from quoridor import MoveAction, QuoridorEnv, WallAction
from quoridor.core.actions import Action


class QuoridorTkApp:
    def __init__(self) -> None:
        self.env = QuoridorEnv()
        self.root = tk.Tk()
        self.root.title("Quoridor Engine")

        self.cell = 52
        self.gap = 10
        self.margin = 24
        self.board_px = self.env.state.board_size * self.cell + (self.env.state.board_size - 1) * self.gap
        canvas_size = self.board_px + self.margin * 2

        self.mode = tk.StringVar(value="move")
        self.status = tk.StringVar()

        toolbar = tk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 0))

        tk.Radiobutton(toolbar, text="Move", variable=self.mode, value="move", command=self.draw).pack(side=tk.LEFT)
        tk.Radiobutton(toolbar, text="Horizontal wall", variable=self.mode, value="H", command=self.draw).pack(side=tk.LEFT)
        tk.Radiobutton(toolbar, text="Vertical wall", variable=self.mode, value="V", command=self.draw).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Reset", command=self.reset).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg="#f3ead7", highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.handle_click)

        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill=tk.X, padx=10, pady=(0, 10))

        self.draw()

    def run(self) -> None:
        self.root.mainloop()

    def reset(self) -> None:
        self.env.reset()
        self.mode.set("move")
        self.draw()

    def draw(self) -> None:
        self.canvas.delete("all")
        self.draw_board()
        self.draw_wall_hints()
        self.draw_walls()
        self.draw_pawns()
        self.draw_status()

    def draw_board(self) -> None:
        size = self.env.state.board_size
        for row in range(size):
            for col in range(size):
                x1, y1, x2, y2 = self.cell_rect(row, col)
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="#f8f1df", outline="#9a835f", width=1)

    def draw_wall_hints(self) -> None:
        mode = self.mode.get()
        if mode not in {"H", "V"} or self.env.state.done:
            return

        legal = {action for action in self.env.legal_actions() if isinstance(action, WallAction)}
        for action in legal:
            if action.orientation != mode:
                continue
            x1, y1, x2, y2 = self.wall_rect(action.orientation, action.row, action.col)
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#8ecae6", outline="", stipple="gray50")

    def draw_walls(self) -> None:
        for orientation, row, col in self.env.state.walls:
            x1, y1, x2, y2 = self.wall_rect(orientation, row, col)
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#5a3825", outline="#5a3825")

    def draw_pawns(self) -> None:
        colors = ("#1d4ed8", "#dc2626")
        for player, pos in enumerate(self.env.state.pawn_positions):
            x1, y1, x2, y2 = self.cell_rect(*pos)
            pad = 10
            self.canvas.create_oval(x1 + pad, y1 + pad, x2 - pad, y2 - pad, fill=colors[player], outline="#111827", width=2)

    def draw_status(self) -> None:
        state = self.env.state
        if state.winner is None:
            text = (
                f"Player {state.current_player} turn | "
                f"walls: P0={state.remaining_walls[0]}, P1={state.remaining_walls[1]} | "
                f"mode: {self.mode.get()}"
            )
        else:
            text = f"Player {state.winner} wins. Press Reset to play again."
        self.status.set(text)

    def handle_click(self, event: tk.Event) -> None:
        if self.env.state.done:
            return

        action = self.action_from_click(event.x, event.y)
        if action is None:
            return

        try:
            self.env.step(action)
        except ValueError:
            messagebox.showinfo("Illegal action", "That action is not legal.")
            return
        self.draw()

    def action_from_click(self, x: int, y: int) -> Action | None:
        mode = self.mode.get()
        if mode == "move":
            cell = self.cell_from_point(x, y)
            return MoveAction(cell) if cell is not None else None

        wall = self.wall_from_point(mode, x, y)
        if wall is None:
            return None
        row, col = wall
        return WallAction(mode, row, col)

    def cell_from_point(self, x: int, y: int) -> tuple[int, int] | None:
        size = self.env.state.board_size
        step = self.cell + self.gap
        local_x = x - self.margin
        local_y = y - self.margin
        if local_x < 0 or local_y < 0:
            return None

        col = local_x // step
        row = local_y // step
        if not (0 <= row < size and 0 <= col < size):
            return None
        if local_x % step >= self.cell or local_y % step >= self.cell:
            return None
        return int(row), int(col)

    def wall_from_point(self, orientation: str, x: int, y: int) -> tuple[int, int] | None:
        size = self.env.state.board_size - 1
        best: tuple[int, int] | None = None
        for row in range(size):
            for col in range(size):
                x1, y1, x2, y2 = self.wall_rect(orientation, row, col)
                if x1 - 5 <= x <= x2 + 5 and y1 - 5 <= y <= y2 + 5:
                    best = (row, col)
        return best

    def cell_rect(self, row: int, col: int) -> tuple[int, int, int, int]:
        step = self.cell + self.gap
        x1 = self.margin + col * step
        y1 = self.margin + row * step
        return x1, y1, x1 + self.cell, y1 + self.cell

    def wall_rect(self, orientation: str, row: int, col: int) -> tuple[int, int, int, int]:
        step = self.cell + self.gap
        if orientation == "H":
            x1 = self.margin + col * step
            y1 = self.margin + (row + 1) * self.cell + row * self.gap
            return x1, y1, x1 + self.cell * 2 + self.gap, y1 + self.gap

        x1 = self.margin + (col + 1) * self.cell + col * self.gap
        y1 = self.margin + row * step
        return x1, y1, x1 + self.gap, y1 + self.cell * 2 + self.gap


def main() -> None:
    QuoridorTkApp().run()


if __name__ == "__main__":
    main()
