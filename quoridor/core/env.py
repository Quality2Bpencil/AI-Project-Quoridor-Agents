"""Environment-style wrapper around the Quoridor rules."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import Action
from .rules import apply_action, legal_actions
from .state import QuoridorState


@dataclass(frozen=True, slots=True)
class StepResult:
    state: QuoridorState
    reward: tuple[int, int]
    done: bool
    winner: int | None


class QuoridorEnv:
    """Small environment API suitable for human UI or external agents."""

    def __init__(self, board_size: int = 9, walls_per_player: int = 10) -> None:
        if board_size != 9:
            raise ValueError("standard Quoridor uses a 9x9 board")
        self._walls_per_player = walls_per_player
        self.state = self._initial_state(board_size, walls_per_player)

    @staticmethod
    def _initial_state(board_size: int, walls_per_player: int) -> QuoridorState:
        center = board_size // 2
        return QuoridorState(
            board_size=board_size,
            pawn_positions=((board_size - 1, center), (0, center)),
            remaining_walls=(walls_per_player, walls_per_player),
        )

    def reset(self) -> QuoridorState:
        self.state = self._initial_state(self.state.board_size, self._walls_per_player)
        return self.state

    def clone(self) -> "QuoridorEnv":
        env = QuoridorEnv(self.state.board_size, self._walls_per_player)
        env.state = self.state
        return env

    def legal_actions(self) -> list[Action]:
        return legal_actions(self.state)

    def step(self, action: Action) -> StepResult:
        self.state = apply_action(self.state, action)
        reward = (0, 0)
        if self.state.winner == 0:
            reward = (1, -1)
        elif self.state.winner == 1:
            reward = (-1, 1)
        return StepResult(
            state=self.state,
            reward=reward,
            done=self.state.done,
            winner=self.state.winner,
        )
