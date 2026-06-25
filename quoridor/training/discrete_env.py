"""Fixed-action-space adapter for model training."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from quoridor import MoveAction, QuoridorEnv, WallAction
from quoridor.core.actions import Action
from quoridor.core.state import QuoridorState

BOARD_SIZE = 9
WALL_SIZE = BOARD_SIZE - 1
MOVE_ACTIONS = BOARD_SIZE * BOARD_SIZE
WALL_ACTIONS_PER_ORIENTATION = WALL_SIZE * WALL_SIZE
ACTION_SIZE = MOVE_ACTIONS + WALL_ACTIONS_PER_ORIENTATION * 2


@dataclass(frozen=True, slots=True)
class TrainingStep:
    observation: dict[str, Any]
    reward: tuple[int, int]
    done: bool
    info: dict[str, Any]


def action_to_id(action: Action) -> int:
    """Convert an engine action object to a discrete action id in [0, 208]."""

    if isinstance(action, MoveAction):
        row, col = action.target
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            raise ValueError(f"move target must be on the {BOARD_SIZE}x{BOARD_SIZE} board")
        return row * BOARD_SIZE + col

    if not isinstance(action, WallAction):
        raise TypeError("action must be a MoveAction or WallAction")

    if not (0 <= action.row < WALL_SIZE and 0 <= action.col < WALL_SIZE):
        raise ValueError(f"wall anchor must be in [0, {WALL_SIZE - 1}] for row and col")

    offset = MOVE_ACTIONS
    if action.orientation == "V":
        offset += WALL_ACTIONS_PER_ORIENTATION
    return offset + action.row * WALL_SIZE + action.col


def id_to_action(action_id: int) -> Action:
    """Convert a discrete action id in [0, 208] to an engine action object."""

    if not isinstance(action_id, Integral) or isinstance(action_id, bool):
        raise TypeError("action_id must be an integer")
    action_id = int(action_id)
    if not 0 <= action_id < ACTION_SIZE:
        raise ValueError(f"action_id must be in [0, {ACTION_SIZE - 1}]")

    if action_id < MOVE_ACTIONS:
        return MoveAction((action_id // BOARD_SIZE, action_id % BOARD_SIZE))

    wall_id = action_id - MOVE_ACTIONS
    orientation = "H"
    if wall_id >= WALL_ACTIONS_PER_ORIENTATION:
        orientation = "V"
        wall_id -= WALL_ACTIONS_PER_ORIENTATION
    return WallAction(orientation, wall_id // WALL_SIZE, wall_id % WALL_SIZE)


class DiscreteQuoridorEnv:
    """RL-friendly wrapper with fixed discrete actions and legal action masks."""

    action_size = ACTION_SIZE

    def __init__(self, invalid_action_penalty: int | None = None) -> None:
        self.env = QuoridorEnv()
        self.invalid_action_penalty = invalid_action_penalty

    @property
    def state(self) -> QuoridorState:
        return self.env.state

    def reset(self) -> dict[str, Any]:
        self.env.reset()
        return self.observation()

    def legal_action_ids(self) -> list[int]:
        return sorted(action_to_id(action) for action in self.env.legal_actions())

    def legal_action_mask(self) -> list[int]:
        mask = [0] * ACTION_SIZE
        for action_id in self.legal_action_ids():
            mask[action_id] = 1
        return mask

    def step(self, action_id: int) -> TrainingStep:
        action = id_to_action(action_id)
        legal_ids = set(self.legal_action_ids())

        if action_id not in legal_ids:
            if self.invalid_action_penalty is None:
                raise ValueError(f"illegal action id for player {self.state.current_player}: {action_id}")
            reward = [0, 0]
            reward[self.state.current_player] = self.invalid_action_penalty
            return TrainingStep(
                observation=self.observation(),
                reward=(reward[0], reward[1]),
                done=self.state.done,
                info={"invalid_action": True, "action": action},
            )

        result = self.env.step(action)
        return TrainingStep(
            observation=self.observation(),
            reward=result.reward,
            done=result.done,
            info={"invalid_action": False, "action": action, "winner": result.winner},
        )

    def observation(self) -> dict[str, Any]:
        state = self.state
        return {
            "current_player": state.current_player,
            "pawn_planes": self.pawn_planes(state),
            "horizontal_walls": self.wall_plane(state, "H"),
            "vertical_walls": self.wall_plane(state, "V"),
            "remaining_walls": list(state.remaining_walls),
            "legal_action_mask": self.legal_action_mask(),
            "done": state.done,
            "winner": state.winner,
        }

    def flat_observation(self) -> list[float]:
        obs = self.observation()
        values: list[float] = []

        for plane in obs["pawn_planes"]:
            values.extend(float(cell) for row in plane for cell in row)
        values.extend(float(cell) for row in obs["horizontal_walls"] for cell in row)
        values.extend(float(cell) for row in obs["vertical_walls"] for cell in row)
        values.extend(wall_count / 10.0 for wall_count in obs["remaining_walls"])
        values.append(float(obs["current_player"]))
        return values

    @staticmethod
    def pawn_planes(state: QuoridorState) -> list[list[list[int]]]:
        planes = [[[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)] for _ in range(2)]
        for player, (row, col) in enumerate(state.pawn_positions):
            planes[player][row][col] = 1
        return planes

    @staticmethod
    def wall_plane(state: QuoridorState, orientation: str) -> list[list[int]]:
        plane = [[0 for _ in range(WALL_SIZE)] for _ in range(WALL_SIZE)]
        for wall_orientation, row, col in state.walls:
            if wall_orientation == orientation:
                plane[row][col] = 1
        return plane
