"""Tabular Q-learning policy agent."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

from quoridor.agents.heuristics import action_sort_key, choose_best, evaluate_action, path_distance
from quoridor.core.actions import Action
from quoridor.core.state import QuoridorState
from quoridor.training.discrete_env import action_to_id

StateKey = tuple[int, ...]
QTable = dict[StateKey, dict[int, float]]


def state_key(state: QuoridorState, player: int | None = None, distance_cap: int = 20) -> StateKey:
    """Compact current-player perspective key for tabular learning."""

    player = state.current_player if player is None else player
    opponent = 1 - player

    def relative_position(pos: tuple[int, int]) -> tuple[int, int]:
        row, col = pos
        if player == 1:
            row = state.board_size - 1 - row
        return row, col

    self_row, self_col = relative_position(state.pawn_positions[player])
    opp_row, opp_col = relative_position(state.pawn_positions[opponent])
    self_dist = min(path_distance(state, player), distance_cap)
    opp_dist = min(path_distance(state, opponent), distance_cap)
    return (
        self_row,
        self_col,
        opp_row,
        opp_col,
        self_dist,
        opp_dist,
        state.remaining_walls[player],
        state.remaining_walls[opponent],
        len(state.walls),
    )


class QLearningAgent:
    """Choose legal actions from a tabular Q policy.

    If the table has not seen a state, the agent falls back to the existing
    one-ply heuristic so an untrained policy is still usable in the UI.
    """

    def __init__(
        self,
        q_table: QTable | None = None,
        *,
        table_path: str | Path | None = None,
        epsilon: float = 0.0,
        heuristic_margin: float = 5.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.q_table = q_table if q_table is not None else {}
        if table_path is not None:
            self.q_table.update(load_q_table(table_path))
        self.epsilon = epsilon
        self.heuristic_margin = heuristic_margin
        self.rng = random.Random(seed)

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("QLearningAgent received no legal actions")

        legal_actions = list(legal_actions)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        key = state_key(state)
        row = self.q_table.get(key, {})
        legal_ids = [action_to_id(action) for action in legal_actions]
        known_ids = [action_id for action_id in legal_ids if action_id in row]
        if not known_ids:
            return choose_best(legal_actions, state, state.current_player)

        best_value = max(row.get(action_id, 0.0) for action_id in legal_ids)
        best_actions = [
            action
            for action in legal_actions
            if row.get(action_to_id(action), 0.0) == best_value
        ]
        q_choice = sorted(best_actions, key=action_sort_key)[0]
        heuristic_choice = choose_best(legal_actions, state, state.current_player)
        if (
            evaluate_action(state, q_choice, state.current_player)
            < evaluate_action(state, heuristic_choice, state.current_player) - self.heuristic_margin
        ):
            return heuristic_choice
        return q_choice


def load_q_table(path: str | Path) -> QTable:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    raw_table = payload.get("q_table", payload)
    table: QTable = {}
    for raw_key, raw_values in raw_table.items():
        key = tuple(int(part) for part in raw_key.split(","))
        table[key] = {int(action_id): float(value) for action_id, value in raw_values.items()}
    return table


def save_q_table(q_table: QTable, path: str | Path, metadata: dict[str, object] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "q_table": {
            _encode_key(key): {str(action_id): value for action_id, value in sorted(values.items())}
            for key, values in sorted(q_table.items())
        },
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _encode_key(key: StateKey) -> str:
    return ",".join(str(part) for part in key)
