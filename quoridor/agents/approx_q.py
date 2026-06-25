"""Linear approximate Q-learning policy agent."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

from quoridor.agents.heuristics import (
    UNREACHABLE_DISTANCE,
    action_sort_key,
    choose_best,
    evaluate_state,
    path_distance,
)
from quoridor.core.actions import Action, MoveAction, WallAction
from quoridor.core.rules import apply_action
from quoridor.core.state import QuoridorState

Weights = dict[str, float]
FeatureVector = dict[str, float]


class ApproxQLearningAgent:
    """Choose actions with a linear Q-function over hand-built game features."""

    def __init__(
        self,
        weights: Weights | None = None,
        *,
        weights_path: str | Path | None = None,
        epsilon: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.weights: Weights = dict(weights or {})
        if weights_path is not None:
            self.weights.update(load_weights(weights_path))
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("ApproxQLearningAgent received no legal actions")

        legal_actions = list(legal_actions)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        if not any(abs(weight) > 1e-12 for weight in self.weights.values()):
            return choose_best(legal_actions, state, state.current_player)

        scored = [(q_value(self.weights, state, action, state.current_player), action) for action in legal_actions]
        best_value = max(value for value, _ in scored)
        best_actions = [action for value, action in scored if value == best_value]
        return sorted(best_actions, key=action_sort_key)[0]


def action_features(state: QuoridorState, action: Action, player: int | None = None) -> FeatureVector:
    """Extract bounded current-player-perspective features for one action."""

    player = state.current_player if player is None else player
    opponent = 1 - player
    before_self = _bounded_distance(path_distance(state, player))
    before_opp = _bounded_distance(path_distance(state, opponent))
    before_score = evaluate_state(state, player)

    try:
        after = apply_action(state, action)
    except ValueError:
        return {"bias": 1.0, "illegal": 1.0}

    after_self = _bounded_distance(path_distance(after, player))
    after_opp = _bounded_distance(path_distance(after, opponent))
    after_score = evaluate_state(after, player)

    features: FeatureVector = {
        "bias": 1.0,
        "score_delta": _clip((after_score - before_score) / 100.0, -1.0, 1.0),
        "self_progress": _clip((before_self - after_self) / 8.0, -1.0, 1.0),
        "opp_slowdown": _clip((after_opp - before_opp) / 8.0, -1.0, 1.0),
        "self_distance": -after_self / 20.0,
        "opp_distance": after_opp / 20.0,
        "wall_balance": (after.remaining_walls[player] - after.remaining_walls[opponent]) / 10.0,
        "remaining_walls": after.remaining_walls[player] / 10.0,
        "wall_count": len(after.walls) / 20.0,
    }

    if isinstance(action, MoveAction):
        current_row, current_col = state.pawn_positions[player]
        target_row, target_col = action.target
        direction = -1 if player == 0 else 1
        row_delta = target_row - current_row
        features.update(
            {
                "is_move": 1.0,
                "is_wall": 0.0,
                "move_forward": 1.0 if row_delta == direction else 0.0,
                "move_backward": 1.0 if row_delta == -direction else 0.0,
                "move_sideways": 1.0 if target_row == current_row and target_col != current_col else 0.0,
                "center_file": 1.0 - abs(target_col - state.board_size // 2) / (state.board_size // 2),
            }
        )
    elif isinstance(action, WallAction):
        features.update(
            {
                "is_move": 0.0,
                "is_wall": 1.0,
                "horizontal_wall": 1.0 if action.orientation == "H" else 0.0,
                "vertical_wall": 1.0 if action.orientation == "V" else 0.0,
            }
        )

    if after.winner == player:
        features["terminal_win"] = 1.0
    elif after.winner == opponent:
        features["terminal_loss"] = 1.0
    return features


def q_value(weights: Weights, state: QuoridorState, action: Action, player: int | None = None) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in action_features(state, action, player).items())


def load_weights(path: str | Path) -> Weights:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    raw_weights = payload.get("weights", payload)
    return {str(name): float(value) for name, value in raw_weights.items()}


def save_weights(weights: Weights, path: str | Path, metadata: dict[str, object] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "weights": {name: value for name, value in sorted(weights.items()) if abs(value) > 1e-12},
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _bounded_distance(distance: int) -> int:
    return min(distance, UNREACHABLE_DISTANCE, 20)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
