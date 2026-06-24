"""Greedy shortest-path baseline agent."""

from __future__ import annotations

import random
from typing import Sequence

from quoridor.agents.heuristics import action_sort_key, evaluate_action, ranked_actions
from quoridor.core.actions import Action, WallAction
from quoridor.core.state import QuoridorState


class GreedyBFSAgent:
    """Choose the one-ply action with the best shortest-path heuristic score."""

    def __init__(
        self,
        seed: int | None = None,
        wall_penalty: float = 0.0,
        action_limit: int = 32,
        wall_limit: int = 24,
        wall_radius: int = 2,
    ) -> None:
        self.rng = random.Random(seed)
        self.wall_penalty = wall_penalty
        self.action_limit = action_limit
        self.wall_limit = wall_limit
        self.wall_radius = wall_radius

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("GreedyBFSAgent received no legal actions")

        player = state.current_player
        legal_set = set(legal_actions)
        candidates = [
            action
            for action in ranked_actions(
                state,
                max_actions=self.action_limit,
                wall_limit=self.wall_limit,
                wall_radius=self.wall_radius,
            )
            if action in legal_set
        ]
        if not candidates:
            candidates = list(legal_actions)

        def score(action: Action) -> tuple[float, tuple[str, int, int, str]]:
            value = evaluate_action(state, action, player)
            if isinstance(action, WallAction):
                value -= self.wall_penalty
            return value, action_sort_key(action)

        best_value = max(score(action)[0] for action in candidates)
        best_actions = [action for action in candidates if score(action)[0] == best_value]
        return self.rng.choice(sorted(best_actions, key=action_sort_key))
