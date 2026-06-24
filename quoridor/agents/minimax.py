"""Depth-limited minimax agent with alpha-beta pruning."""

from __future__ import annotations

import math
import random
from typing import Sequence

from quoridor.agents.heuristics import WIN_SCORE, action_sort_key, evaluate_state, ranked_actions
from quoridor.core.actions import Action
from quoridor.core.rules import apply_action
from quoridor.core.state import QuoridorState


class MinimaxAgent:
    """Search a pruned action set with alpha-beta minimax."""

    def __init__(
        self,
        depth: int = 2,
        action_limit: int = 24,
        wall_limit: int = 16,
        wall_radius: int = 2,
        seed: int | None = None,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth
        self.action_limit = action_limit
        self.wall_limit = wall_limit
        self.wall_radius = wall_radius
        self.rng = random.Random(seed)

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("MinimaxAgent received no legal actions")

        root_player = state.current_player
        candidates = self._candidate_actions(state)
        legal_set = set(legal_actions)
        candidates = [action for action in candidates if action in legal_set]
        if not candidates:
            candidates = list(legal_actions)

        best_score = -math.inf
        best_actions: list[Action] = []
        for action in candidates:
            next_state = apply_action(state, action)
            score = self._search(next_state, self.depth - 1, root_player, -math.inf, math.inf)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return self.rng.choice(sorted(best_actions, key=action_sort_key))

    def _candidate_actions(self, state: QuoridorState) -> list[Action]:
        return ranked_actions(
            state,
            max_actions=self.action_limit,
            wall_limit=self.wall_limit,
            wall_radius=self.wall_radius,
        )

    def _search(
        self,
        state: QuoridorState,
        depth: int,
        root_player: int,
        alpha: float,
        beta: float,
    ) -> float:
        if depth == 0 or state.done:
            return evaluate_state(state, root_player)

        actions = self._candidate_actions(state)
        if not actions:
            return evaluate_state(state, root_player)

        maximizing = state.current_player == root_player
        if maximizing:
            value = -math.inf
            for action in actions:
                value = max(value, self._search(apply_action(state, action), depth - 1, root_player, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta or value >= WIN_SCORE:
                    break
            return value

        value = math.inf
        for action in actions:
            value = min(value, self._search(apply_action(state, action), depth - 1, root_player, alpha, beta))
            beta = min(beta, value)
            if beta <= alpha or value <= -WIN_SCORE:
                break
        return value
