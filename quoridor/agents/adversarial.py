"""Minimal adversarial policies that target common baseline blind spots."""

from __future__ import annotations

import random
from typing import Sequence

from quoridor.agents.greedy_bfs import GreedyBFSAgent
from quoridor.agents.heuristics import action_sort_key, evaluate_state, path_diversity, ranked_actions
from quoridor.core.actions import Action
from quoridor.core.rules import apply_action
from quoridor.core.state import QuoridorState


class PathLureAgent:
    """One-ply adversary for greedy shortest-path agents.

    The policy scores our candidate action after the victim's likely greedy
    response, rewarding states where the victim has fewer shortest-path choices.
    """

    def __init__(
        self,
        seed: int | None = None,
        victim: GreedyBFSAgent | None = None,
        trap_weight: float = 8.0,
        action_limit: int = 24,
        wall_limit: int = 16,
        wall_radius: int = 2,
        victim_action_limit: int = 24,
    ) -> None:
        self.rng = random.Random(seed)
        self.victim = victim or GreedyBFSAgent(seed=seed)
        self.trap_weight = trap_weight
        self.action_limit = action_limit
        self.wall_limit = wall_limit
        self.wall_radius = wall_radius
        self.victim_action_limit = victim_action_limit

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("PathLureAgent received no legal actions")

        player = state.current_player
        opponent = 1 - player
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
            try:
                after_ours = apply_action(state, action)
                if after_ours.done:
                    return evaluate_state(after_ours, player), action_sort_key(action)

                victim_actions = ranked_actions(
                    after_ours,
                    max_actions=self.victim_action_limit,
                    wall_limit=self.wall_limit,
                    wall_radius=self.wall_radius,
                )
                if not victim_actions:
                    return evaluate_state(after_ours, player), action_sort_key(action)

                victim_action = self.victim.choose_action(after_ours, victim_actions)
                after_victim = apply_action(after_ours, victim_action)
                diversity = path_diversity(after_victim, opponent)
                trap_bonus = self.trap_weight * max(0, 2 - diversity)
                return evaluate_state(after_victim, player) + trap_bonus, action_sort_key(action)
            except ValueError:
                return -100_000.0, action_sort_key(action)

        best_value = max(score(action)[0] for action in candidates)
        best_actions = [action for action in candidates if score(action)[0] == best_value]
        return self.rng.choice(sorted(best_actions, key=action_sort_key))
