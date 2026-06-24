"""Minimal adversarial policies that target common baseline blind spots."""

from __future__ import annotations

import random
from typing import Sequence

from quoridor.agents.greedy_bfs import GreedyBFSAgent
from quoridor.agents.heuristics import (
    action_sort_key,
    evaluate_action,
    evaluate_state,
    path_diversity,
    ranked_actions,
)
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


class DepthTrapAgent:
    """Exploit depth-limited search by valuing strong post-response followups."""

    def __init__(
        self,
        seed: int | None = None,
        action_limit: int = 24,
        wall_limit: int = 16,
        wall_radius: int = 2,
        victim_action_limit: int = 12,
        followup_limit: int = 8,
        horizon_weight: float = 0.5,
    ) -> None:
        self.rng = random.Random(seed)
        self.action_limit = action_limit
        self.wall_limit = wall_limit
        self.wall_radius = wall_radius
        self.victim_action_limit = victim_action_limit
        self.followup_limit = followup_limit
        self.horizon_weight = horizon_weight

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("DepthTrapAgent received no legal actions")

        player = state.current_player
        candidates = _candidate_actions(
            state,
            legal_actions,
            action_limit=self.action_limit,
            wall_limit=self.wall_limit,
            wall_radius=self.wall_radius,
        )

        def score(action: Action) -> tuple[float, tuple[str, int, int, str]]:
            try:
                after_ours = apply_action(state, action)
                if after_ours.done:
                    return evaluate_state(after_ours, player), action_sort_key(action)

                victim_action = _best_action_for_current_player(
                    after_ours,
                    action_limit=self.victim_action_limit,
                    wall_limit=self.wall_limit,
                    wall_radius=self.wall_radius,
                )
                after_victim = apply_action(after_ours, victim_action)
                immediate = evaluate_state(after_victim, player)
                followups = ranked_actions(
                    after_victim,
                    max_actions=self.followup_limit,
                    wall_limit=self.wall_limit,
                    wall_radius=self.wall_radius,
                )
                if not followups:
                    return immediate, action_sort_key(action)
                best_followup = max(evaluate_action(after_victim, followup, player) for followup in followups)
                horizon_bonus = max(0.0, best_followup - immediate)
                return immediate + self.horizon_weight * horizon_bonus, action_sort_key(action)
            except ValueError:
                return -100_000.0, action_sort_key(action)

        best_value = max(score(action)[0] for action in candidates)
        best_actions = [action for action in candidates if score(action)[0] == best_value]
        return self.rng.choice(sorted(best_actions, key=action_sort_key))


class RolloutPoisonAgent:
    """Prefer actions whose victim responses look ambiguous under shallow rollouts."""

    def __init__(
        self,
        seed: int | None = None,
        action_limit: int = 16,
        wall_limit: int = 10,
        wall_radius: int = 2,
        victim_action_limit: int = 8,
        rollout_depth: int = 4,
        ambiguity_weight: float = 2.0,
    ) -> None:
        self.rng = random.Random(seed)
        self.action_limit = action_limit
        self.wall_limit = wall_limit
        self.wall_radius = wall_radius
        self.victim_action_limit = victim_action_limit
        self.rollout_depth = rollout_depth
        self.ambiguity_weight = ambiguity_weight

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("RolloutPoisonAgent received no legal actions")

        player = state.current_player
        candidates = _candidate_actions(
            state,
            legal_actions,
            action_limit=self.action_limit,
            wall_limit=self.wall_limit,
            wall_radius=self.wall_radius,
        )

        def score(action: Action) -> tuple[float, tuple[str, int, int, str]]:
            try:
                after_ours = apply_action(state, action)
                if after_ours.done:
                    return evaluate_state(after_ours, player), action_sort_key(action)

                victim_scores = _shallow_victim_rollout_scores(
                    after_ours,
                    action_limit=self.victim_action_limit,
                    wall_limit=self.wall_limit,
                    wall_radius=self.wall_radius,
                    depth=self.rollout_depth,
                )
                if not victim_scores:
                    return evaluate_state(after_ours, player), action_sort_key(action)

                victim_action, top_score = victim_scores[0]
                second_score = victim_scores[1][1] if len(victim_scores) > 1 else top_score
                after_victim = apply_action(after_ours, victim_action)
                ambiguity_bonus = self.ambiguity_weight / (1.0 + abs(top_score - second_score))
                return evaluate_state(after_victim, player) + ambiguity_bonus, action_sort_key(action)
            except ValueError:
                return -100_000.0, action_sort_key(action)

        best_value = max(score(action)[0] for action in candidates)
        best_actions = [action for action in candidates if score(action)[0] == best_value]
        return self.rng.choice(sorted(best_actions, key=action_sort_key))


def _candidate_actions(
    state: QuoridorState,
    legal_actions: Sequence[Action],
    *,
    action_limit: int,
    wall_limit: int,
    wall_radius: int,
) -> list[Action]:
    legal_set = set(legal_actions)
    candidates = [
        action
        for action in ranked_actions(
            state,
            max_actions=action_limit,
            wall_limit=wall_limit,
            wall_radius=wall_radius,
        )
        if action in legal_set
    ]
    return candidates or list(legal_actions)


def _best_action_for_current_player(
    state: QuoridorState,
    *,
    action_limit: int,
    wall_limit: int,
    wall_radius: int,
) -> Action:
    player = state.current_player
    actions = ranked_actions(state, max_actions=action_limit, wall_limit=wall_limit, wall_radius=wall_radius)
    if not actions:
        raise ValueError("no victim actions")
    return max(actions, key=lambda action: (evaluate_action(state, action, player), action_sort_key(action)))


def _shallow_victim_rollout_scores(
    state: QuoridorState,
    *,
    action_limit: int,
    wall_limit: int,
    wall_radius: int,
    depth: int,
) -> list[tuple[Action, float]]:
    victim = state.current_player
    actions = ranked_actions(state, max_actions=action_limit, wall_limit=wall_limit, wall_radius=wall_radius)
    scores: list[tuple[Action, float]] = []
    for action in actions:
        rollout_state = apply_action(state, action)
        for _ in range(depth):
            if rollout_state.done:
                break
            rollout_action = _best_action_for_current_player(
                rollout_state,
                action_limit=min(6, action_limit),
                wall_limit=min(4, wall_limit),
                wall_radius=wall_radius,
            )
            rollout_state = apply_action(rollout_state, rollout_action)
        scores.append((action, evaluate_state(rollout_state, victim)))
    return sorted(scores, key=lambda item: (item[1], action_sort_key(item[0])), reverse=True)
