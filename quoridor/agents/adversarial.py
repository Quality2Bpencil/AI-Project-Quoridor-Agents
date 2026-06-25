"""Minimal adversarial policies that target common baseline blind spots."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

from quoridor.agents.greedy_bfs import GreedyBFSAgent
from quoridor.agents.heuristics import (
    action_sort_key,
    evaluate_action,
    evaluate_state,
    path_distance,
    path_diversity,
    ranked_actions,
)
from quoridor.agents.q_learning import QLearningAgent
from quoridor.core.actions import Action, WallAction
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
        path_delta_weight: float = 4.0,
        followup_limit: int = 6,
        followup_weight: float = 0.25,
        wall_overuse_penalty: float = 0.5,
        bad_wall_penalty: float = 35.0,
    ) -> None:
        self.rng = random.Random(seed)
        self.victim = victim or GreedyBFSAgent(seed=seed)
        self.trap_weight = trap_weight
        self.action_limit = action_limit
        self.wall_limit = wall_limit
        self.wall_radius = wall_radius
        self.victim_action_limit = victim_action_limit
        self.path_delta_weight = path_delta_weight
        self.followup_limit = followup_limit
        self.followup_weight = followup_weight
        self.wall_overuse_penalty = wall_overuse_penalty
        self.bad_wall_penalty = bad_wall_penalty

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
                before_opponent_path = path_distance(state, opponent)
                before_opponent_diversity = path_diversity(state, opponent)
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
                opponent_path = path_distance(after_victim, opponent)
                diversity = path_diversity(after_victim, opponent)
                path_delta = opponent_path - before_opponent_path
                diversity_drop = max(0, before_opponent_diversity - diversity)
                trap_bonus = self.trap_weight * max(0, 2 - diversity)
                immediate = evaluate_state(after_victim, player)
                value = immediate
                value += self.path_delta_weight * path_delta
                value += self.trap_weight * diversity_drop
                value += trap_bonus
                value += self._followup_bonus(after_victim, player, immediate)
                if isinstance(action, WallAction):
                    value -= self.wall_overuse_penalty
                    if path_delta <= 0 and diversity_drop <= 0:
                        value -= self.bad_wall_penalty
                return value, action_sort_key(action)
            except ValueError:
                return -100_000.0, action_sort_key(action)

        best_value = max(score(action)[0] for action in candidates)
        best_actions = [action for action in candidates if score(action)[0] == best_value]
        return self.rng.choice(sorted(best_actions, key=action_sort_key))

    def _followup_bonus(self, state: QuoridorState, player: int, immediate: float) -> float:
        followups = ranked_actions(
            state,
            max_actions=self.followup_limit,
            wall_limit=min(self.wall_limit, self.followup_limit),
            wall_radius=self.wall_radius,
        )
        if not followups:
            return 0.0
        best_followup = max(evaluate_action(state, followup, player) for followup in followups)
        return self.followup_weight * max(0.0, best_followup - immediate)


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


class CounterfactualTrapAgent:
    """Robust trap policy scored against several plausible victim responses."""

    def __init__(
        self,
        seed: int | None = None,
        victim: object | None = None,
        action_limit: int = 18,
        wall_limit: int = 10,
        wall_radius: int = 2,
        victim_action_limit: int = 8,
        response_width: int = 2,
        followup_limit: int = 6,
        trap_weight: float = 7.0,
        path_delta_weight: float = 2.5,
        followup_weight: float = 0.35,
    ) -> None:
        self.rng = random.Random(seed)
        self.victim = victim
        self.action_limit = action_limit
        self.wall_limit = wall_limit
        self.wall_radius = wall_radius
        self.victim_action_limit = victim_action_limit
        self.response_width = max(1, response_width)
        self.followup_limit = followup_limit
        self.trap_weight = trap_weight
        self.path_delta_weight = path_delta_weight
        self.followup_weight = followup_weight

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("CounterfactualTrapAgent received no legal actions")

        player = state.current_player
        opponent = 1 - player
        candidates = _candidate_actions(
            state,
            legal_actions,
            action_limit=self.action_limit,
            wall_limit=self.wall_limit,
            wall_radius=self.wall_radius,
        )

        scored = [(self._score_action(state, action, player, opponent), action) for action in candidates]
        best_value = max(value for value, _ in scored)
        best_actions = [action for value, action in scored if value == best_value]
        return self.rng.choice(sorted(best_actions, key=action_sort_key))

    def _score_action(self, state: QuoridorState, action: Action, player: int, opponent: int) -> float:
        before_opponent_path = path_distance(state, opponent)
        try:
            after_ours = apply_action(state, action)
        except ValueError:
            return -100_000.0
        if after_ours.done:
            return evaluate_state(after_ours, player)

        responses = self._plausible_responses(after_ours)
        if not responses:
            return evaluate_state(after_ours, player)

        outcomes: list[float] = []
        for response in responses:
            try:
                after_response = apply_action(after_ours, response)
            except ValueError:
                continue
            immediate = evaluate_state(after_response, player)
            trap_bonus = self._trap_bonus(after_response, opponent, before_opponent_path)
            followup_bonus = self._followup_bonus(after_response, player, immediate)
            outcomes.append(immediate + trap_bonus + followup_bonus)

        return min(outcomes) if outcomes else -100_000.0

    def _plausible_responses(self, state: QuoridorState) -> list[Action]:
        actions = ranked_actions(
            state,
            max_actions=self.victim_action_limit,
            wall_limit=self.wall_limit,
            wall_radius=self.wall_radius,
        )
        if not actions:
            return []

        responses: list[Action] = []
        choose_action = getattr(self.victim, "choose_action", None)
        if choose_action is not None:
            try:
                chosen = choose_action(state, actions)
                if chosen in actions:
                    responses.append(chosen)
            except Exception:
                pass

        for action in actions:
            if action not in responses:
                responses.append(action)
            if len(responses) >= self.response_width:
                break
        return responses

    def _trap_bonus(self, state: QuoridorState, opponent: int, before_opponent_path: int) -> float:
        opponent_path = path_distance(state, opponent)
        opponent_diversity = path_diversity(state, opponent)
        path_delta = max(0, opponent_path - before_opponent_path)
        transition_bonus = self.trap_weight if path_delta > 0 and opponent_diversity <= 1 else 0.0
        return self.path_delta_weight * path_delta + transition_bonus

    def _followup_bonus(self, state: QuoridorState, player: int, immediate: float) -> float:
        followups = ranked_actions(
            state,
            max_actions=self.followup_limit,
            wall_limit=min(self.wall_limit, self.followup_limit),
            wall_radius=self.wall_radius,
        )
        if not followups:
            return 0.0
        best_followup = max(evaluate_action(state, followup, player) for followup in followups)
        return self.followup_weight * max(0.0, best_followup - immediate)


class ArgmaxQTrapAgent:
    """Counterfactual trap policy specialized for deterministic Q victims."""

    def __init__(
        self,
        seed: int | None = None,
        table_path: str | Path | None = Path("experiments/results/q_learning_policy.json"),
        victim: object | None = None,
        action_limit: int = 18,
        wall_limit: int = 10,
        wall_radius: int = 2,
        victim_action_limit: int = 8,
        response_width: int = 1,
        followup_limit: int = 6,
        trap_weight: float = 10.0,
        path_delta_weight: float = 4.0,
        followup_weight: float = 0.6,
    ) -> None:
        q_victim = victim or QLearningAgent(seed=seed, table_path=table_path, epsilon=0.0)
        self.search = CounterfactualTrapAgent(
            seed=seed,
            victim=q_victim,
            action_limit=action_limit,
            wall_limit=wall_limit,
            wall_radius=wall_radius,
            victim_action_limit=victim_action_limit,
            response_width=response_width,
            followup_limit=followup_limit,
            trap_weight=trap_weight,
            path_delta_weight=path_delta_weight,
            followup_weight=followup_weight,
        )

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        return self.search.choose_action(state, legal_actions)


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
