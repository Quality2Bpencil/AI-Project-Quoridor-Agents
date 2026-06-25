"""UCT Monte Carlo Tree Search baseline agent."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from quoridor.agents.heuristics import action_sort_key, evaluate_state, plausible_wall_actions, ranked_actions
from quoridor.core.actions import Action, MoveAction, WallAction
from quoridor.core.rules import apply_action, legal_pawn_moves
from quoridor.core.state import QuoridorState


@dataclass
class _Node:
    state: QuoridorState
    parent: "_Node | None" = None
    action: Action | None = None
    untried_actions: list[Action] = field(default_factory=list)
    children: list["_Node"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0


class MCTSAgent:
    """Finite-budget UCT search with shallow heuristic rollouts."""

    def __init__(
        self,
        iterations: int = 100,
        exploration: float = math.sqrt(2.0),
        rollout_depth: int = 16,
        action_limit: int = 20,
        wall_limit: int = 10,
        wall_radius: int = 2,
        rollout_move_probability: float = 0.85,
        wall_penalty: float = 2.0,
        wall_candidate_margin: float = 0.0,
        root_blunder_margin: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if not 0.0 <= rollout_move_probability <= 1.0:
            raise ValueError("rollout_move_probability must be in [0, 1]")
        self.iterations = iterations
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.action_limit = action_limit
        self.wall_limit = wall_limit
        self.wall_radius = wall_radius
        self.rollout_move_probability = rollout_move_probability
        self.wall_penalty = wall_penalty
        self.wall_candidate_margin = wall_candidate_margin
        self.root_blunder_margin = root_blunder_margin
        self.rng = random.Random(seed)
        self._candidate_cache: dict[QuoridorState, list[Action]] = {}
        self._transition_cache: dict[tuple[QuoridorState, Action], QuoridorState] = {}

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("MCTSAgent received no legal actions")

        self._candidate_cache.clear()
        self._transition_cache.clear()
        try:
            root_player = state.current_player
            root = _Node(state=state, untried_actions=self._untried_actions(state))
            if not root.untried_actions:
                root.untried_actions = list(reversed(sorted(legal_actions, key=action_sort_key)))

            for _ in range(self.iterations):
                node = self._select(root, root_player)
                if node.untried_actions and not node.state.done:
                    node = self._expand(node)
                value = self._rollout(node.state, root_player)
                self._backpropagate(node, value)

            if not root.children:
                return self.rng.choice(list(legal_actions))

            best_visits = max(child.visits for child in root.children)
            most_visited = [child for child in root.children if child.visits == best_visits]
            best_value = max(child.value / child.visits for child in most_visited)
            best_children = [child for child in most_visited if child.value / child.visits == best_value]
            best_child = sorted(best_children, key=lambda child: action_sort_key(child.action))[0]  # type: ignore[arg-type]
            chosen = best_child.action
            heuristic_best = self._candidate_actions(state)[0]
            if (
                chosen is not None
                and self._evaluate_action(state, chosen, root_player)
                < self._evaluate_action(state, heuristic_best, root_player) - self.root_blunder_margin
            ):
                return heuristic_best
            return chosen  # type: ignore[return-value]
        finally:
            self._candidate_cache.clear()
            self._transition_cache.clear()

    def _candidate_actions(self, state: QuoridorState) -> list[Action]:
        cached = self._candidate_cache.get(state)
        if cached is not None:
            return list(cached)
        actions = ranked_actions(
            state,
            max_actions=None,
            wall_limit=self.wall_limit,
            wall_radius=self.wall_radius,
            wall_penalty=self.wall_penalty,
        )
        if not actions:
            self._candidate_cache[state] = []
            return []

        move_actions = [action for action in actions if isinstance(action, MoveAction)]
        wall_actions = [action for action in actions if isinstance(action, WallAction)]
        if move_actions and wall_actions:
            player = state.current_player
            best_move_score = max(self._evaluate_action(state, action, player) for action in move_actions)
            wall_actions = [
                action
                for action in wall_actions
                if self._evaluate_action(state, action, player) >= best_move_score + self.wall_candidate_margin
            ]

        filtered = move_actions + wall_actions
        filtered = sorted(
            filtered,
            key=lambda action: (self._evaluate_action(state, action, state.current_player), action_sort_key(action)),
            reverse=True,
        )[: self.action_limit]
        self._candidate_cache[state] = filtered
        return list(filtered)

    def _untried_actions(self, state: QuoridorState) -> list[Action]:
        # ranked_actions returns best-first; pop() should therefore expand the
        # strongest heuristic candidate first for low-iteration searches.
        return list(reversed(self._candidate_actions(state)))

    def _select(self, node: _Node, root_player: int) -> _Node:
        while not node.untried_actions and node.children and not node.state.done:
            log_parent = math.log(max(1, node.visits))
            maximizing_root_value = node.state.current_player == root_player
            node = max(
                node.children,
                key=lambda child: self._uct_score(child, log_parent, maximizing_root_value),
            )
        return node

    def _uct_score(self, child: _Node, log_parent: float, maximizing_root_value: bool) -> float:
        if child.visits == 0:
            return math.inf
        mean_value = child.value / child.visits
        if not maximizing_root_value:
            mean_value = -mean_value
        return mean_value + self.exploration * math.sqrt(log_parent / child.visits)

    def _expand(self, node: _Node) -> _Node:
        action = node.untried_actions.pop()
        next_state = self._apply_action(node.state, action)
        child = _Node(
            state=next_state,
            parent=node,
            action=action,
            untried_actions=self._untried_actions(next_state),
        )
        node.children.append(child)
        return child

    def _rollout(self, state: QuoridorState, root_player: int) -> float:
        current = state
        for _ in range(self.rollout_depth):
            if current.done:
                break
            action = self._rollout_policy_action(current)
            if action is None:
                break
            current = self._apply_action(current, action)
        return evaluate_state(current, root_player)

    def _rollout_policy_action(self, state: QuoridorState) -> Action | None:
        move_actions: list[Action] = list(legal_pawn_moves(state))
        if self.rng.random() < self.rollout_move_probability:
            if move_actions:
                return self._rank_biased_choice(self._rank_rollout_actions(state, move_actions))

        wall_actions: list[Action] = list(
            plausible_wall_actions(state, state.current_player, radius=max(1, self.wall_radius - 1))
        )
        if wall_actions:
            wall_cap = max(1, min(3, self.wall_limit))
            return self._rank_biased_choice(self._rank_rollout_actions(state, wall_actions)[:wall_cap])

        if move_actions:
            return self._rank_biased_choice(self._rank_rollout_actions(state, move_actions))
        return None

    def _rank_rollout_actions(self, state: QuoridorState, actions: list[Action]) -> list[Action]:
        player = state.current_player
        return sorted(
            actions,
            key=lambda action: (self._evaluate_action(state, action, player), action_sort_key(action)),
            reverse=True,
        )

    def _evaluate_action(self, state: QuoridorState, action: Action, player: int) -> float:
        next_state = self._apply_action(state, action)
        value = evaluate_state(next_state, player)
        if isinstance(action, WallAction):
            value -= self.wall_penalty
        return value

    def _rank_biased_choice(self, actions: list[Action]) -> Action:
        top_actions = actions[: min(6, len(actions))]
        weights = list(range(len(top_actions), 0, -1))
        return self.rng.choices(top_actions, weights=weights, k=1)[0]

    def _apply_action(self, state: QuoridorState, action: Action) -> QuoridorState:
        key = (state, action)
        next_state = self._transition_cache.get(key)
        if next_state is None:
            next_state = apply_action(state, action)
            self._transition_cache[key] = next_state
        return next_state

    @staticmethod
    def _backpropagate(node: _Node, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent  # type: ignore[assignment]
