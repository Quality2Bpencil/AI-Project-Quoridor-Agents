"""PUCT search with heuristic policy priors and value estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from quoridor.agents.heuristics import action_sort_key, evaluate_action, evaluate_state, ranked_actions
from quoridor.core.actions import Action, MoveAction, WallAction
from quoridor.core.rules import apply_action
from quoridor.core.state import QuoridorState

PriorFn = Callable[[QuoridorState, Sequence[Action]], Mapping[Action, float]]
ValueFn = Callable[[QuoridorState, int], float]


@dataclass
class _PUCTNode:
    state: QuoridorState
    prior: float = 1.0
    parent: "_PUCTNode | None" = None
    action: Action | None = None
    children: dict[Action, "_PUCTNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def expanded(self) -> bool:
        return bool(self.children) or self.state.done

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


class PUCTAgent:
    """AlphaZero-style PUCT baseline using pluggable priors and value estimates.

    The default prior/value functions are heuristic stand-ins for the proposal's
    neural policy/value model. They keep the same interface so a trained model
    can replace them without changing the tournament or UI agent contract.
    """

    def __init__(
        self,
        simulations: int = 32,
        c_puct: float = 1.5,
        action_limit: int = 16,
        wall_limit: int = 8,
        wall_radius: int = 2,
        prior_temperature: float = 3.0,
        value_scale: float = 30.0,
        wall_penalty: float = 2.0,
        wall_candidate_margin: float = 0.0,
        root_blunder_margin: float = 1.0,
        tactical_shortcut_margin: float = 18.0,
        prior_fn: PriorFn | None = None,
        value_fn: ValueFn | None = None,
        seed: int | None = None,
    ) -> None:
        if simulations < 1:
            raise ValueError("simulations must be at least 1")
        if c_puct < 0.0:
            raise ValueError("c_puct must be non-negative")
        if prior_temperature <= 0.0:
            raise ValueError("prior_temperature must be positive")
        if value_scale <= 0.0:
            raise ValueError("value_scale must be positive")
        self.simulations = simulations
        self.c_puct = c_puct
        self.action_limit = action_limit
        self.wall_limit = wall_limit
        self.wall_radius = wall_radius
        self.prior_temperature = prior_temperature
        self.value_scale = value_scale
        self.wall_penalty = wall_penalty
        self.wall_candidate_margin = wall_candidate_margin
        self.root_blunder_margin = root_blunder_margin
        self.tactical_shortcut_margin = tactical_shortcut_margin
        self.prior_fn = prior_fn
        self.value_fn = value_fn
        self.seed = seed
        self._candidate_cache: dict[QuoridorState, list[Action]] = {}
        self._transition_cache: dict[tuple[QuoridorState, Action], QuoridorState] = {}

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("PUCTAgent received no legal actions")

        self._candidate_cache.clear()
        self._transition_cache.clear()
        try:
            root_player = state.current_player
            shortcut = self._tactical_shortcut(state, legal_actions, root_player)
            if shortcut is not None:
                return shortcut

            root = self._search_root(state, legal_actions, root_player)

            if not root.children:
                return sorted(legal_actions, key=action_sort_key)[0]

            chosen = self._best_root_action(root)
            heuristic_best = self._candidate_actions(state, legal_actions)[0]
            if (
                self._evaluate_action_raw(state, chosen, root_player)
                < self._evaluate_action_raw(state, heuristic_best, root_player) - self.root_blunder_margin
            ):
                return heuristic_best
            return chosen
        finally:
            self._candidate_cache.clear()
            self._transition_cache.clear()

    def search_policy(
        self,
        state: QuoridorState,
        legal_actions: Sequence[Action],
        *,
        temperature: float = 1.0,
    ) -> dict[Action, float]:
        """Return the PUCT visit-count policy used by AlphaZero self-play."""

        if not legal_actions:
            raise ValueError("PUCTAgent received no legal actions")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")

        self._candidate_cache.clear()
        self._transition_cache.clear()
        try:
            root_player = state.current_player
            shortcut = self._tactical_shortcut(state, legal_actions, root_player)
            if shortcut is not None:
                return {action: 1.0 if action == shortcut else 0.0 for action in legal_actions}

            root = self._search_root(state, legal_actions, root_player)
            if not root.children:
                uniform = 1.0 / len(legal_actions)
                return {action: uniform for action in legal_actions}

            visits = {action: float(child.visits) for action, child in root.children.items()}
            if temperature <= 1e-8:
                best_action = max(visits, key=lambda action: (visits[action], action_sort_key(action)))
                return {action: 1.0 if action == best_action else 0.0 for action in legal_actions}

            exponent = 1.0 / temperature
            weights = {action: visits.get(action, 0.0) ** exponent for action in legal_actions}
            total = sum(weights.values())
            if total <= 0.0:
                uniform = 1.0 / len(legal_actions)
                return {action: uniform for action in legal_actions}
            return {action: weight / total for action, weight in weights.items()}
        finally:
            self._candidate_cache.clear()
            self._transition_cache.clear()

    def _search_root(
        self,
        state: QuoridorState,
        legal_actions: Sequence[Action],
        root_player: int,
    ) -> _PUCTNode:
        root = _PUCTNode(state=state)
        self._expand(root, root_player, legal_actions)

        for _ in range(self.simulations):
            node = root
            path = [node]
            while node.children and not node.state.done:
                node = self._select_child(node, root_player)
                path.append(node)
            if not node.state.done and not node.children:
                self._expand(node, root_player)
            value = self._value(node.state, root_player)
            self._backpropagate(path, value)
        return root

    def _expand(
        self,
        node: _PUCTNode,
        root_player: int,
        legal_actions: Sequence[Action] | None = None,
    ) -> None:
        actions = self._candidate_actions(node.state, legal_actions)
        if not actions:
            return
        priors = self._priors(node.state, actions, root_player)
        for action in actions:
            if action in node.children:
                continue
            node.children[action] = _PUCTNode(
                state=self._apply_action(node.state, action),
                prior=priors.get(action, 0.0),
                parent=node,
                action=action,
            )

    def _candidate_actions(
        self,
        state: QuoridorState,
        legal_actions: Sequence[Action] | None = None,
    ) -> list[Action]:
        cached = self._candidate_cache.get(state)
        if cached is not None:
            if legal_actions is None:
                return list(cached)
            legal_set = set(legal_actions)
            return [action for action in cached if action in legal_set] or list(legal_actions)

        actions = ranked_actions(
            state,
            max_actions=None,
            wall_limit=self.wall_limit,
            wall_radius=self.wall_radius,
            wall_penalty=self.wall_penalty,
        )
        move_actions = [action for action in actions if isinstance(action, MoveAction)]
        wall_actions = [action for action in actions if isinstance(action, WallAction)]
        if move_actions and wall_actions:
            best_move_score = max(self._evaluate_action_raw(state, action, state.current_player) for action in move_actions)
            wall_actions = [
                action
                for action in wall_actions
                if self._evaluate_action_raw(state, action, state.current_player)
                >= best_move_score + self.wall_candidate_margin
            ]

        filtered = move_actions + wall_actions
        filtered = sorted(
            filtered,
            key=lambda action: (self._evaluate_action_raw(state, action, state.current_player), action_sort_key(action)),
            reverse=True,
        )[: self.action_limit]
        self._candidate_cache[state] = filtered
        if legal_actions is None:
            return list(filtered)
        legal_set = set(legal_actions)
        return [action for action in filtered if action in legal_set] or list(legal_actions)

    def _tactical_shortcut(
        self,
        state: QuoridorState,
        legal_actions: Sequence[Action],
        root_player: int,
    ) -> Action | None:
        candidates = self._candidate_actions(state, legal_actions)
        if not candidates:
            return None
        best = candidates[0]
        if not isinstance(best, WallAction):
            return None
        moves = [action for action in candidates if isinstance(action, MoveAction)]
        if not moves:
            return None
        best_score = self._evaluate_action_raw(state, best, root_player)
        best_move_score = max(self._evaluate_action_raw(state, action, root_player) for action in moves)
        if best_score >= best_move_score + self.tactical_shortcut_margin:
            return best
        return None

    def _priors(self, state: QuoridorState, actions: Sequence[Action], root_player: int) -> dict[Action, float]:
        if self.prior_fn is not None:
            raw = dict(self.prior_fn(state, actions))
            total = sum(max(0.0, raw.get(action, 0.0)) for action in actions)
            if total > 0.0:
                return {action: max(0.0, raw.get(action, 0.0)) / total for action in actions}

        scores = [self._evaluate_action_raw(state, action, state.current_player) for action in actions]
        best = max(scores)
        weights = [
            math.exp(max(-50.0, min(0.0, (score - best) / self.prior_temperature)))
            for score in scores
        ]
        total = sum(weights)
        if total <= 0.0:
            uniform = 1.0 / len(actions)
            return {action: uniform for action in actions}
        return {action: weight / total for action, weight in zip(actions, weights)}

    def _select_child(self, node: _PUCTNode, root_player: int) -> _PUCTNode:
        maximizing_root_value = node.state.current_player == root_player
        sqrt_parent = math.sqrt(max(1, node.visits))
        return max(
            node.children.values(),
            key=lambda child: (
                self._puct_score(child, sqrt_parent, maximizing_root_value),
                action_sort_key(child.action),  # type: ignore[arg-type]
            ),
        )

    def _puct_score(self, child: _PUCTNode, sqrt_parent: float, maximizing_root_value: bool) -> float:
        mean_value = child.mean_value
        if not maximizing_root_value:
            mean_value = -mean_value
        exploration = self.c_puct * child.prior * sqrt_parent / (1 + child.visits)
        return mean_value + exploration

    def _best_root_action(self, root: _PUCTNode) -> Action:
        best_visits = max(child.visits for child in root.children.values())
        candidates = [child for child in root.children.values() if child.visits == best_visits]
        best_value = max(child.mean_value for child in candidates)
        best_children = [child for child in candidates if child.mean_value == best_value]
        return sorted(best_children, key=lambda child: action_sort_key(child.action))[0].action  # type: ignore[return-value,arg-type]

    def _value(self, state: QuoridorState, root_player: int) -> float:
        if self.value_fn is not None:
            return max(-1.0, min(1.0, self.value_fn(state, root_player)))
        return math.tanh(evaluate_state(state, root_player) / self.value_scale)

    def _evaluate_action_raw(self, state: QuoridorState, action: Action, player: int) -> float:
        value = evaluate_action(state, action, player)
        if isinstance(action, WallAction):
            value -= self.wall_penalty
        return value

    def _apply_action(self, state: QuoridorState, action: Action) -> QuoridorState:
        key = (state, action)
        next_state = self._transition_cache.get(key)
        if next_state is None:
            next_state = apply_action(state, action)
            self._transition_cache[key] = next_state
        return next_state

    @staticmethod
    def _backpropagate(path: list[_PUCTNode], value: float) -> None:
        for node in path:
            node.visits += 1
            node.value_sum += value
