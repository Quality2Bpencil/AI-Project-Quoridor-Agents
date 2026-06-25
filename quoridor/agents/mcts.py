"""UCT Monte Carlo Tree Search baseline agent."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from quoridor.agents.heuristics import action_sort_key, evaluate_state, ranked_actions
from quoridor.core.actions import Action
from quoridor.core.rules import apply_action
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
        rollout_depth: int = 24,
        action_limit: int = 32,
        wall_limit: int = 20,
        wall_radius: int = 2,
        seed: int | None = None,
    ) -> None:
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        self.iterations = iterations
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.action_limit = action_limit
        self.wall_limit = wall_limit
        self.wall_radius = wall_radius
        self.rng = random.Random(seed)

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("MCTSAgent received no legal actions")

        root_player = state.current_player
        root = _Node(state=state, untried_actions=self._candidate_actions(state))
        if not root.untried_actions:
            root.untried_actions = list(legal_actions)
        self.rng.shuffle(root.untried_actions)

        for _ in range(self.iterations):
            node = self._select(root)
            if node.untried_actions and not node.state.done:
                node = self._expand(node)
            value = self._rollout(node.state, root_player)
            self._backpropagate(node, value)

        if not root.children:
            return self.rng.choice(list(legal_actions))

        best_visits = max(child.visits for child in root.children)
        best_children = [child for child in root.children if child.visits == best_visits]
        best_child = self.rng.choice(sorted(best_children, key=lambda child: action_sort_key(child.action)))  # type: ignore[arg-type]
        return best_child.action  # type: ignore[return-value]

    def _candidate_actions(self, state: QuoridorState) -> list[Action]:
        return ranked_actions(
            state,
            max_actions=self.action_limit,
            wall_limit=self.wall_limit,
            wall_radius=self.wall_radius,
        )

    def _select(self, node: _Node) -> _Node:
        while not node.untried_actions and node.children and not node.state.done:
            log_parent = math.log(max(1, node.visits))
            node = max(
                node.children,
                key=lambda child: (child.value / child.visits)
                + self.exploration * math.sqrt(log_parent / child.visits),
            )
        return node

    def _expand(self, node: _Node) -> _Node:
        action = node.untried_actions.pop()
        next_state = apply_action(node.state, action)
        child = _Node(
            state=next_state,
            parent=node,
            action=action,
            untried_actions=self._candidate_actions(next_state),
        )
        self.rng.shuffle(child.untried_actions)
        node.children.append(child)
        return child

    def _rollout(self, state: QuoridorState, root_player: int) -> float:
        current = state
        for _ in range(self.rollout_depth):
            if current.done:
                break
            actions = self._candidate_actions(current)
            if not actions:
                break
            action = self.rng.choice(actions[: min(6, len(actions))])
            current = apply_action(current, action)
        return evaluate_state(current, root_player)

    @staticmethod
    def _backpropagate(node: _Node, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent  # type: ignore[assignment]
