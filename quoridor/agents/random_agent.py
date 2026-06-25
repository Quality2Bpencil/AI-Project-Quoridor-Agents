"""A tiny random agent example."""

from __future__ import annotations

import random
from typing import Sequence

from quoridor.core.actions import Action, MoveAction, WallAction
from quoridor.core.state import QuoridorState


class RandomAgent:
    """Choose a random legal action with an explicit move/wall mix.

    Uniformly sampling from every legal action makes the opening wall rate about
    128/(128+3), which is not a useful random baseline. Instead, first decide
    whether this turn should consider a wall, then sample uniformly inside that
    action type.
    """

    def __init__(self, seed: int | None = None, wall_probability: float = 0.18) -> None:
        if not 0.0 <= wall_probability <= 1.0:
            raise ValueError("wall_probability must be in [0, 1]")
        self.rng = random.Random(seed)
        self.wall_probability = wall_probability

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("RandomAgent received no legal actions")

        moves = [action for action in legal_actions if isinstance(action, MoveAction)]
        walls = [action for action in legal_actions if isinstance(action, WallAction)]
        if walls and self.rng.random() < self.wall_probability:
            return self.rng.choice(walls)
        if moves:
            return self.rng.choice(moves)
        return self.rng.choice(walls)
