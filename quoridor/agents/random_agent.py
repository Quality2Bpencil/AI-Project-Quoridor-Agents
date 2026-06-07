"""A tiny random agent example."""

from __future__ import annotations

import random
from typing import Sequence

from quoridor.core.actions import Action
from quoridor.core.state import QuoridorState


class RandomAgent:
    """Choose uniformly from the legal actions provided by the engine."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("RandomAgent received no legal actions")
        return self.rng.choice(list(legal_actions))
