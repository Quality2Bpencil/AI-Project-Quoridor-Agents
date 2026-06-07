"""Minimal agent protocol used by examples and demos."""

from __future__ import annotations

from typing import Protocol, Sequence

from quoridor.core.actions import Action
from quoridor.core.state import QuoridorState


class Agent(Protocol):
    """Any agent only needs to choose one action from the legal action list."""

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        ...
