"""Game execution and metrics collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from quoridor.core.actions import Action
from quoridor.core.env import QuoridorEnv


@dataclass(frozen=True, slots=True)
class GameRecord:
    agent0: str
    agent1: str
    winner: int | None
    turns: int
    max_turns_reached: bool
    disqualified_player: int | None = None
    remaining_walls: tuple[int, int] = (10, 10)
    actions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def winner_name(self) -> str:
        if self.winner is None:
            return "draw"
        return self.agent0 if self.winner == 0 else self.agent1


def play_game(
    agent0: object,
    agent1: object,
    *,
    agent0_name: str = "agent0",
    agent1_name: str = "agent1",
    max_turns: int = 200,
    record_actions: bool = False,
) -> GameRecord:
    """Play one game and return a compact record.

    Invalid actions are treated as disqualifications so tournaments do not crash
    when a candidate agent is still under development.
    """

    env = QuoridorEnv()
    agents = (agent0, agent1)
    action_log: list[str] = []
    disqualified: int | None = None

    while not env.state.done and env.state.turn_count < max_turns:
        player = env.state.current_player
        legal_actions = env.legal_actions()
        try:
            action = _choose_action(agents[player], env.state, legal_actions)
            if action not in legal_actions:
                disqualified = player
                break
            env.step(action)
            if record_actions:
                action_log.append(repr(action))
        except Exception:
            disqualified = player
            break

    winner = env.state.winner
    if disqualified is not None:
        winner = 1 - disqualified

    return GameRecord(
        agent0=agent0_name,
        agent1=agent1_name,
        winner=winner,
        turns=env.state.turn_count,
        max_turns_reached=(winner is None and env.state.turn_count >= max_turns),
        disqualified_player=disqualified,
        remaining_walls=env.state.remaining_walls,
        actions=tuple(action_log),
    )


def _choose_action(agent: object, state: object, legal_actions: Sequence[Action]) -> Action:
    choose_action = getattr(agent, "choose_action", None)
    if choose_action is None:
        raise TypeError(f"{agent!r} does not implement choose_action")
    return choose_action(state, legal_actions)
