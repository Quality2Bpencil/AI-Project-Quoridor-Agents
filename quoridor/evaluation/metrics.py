"""Game execution and metrics collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from quoridor.agents.heuristics import path_distance, path_diversity
from quoridor.core.actions import Action, MoveAction, WallAction
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
    initial_path_lengths: tuple[int, int] = (8, 8)
    final_path_lengths: tuple[int, int] = (8, 8)
    final_path_diversity: tuple[int, int] = (1, 1)
    min_path_diversity: tuple[int, int] = (1, 1)
    move_actions: tuple[int, int] = (0, 0)
    wall_actions: tuple[int, int] = (0, 0)
    trap_events: tuple[int, int] = (0, 0)
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
    initial_paths = _path_lengths(env)
    min_diversity = list(_path_diversities(env))
    move_counts = [0, 0]
    wall_counts = [0, 0]
    trap_counts = [0, 0]

    while not env.state.done and env.state.turn_count < max_turns:
        player = env.state.current_player
        legal_actions = env.legal_actions()
        try:
            action = _choose_action(agents[player], env.state, legal_actions)
            if action not in legal_actions:
                disqualified = player
                break
            previous_paths = _path_lengths(env)
            previous_diversity = _path_diversities(env)
            env.step(action)
            if isinstance(action, MoveAction):
                move_counts[player] += 1
            elif isinstance(action, WallAction):
                wall_counts[player] += 1
            current_paths = _path_lengths(env)
            current_diversity = _path_diversities(env)
            _update_trap_metrics(
                acting_player=player,
                initial_paths=initial_paths,
                previous_paths=previous_paths,
                previous_diversity=previous_diversity,
                current_paths=current_paths,
                current_diversity=current_diversity,
                min_diversity=min_diversity,
                trap_counts=trap_counts,
            )
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
        initial_path_lengths=initial_paths,
        final_path_lengths=_path_lengths(env),
        final_path_diversity=_path_diversities(env),
        min_path_diversity=(min_diversity[0], min_diversity[1]),
        move_actions=(move_counts[0], move_counts[1]),
        wall_actions=(wall_counts[0], wall_counts[1]),
        trap_events=(trap_counts[0], trap_counts[1]),
        actions=tuple(action_log),
    )


def _choose_action(agent: object, state: object, legal_actions: Sequence[Action]) -> Action:
    choose_action = getattr(agent, "choose_action", None)
    if choose_action is None:
        raise TypeError(f"{agent!r} does not implement choose_action")
    return choose_action(state, legal_actions)


def _path_lengths(env: QuoridorEnv) -> tuple[int, int]:
    return path_distance(env.state, 0), path_distance(env.state, 1)


def _path_diversities(env: QuoridorEnv) -> tuple[int, int]:
    return path_diversity(env.state, 0), path_diversity(env.state, 1)


def _update_trap_metrics(
    *,
    acting_player: int,
    initial_paths: tuple[int, int],
    previous_paths: tuple[int, int],
    previous_diversity: tuple[int, int],
    current_paths: tuple[int, int],
    current_diversity: tuple[int, int],
    min_diversity: list[int],
    trap_counts: list[int],
) -> None:
    min_diversity[0] = min(min_diversity[0], current_diversity[0])
    min_diversity[1] = min(min_diversity[1], current_diversity[1])

    opponent = 1 - acting_player
    was_trapped = _is_trap_condition(previous_paths, previous_diversity, initial_paths, opponent)
    is_trapped = _is_trap_condition(current_paths, current_diversity, initial_paths, opponent)
    if is_trapped and not was_trapped:
        trap_counts[acting_player] += 1


def _is_trap_condition(
    path_lengths: tuple[int, int],
    path_diversity_values: tuple[int, int],
    initial_paths: tuple[int, int],
    player: int,
) -> bool:
    return path_diversity_values[player] <= 1 and path_lengths[player] >= initial_paths[player] + 1
