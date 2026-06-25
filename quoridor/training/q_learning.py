"""Lightweight tabular Q-learning for Quoridor."""

from __future__ import annotations

import random
from dataclasses import dataclass

from quoridor import QuoridorEnv
from quoridor.agents.heuristics import choose_best, evaluate_state
from quoridor.agents.q_learning import QTable, save_q_table, state_key
from quoridor.core.actions import Action, WallAction
from quoridor.training.discrete_env import action_to_id


@dataclass(frozen=True, slots=True)
class QLearningStats:
    episodes: int
    wins: tuple[int, int]
    draws: int
    q_states: int
    q_entries: int
    final_epsilon: float


def train_q_learning(
    *,
    episodes: int = 100,
    max_turns: int = 120,
    alpha: float = 0.25,
    gamma: float = 0.95,
    epsilon: float = 0.35,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.05,
    terminal_reward: float = 100.0,
    shaping_scale: float = 0.02,
    wall_step_penalty: float = 0.03,
    seed: int | None = None,
) -> tuple[QTable, QLearningStats]:
    """Train one shared current-player-perspective Q-table by self-play."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= epsilon <= 1.0 or not 0.0 <= min_epsilon <= 1.0:
        raise ValueError("epsilon values must be in [0, 1]")

    rng = random.Random(seed)
    q_table: QTable = {}
    wins = [0, 0]
    draws = 0

    for _ in range(episodes):
        env = QuoridorEnv()
        while not env.state.done and env.state.turn_count < max_turns:
            state = env.state
            player = state.current_player
            legal_actions = env.legal_actions()
            action = _select_training_action(q_table, state, legal_actions, epsilon, rng)
            action_id = action_to_id(action)
            key = state_key(state, player)
            before_score = evaluate_state(state, player)

            result = env.step(action)
            reward = terminal_reward * result.reward[player]
            if not result.done:
                reward += shaping_scale * (evaluate_state(result.state, player) - before_score)
            if isinstance(action, WallAction):
                reward -= wall_step_penalty

            next_value = 0.0
            if not result.done and env.state.turn_count < max_turns:
                next_key = state_key(result.state, result.state.current_player)
                next_ids = [action_to_id(next_action) for next_action in env.legal_actions()]
                next_value = _max_q(q_table, next_key, next_ids)

            row = q_table.setdefault(key, {})
            old_value = row.get(action_id, 0.0)
            target = reward - gamma * next_value
            row[action_id] = old_value + alpha * (target - old_value)

        if env.state.winner is None:
            draws += 1
        else:
            wins[env.state.winner] += 1
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    q_entries = sum(len(row) for row in q_table.values())
    return q_table, QLearningStats(
        episodes=episodes,
        wins=(wins[0], wins[1]),
        draws=draws,
        q_states=len(q_table),
        q_entries=q_entries,
        final_epsilon=epsilon,
    )


def save_trained_q_table(q_table: QTable, stats: QLearningStats, path: str) -> None:
    save_q_table(
        q_table,
        path,
        metadata={
            "episodes": stats.episodes,
            "wins": list(stats.wins),
            "draws": stats.draws,
            "q_states": stats.q_states,
            "q_entries": stats.q_entries,
            "final_epsilon": stats.final_epsilon,
        },
    )


def _select_training_action(
    q_table: QTable,
    state,
    legal_actions: list[Action],
    epsilon: float,
    rng: random.Random,
) -> Action:
    if rng.random() < epsilon:
        return rng.choice(legal_actions)

    key = state_key(state, state.current_player)
    row = q_table.get(key, {})
    legal_ids = [action_to_id(action) for action in legal_actions]
    if not any(action_id in row for action_id in legal_ids):
        return choose_best(legal_actions, state, state.current_player)

    best_value = max(row.get(action_id, 0.0) for action_id in legal_ids)
    best_actions = [
        action for action in legal_actions if row.get(action_to_id(action), 0.0) == best_value
    ]
    return rng.choice(best_actions)


def _max_q(q_table: QTable, key, legal_action_ids: list[int]) -> float:
    if not legal_action_ids:
        return 0.0
    row = q_table.get(key, {})
    return max(row.get(action_id, 0.0) for action_id in legal_action_ids)
