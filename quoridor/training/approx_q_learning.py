"""Self-play training for the linear approximate Q-learning agent."""

from __future__ import annotations

import random
from dataclasses import dataclass

from quoridor import QuoridorEnv
from quoridor.agents.approx_q import FeatureVector, Weights, action_features, q_value, save_weights
from quoridor.agents.heuristics import choose_best, evaluate_state
from quoridor.core.actions import Action, WallAction


@dataclass(frozen=True, slots=True)
class ApproxQLearningStats:
    episodes: int
    wins: tuple[int, int]
    draws: int
    nonzero_weights: int
    final_epsilon: float


def train_approx_q_learning(
    *,
    episodes: int = 100,
    max_turns: int = 120,
    alpha: float = 0.05,
    gamma: float = 0.95,
    epsilon: float = 0.35,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.05,
    terminal_reward: float = 1.0,
    shaping_scale: float = 0.05,
    wall_step_penalty: float = 0.02,
    seed: int | None = None,
) -> tuple[Weights, ApproxQLearningStats]:
    """Train a shared current-player-perspective linear Q-function."""

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
    weights: Weights = {}
    wins = [0, 0]
    draws = 0

    for _ in range(episodes):
        env = QuoridorEnv()
        while not env.state.done and env.state.turn_count < max_turns:
            state = env.state
            player = state.current_player
            legal_actions = env.legal_actions()
            action = _select_training_action(weights, state, legal_actions, epsilon, rng)
            features = action_features(state, action, player)
            prediction = _dot(weights, features)
            before_score = evaluate_state(state, player)

            result = env.step(action)
            reward = terminal_reward * result.reward[player]
            if not result.done:
                reward += shaping_scale * _clip((evaluate_state(result.state, player) - before_score) / 100.0, -1.0, 1.0)
            if isinstance(action, WallAction):
                reward -= wall_step_penalty

            next_value = 0.0
            if not result.done and env.state.turn_count < max_turns:
                next_actions = env.legal_actions()
                next_value = max(q_value(weights, result.state, next_action, result.state.current_player) for next_action in next_actions)

            target = reward - gamma * next_value
            error = target - prediction
            for name, value in features.items():
                weights[name] = weights.get(name, 0.0) + alpha * error * value

        if env.state.winner is None:
            draws += 1
        else:
            wins[env.state.winner] += 1
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    nonzero_weights = sum(1 for value in weights.values() if abs(value) > 1e-12)
    return weights, ApproxQLearningStats(
        episodes=episodes,
        wins=(wins[0], wins[1]),
        draws=draws,
        nonzero_weights=nonzero_weights,
        final_epsilon=epsilon,
    )


def save_trained_weights(weights: Weights, stats: ApproxQLearningStats, path: str) -> None:
    save_weights(
        weights,
        path,
        metadata={
            "episodes": stats.episodes,
            "wins": list(stats.wins),
            "draws": stats.draws,
            "nonzero_weights": stats.nonzero_weights,
            "final_epsilon": stats.final_epsilon,
        },
    )


def _select_training_action(
    weights: Weights,
    state,
    legal_actions: list[Action],
    epsilon: float,
    rng: random.Random,
) -> Action:
    if rng.random() < epsilon:
        return rng.choice(legal_actions)
    if not any(abs(weight) > 1e-12 for weight in weights.values()):
        return choose_best(legal_actions, state, state.current_player)

    scored = [(q_value(weights, state, action, state.current_player), action) for action in legal_actions]
    best_value = max(value for value, _ in scored)
    best_actions = [action for value, action in scored if value == best_value]
    return rng.choice(best_actions)


def _dot(weights: Weights, features: FeatureVector) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
