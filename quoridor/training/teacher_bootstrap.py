"""Teacher-guided AlphaZero bootstrap data generation."""

from __future__ import annotations

import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Sequence

from quoridor import QuoridorEnv
from quoridor.core.actions import Action
from quoridor.core.rules import apply_action
from quoridor.training.alphazero import AlphaZeroExample, _draw_value_target, policy_vector
from quoridor.training.discrete_env import DiscreteQuoridorEnv


@dataclass(frozen=True, slots=True)
class TeacherBootstrapStats:
    games: int
    examples: int
    wins: tuple[int, int]
    draws: int
    value_mean_abs: float
    value_nonzero_examples: int
    elapsed_seconds: float


def generate_teacher_bootstrap_examples(
    *,
    games: int = 32,
    max_turns: int = 120,
    seed: int = 0,
    workers: int = 1,
    teacher_profile: str = "mixed_strong",
    draw_value_scale: float = 40.0,
    temperature_turns: int = 12,
) -> tuple[list[AlphaZeroExample], TeacherBootstrapStats]:
    if games < 1:
        raise ValueError("games must be at least 1")
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    started = time.perf_counter()
    worker_count = min(workers, games)
    assignments = _game_assignments(games, worker_count)
    all_examples: list[AlphaZeroExample] = []
    wins = [0, 0]
    draws = 0

    if worker_count == 1:
        results = [
            _generate_teacher_worker(
                games=games,
                max_turns=max_turns,
                seed=seed,
                teacher_profile=teacher_profile,
                draw_value_scale=draw_value_scale,
                temperature_turns=temperature_turns,
            )
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(
                    _generate_teacher_worker,
                    games=count,
                    max_turns=max_turns,
                    seed=seed + offset * 100_003,
                    teacher_profile=teacher_profile,
                    draw_value_scale=draw_value_scale,
                    temperature_turns=temperature_turns,
                )
                for offset, count in enumerate(assignments)
                if count > 0
            ]
            for future in as_completed(futures):
                results.append(future.result())

    for examples, stats in results:
        all_examples.extend(examples)
        wins[0] += stats.wins[0]
        wins[1] += stats.wins[1]
        draws += stats.draws

    value_abs_total = sum(abs(item.value) for item in all_examples)
    value_nonzero = sum(1 for item in all_examples if abs(item.value) > 1e-9)
    stats = TeacherBootstrapStats(
        games=games,
        examples=len(all_examples),
        wins=(wins[0], wins[1]),
        draws=draws,
        value_mean_abs=value_abs_total / len(all_examples) if all_examples else 0.0,
        value_nonzero_examples=value_nonzero,
        elapsed_seconds=time.perf_counter() - started,
    )
    return all_examples, stats


def _generate_teacher_worker(
    *,
    games: int,
    max_turns: int,
    seed: int,
    teacher_profile: str,
    draw_value_scale: float,
    temperature_turns: int,
) -> tuple[list[AlphaZeroExample], TeacherBootstrapStats]:
    rng = random.Random(seed)
    examples: list[AlphaZeroExample] = []
    wins = [0, 0]
    draws = 0
    started = time.perf_counter()

    for game_index in range(games):
        pair = _teacher_pair(teacher_profile, seed + game_index * 17)
        env = QuoridorEnv()
        pending: list[tuple[list[float], list[float], int]] = []
        while not env.state.done and env.state.turn_count < max_turns:
            player = env.state.current_player
            agent = pair[player]
            legal_actions = env.legal_actions()
            policy, action = _teacher_policy_action(
                agent,
                env.state,
                legal_actions,
                rng,
                temperature=1.0 if env.state.turn_count < temperature_turns else 0.0,
            )
            pending.append((_flat_observation(env.state), policy_vector(policy), player))
            env.step(action)

        if env.state.winner is None:
            draws += 1
        else:
            wins[env.state.winner] += 1

        for observation, policy, player in pending:
            if env.state.winner is None:
                value = _draw_value_target(env.state, player, draw_value_scale)
            else:
                value = 1.0 if env.state.winner == player else -1.0
            examples.append(AlphaZeroExample(observation=observation, policy=policy, value=value))

    value_abs_total = sum(abs(item.value) for item in examples)
    value_nonzero = sum(1 for item in examples if abs(item.value) > 1e-9)
    stats = TeacherBootstrapStats(
        games=games,
        examples=len(examples),
        wins=(wins[0], wins[1]),
        draws=draws,
        value_mean_abs=value_abs_total / len(examples) if examples else 0.0,
        value_nonzero_examples=value_nonzero,
        elapsed_seconds=time.perf_counter() - started,
    )
    return examples, stats


def _teacher_pair(profile: str, seed: int) -> tuple[object, object]:
    from quoridor.agents.adversarial import CounterfactualTrapAgent, DepthTrapAgent
    from quoridor.agents.mcts import MCTSAgent
    from quoridor.agents.minimax import MinimaxAgent
    from quoridor.agents.puct import PUCTAgent

    if profile == "fast":
        return (
            PUCTAgent(simulations=16, action_limit=14, wall_limit=8, seed=seed),
            MinimaxAgent(depth=1, action_limit=14, wall_limit=8, seed=seed + 1),
        )
    if profile != "mixed_strong":
        raise ValueError(f"unknown teacher_profile: {profile}")

    options = [
        PUCTAgent(simulations=48, action_limit=24, wall_limit=14, wall_radius=3, seed=seed),
        MinimaxAgent(depth=2, action_limit=18, wall_limit=10, wall_radius=3, seed=seed + 1),
        MCTSAgent(iterations=48, rollout_depth=10, action_limit=18, wall_limit=10, wall_radius=3, seed=seed + 2),
        DepthTrapAgent(action_limit=18, wall_limit=10, wall_radius=3, seed=seed + 3),
        CounterfactualTrapAgent(action_limit=18, wall_limit=10, wall_radius=3, response_width=2, seed=seed + 4),
    ]
    first = options[seed % len(options)]
    second = options[(seed // len(options) + 1) % len(options)]
    return first, second


def _teacher_policy_action(
    agent: object,
    state: object,
    legal_actions: Sequence[Action],
    rng: random.Random,
    *,
    temperature: float,
) -> tuple[dict[Action, float], Action]:
    search_policy = getattr(agent, "search_policy", None)
    if search_policy is not None:
        policy = dict(search_policy(state, legal_actions, temperature=temperature))
        action = _sample_policy_action(policy, rng) if temperature > 0.0 else _best_policy_action(policy)
        return policy, action

    choose_action = getattr(agent, "choose_action")
    action = choose_action(state, legal_actions)
    return {candidate: 1.0 if candidate == action else 0.0 for candidate in legal_actions}, action


def _sample_policy_action(policy: dict[Action, float], rng: random.Random) -> Action:
    actions = list(policy)
    weights = [max(0.0, policy[action]) for action in actions]
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(actions)
    return rng.choices(actions, weights=weights, k=1)[0]


def _best_policy_action(policy: dict[Action, float]) -> Action:
    from quoridor.agents.heuristics import action_sort_key

    return max(policy, key=lambda action: (policy[action], action_sort_key(action)))


def _flat_observation(state: object) -> list[float]:
    wrapper = DiscreteQuoridorEnv()
    wrapper.env.state = state  # type: ignore[assignment]
    return wrapper.flat_observation()


def _game_assignments(games: int, workers: int) -> list[int]:
    base = games // workers
    extra = games % workers
    return [base + (1 if index < extra else 0) for index in range(workers)]
