"""AlphaZero-style policy/value network utilities for Quoridor."""

from __future__ import annotations

import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import tanh
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from quoridor import QuoridorEnv
from quoridor.core.actions import Action
from quoridor.core.rules import legal_pawn_moves, shortest_path_length
from quoridor.training.discrete_env import ACTION_SIZE, DiscreteQuoridorEnv
from quoridor.training.discrete_env import action_to_id

UNREACHABLE_DISTANCE = 1_000


class AlphaZeroNet(nn.Module):
    """Small policy/value network over the existing flat Quoridor encoding."""

    def __init__(self, obs_dim: int, action_size: int = ACTION_SIZE, hidden_size: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, max(32, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(32, hidden_size // 2), 1),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(obs)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


@dataclass(frozen=True, slots=True)
class AlphaZeroCheckpoint:
    model: AlphaZeroNet
    metadata: dict[str, Any]
    device: torch.device


@dataclass(frozen=True, slots=True)
class AlphaZeroExample:
    observation: list[float]
    policy: list[float]
    value: float


@dataclass(frozen=True, slots=True)
class AlphaZeroSelfPlayStats:
    games: int
    examples: int
    updates: int
    wins: tuple[int, int]
    draws: int
    device: str
    elapsed_seconds: float
    mcts_batch_size: int = 1
    value_mean_abs: float = 0.0
    value_nonzero_examples: int = 0


class AlphaZeroBatchEvaluator:
    """Batched policy/value evaluator for PUCT leaf expansion."""

    def __init__(self, model: AlphaZeroNet, device: torch.device, *, cache_size: int = 0) -> None:
        self.model = model
        self.device = device
        self.cache_size = max(0, cache_size)
        self.calls = 0
        self.requested_states = 0
        self.forward_states = 0
        self.cache_hits = 0
        self.batch_sizes: list[int] = []
        self._cache: dict[object, tuple[torch.Tensor, float]] = {}
        self._cache_order: list[object] = []

    def evaluate(
        self,
        requests: Sequence[tuple[object, Sequence[Action], int]],
    ) -> list[tuple[dict[Action, float], float]]:
        if not requests:
            return []

        self.calls += 1
        self.requested_states += len(requests)
        missing_states: list[object] = []
        missing_indexes: list[int] = []
        cached: list[tuple[torch.Tensor, float] | None] = []
        for state, _, _ in requests:
            item = self._cache.get(state)
            if item is None:
                missing_indexes.append(len(cached))
                missing_states.append(state)
                cached.append(None)
            else:
                self.cache_hits += 1
                cached.append(item)

        if missing_states:
            self.batch_sizes.append(len(missing_states))
            self.forward_states += len(missing_states)
            obs = torch.tensor(
                [_flat_observation(state) for state in missing_states],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                logits_batch, values_batch = self.model(obs)
            logits_cpu = logits_batch.detach().cpu()
            values_cpu = values_batch.detach().cpu().tolist()
            for state, index, logits, value in zip(missing_states, missing_indexes, logits_cpu, values_cpu):
                item = (logits, float(value))
                cached[index] = item
                self._remember(state, item)

        output: list[tuple[dict[Action, float], float]] = []
        for item, (state, actions, root_player) in zip(cached, requests):
            if item is None:
                raise RuntimeError("internal AlphaZero evaluator cache miss")
            logits, current_value = item
            priors = _policy_from_logits(logits, actions)
            value = current_value if getattr(state, "current_player") == root_player else -current_value
            output.append((priors, max(-1.0, min(1.0, value))))
        return output

    def _remember(self, state: object, item: tuple[torch.Tensor, float]) -> None:
        if self.cache_size <= 0:
            return
        if state in self._cache:
            self._cache[state] = item
            return
        self._cache[state] = item
        self._cache_order.append(state)
        while len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)


@dataclass(frozen=True, slots=True)
class AlphaZeroTrainStats:
    examples: int
    updates: int
    epochs: int
    batch_size: int
    device: str
    elapsed_seconds: float
    loss: float
    value_mean_abs: float = 0.0
    value_nonzero_examples: int = 0


def default_obs_dim() -> int:
    env = DiscreteQuoridorEnv()
    env.reset()
    return len(env.flat_observation())


def policy_vector(policy: Mapping[Action, float]) -> list[float]:
    vector = [0.0] * ACTION_SIZE
    for action, probability in policy.items():
        vector[action_to_id(action)] = float(probability)
    total = sum(vector)
    if total > 0.0:
        vector = [value / total for value in vector]
    return vector


def alphazero_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
) -> torch.Tensor:
    value_loss = F.mse_loss(values, target_value)
    policy_loss = -(target_policy * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    return value_loss + policy_loss


def train_alphazero_self_play(
    *,
    games: int = 4,
    simulations: int = 8,
    max_turns: int = 80,
    hidden_size: int = 128,
    action_limit: int = 10,
    wall_limit: int = 5,
    batch_size: int = 32,
    epochs_per_game: int = 1,
    lr: float = 1e-3,
    temperature_turns: int = 8,
    replay_capacity: int = 100_000,
    draw_value_mode: str = "zero",
    draw_value_scale: float = 40.0,
    root_dirichlet_alpha: float = 0.3,
    root_noise_fraction: float = 0.25,
    mcts_batch_size: int = 1,
    inference_cache_size: int = 4096,
    seed: int | None = None,
    device: str | None = None,
    initial_checkpoint: str | Path | None = None,
) -> tuple[AlphaZeroNet, AlphaZeroSelfPlayStats]:
    if games < 1:
        raise ValueError("games must be at least 1")
    if simulations < 1:
        raise ValueError("simulations must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if replay_capacity < batch_size:
        raise ValueError("replay_capacity must be at least batch_size")
    if draw_value_mode not in {"zero", "heuristic"}:
        raise ValueError("draw_value_mode must be 'zero' or 'heuristic'")
    if draw_value_scale <= 0:
        raise ValueError("draw_value_scale must be positive")
    if mcts_batch_size < 1:
        raise ValueError("mcts_batch_size must be at least 1")
    if inference_cache_size < 0:
        raise ValueError("inference_cache_size must be non-negative")

    rng = random.Random(seed)
    torch.manual_seed(seed or 0)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if initial_checkpoint is not None:
        checkpoint = load_alphazero_checkpoint(initial_checkpoint, device=selected_device)
        model = checkpoint.model
    else:
        model = AlphaZeroNet(default_obs_dim(), hidden_size=hidden_size).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)

    replay: list[AlphaZeroExample] = []
    wins = [0, 0]
    draws = 0
    updates = 0
    total_examples = 0
    value_abs_total = 0.0
    value_nonzero_examples = 0
    started = time.perf_counter()

    for _ in range(games):
        from quoridor.agents.puct import PUCTAgent

        env = QuoridorEnv()
        pending: list[tuple[list[float], list[float], int]] = []
        evaluator = AlphaZeroBatchEvaluator(model, selected_device, cache_size=inference_cache_size)
        while not env.state.done and env.state.turn_count < max_turns:
            player = env.state.current_player
            legal_actions = env.legal_actions()
            search = PUCTAgent(
                simulations=simulations,
                action_limit=action_limit,
                wall_limit=wall_limit,
                prior_fn=lambda state, actions: evaluator.evaluate([(state, actions, state.current_player)])[0][0],
                value_fn=lambda state, root_player: evaluator.evaluate([(state, (), root_player)])[0][1],
                policy_value_batch_fn=evaluator.evaluate,
                inference_batch_size=mcts_batch_size,
                root_dirichlet_alpha=root_dirichlet_alpha,
                root_noise_fraction=root_noise_fraction,
                seed=rng.randrange(2**31),
            )
            temperature = 1.0 if env.state.turn_count < temperature_turns else 0.0
            visit_policy = search.search_policy(env.state, legal_actions, temperature=temperature)
            pending.append((_flat_observation(env.state), policy_vector(visit_policy), player))
            action = _sample_policy_action(visit_policy, rng)
            env.step(action)

        if env.state.winner is None:
            draws += 1
        else:
            wins[env.state.winner] += 1

        for observation, policy, player in pending:
            value = 0.0
            if env.state.winner is not None:
                value = 1.0 if env.state.winner == player else -1.0
            elif draw_value_mode == "heuristic":
                value = _draw_value_target(env.state, player, draw_value_scale)
            value_abs_total += abs(value)
            if abs(value) > 1e-9:
                value_nonzero_examples += 1
            replay.append(AlphaZeroExample(observation=observation, policy=policy, value=value))
        total_examples += len(pending)
        if len(replay) > replay_capacity:
            del replay[: len(replay) - replay_capacity]

        updates += _train_replay_epochs(model, optimizer, replay, batch_size, epochs_per_game, selected_device, rng)

    stats = AlphaZeroSelfPlayStats(
        games=games,
        examples=total_examples,
        updates=updates,
        wins=(wins[0], wins[1]),
        draws=draws,
        device=str(selected_device),
        elapsed_seconds=time.perf_counter() - started,
        mcts_batch_size=mcts_batch_size,
        value_mean_abs=value_abs_total / total_examples if total_examples else 0.0,
        value_nonzero_examples=value_nonzero_examples,
    )
    return model, stats


def train_alphazero_examples(
    examples: Sequence[AlphaZeroExample],
    *,
    hidden_size: int = 256,
    batch_size: int = 256,
    epochs: int = 4,
    lr: float = 1e-3,
    seed: int | None = None,
    device: str | None = None,
    initial_checkpoint: str | Path | None = None,
) -> tuple[AlphaZeroNet, AlphaZeroTrainStats]:
    if not examples:
        raise ValueError("examples must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    torch.manual_seed(seed or 0)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if initial_checkpoint is not None:
        checkpoint = load_alphazero_checkpoint(initial_checkpoint, device=selected_device)
        model = checkpoint.model
    else:
        model = AlphaZeroNet(default_obs_dim(), hidden_size=hidden_size).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)

    observations = torch.tensor([item.observation for item in examples], dtype=torch.float32)
    target_policy = torch.tensor([item.policy for item in examples], dtype=torch.float32)
    target_value = torch.tensor([item.value for item in examples], dtype=torch.float32)
    value_abs_total = float(target_value.abs().sum().item())
    value_nonzero_examples = int((target_value.abs() > 1e-9).sum().item())

    generator = torch.Generator()
    generator.manual_seed(seed or 0)
    started = time.perf_counter()
    updates = 0
    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=generator)
        for start in range(0, len(examples), batch_size):
            batch_ids = order[start : start + batch_size]
            obs_batch = observations[batch_ids].to(selected_device, non_blocking=True)
            policy_batch = target_policy[batch_ids].to(selected_device, non_blocking=True)
            value_batch = target_value[batch_ids].to(selected_device, non_blocking=True)
            policy_logits, values = model(obs_batch)
            loss = alphazero_loss(policy_logits, values, policy_batch, value_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            updates += 1
            last_loss = float(loss.detach().cpu().item())
    model.eval()
    stats = AlphaZeroTrainStats(
        examples=len(examples),
        updates=updates,
        epochs=epochs,
        batch_size=batch_size,
        device=str(selected_device),
        elapsed_seconds=time.perf_counter() - started,
        loss=last_loss,
        value_mean_abs=value_abs_total / len(examples),
        value_nonzero_examples=value_nonzero_examples,
    )
    return model, stats


def generate_alphazero_self_play_examples(
    *,
    games: int = 32,
    simulations: int = 16,
    max_turns: int = 120,
    hidden_size: int = 256,
    action_limit: int = 16,
    wall_limit: int = 8,
    temperature_turns: int = 12,
    draw_value_mode: str = "heuristic",
    draw_value_scale: float = 40.0,
    root_dirichlet_alpha: float = 0.3,
    root_noise_fraction: float = 0.25,
    mcts_batch_size: int = 8,
    inference_cache_size: int = 4096,
    seed: int | None = None,
    device: str | None = None,
    initial_checkpoint: str | Path | None = None,
    workers: int = 1,
) -> tuple[list[AlphaZeroExample], AlphaZeroSelfPlayStats]:
    """Generate AlphaZero self-play data without training the model.

    This mirrors the arena runner: independent games can be split across worker
    processes, then the caller can train on the merged sample set with a large
    GPU batch.
    """

    if games < 1:
        raise ValueError("games must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    worker_count = min(workers, games)
    started = time.perf_counter()
    assignments = _game_assignments(games, worker_count)
    base_seed = seed or 0

    if worker_count == 1:
        results = [
            _generate_self_play_worker(
                games=games,
                simulations=simulations,
                max_turns=max_turns,
                hidden_size=hidden_size,
                action_limit=action_limit,
                wall_limit=wall_limit,
                temperature_turns=temperature_turns,
                draw_value_mode=draw_value_mode,
                draw_value_scale=draw_value_scale,
                root_dirichlet_alpha=root_dirichlet_alpha,
                root_noise_fraction=root_noise_fraction,
                mcts_batch_size=mcts_batch_size,
                inference_cache_size=inference_cache_size,
                seed=base_seed,
                device=device,
                initial_checkpoint=None if initial_checkpoint is None else str(initial_checkpoint),
            )
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(
                    _generate_self_play_worker,
                    games=count,
                    simulations=simulations,
                    max_turns=max_turns,
                    hidden_size=hidden_size,
                    action_limit=action_limit,
                    wall_limit=wall_limit,
                    temperature_turns=temperature_turns,
                    draw_value_mode=draw_value_mode,
                    draw_value_scale=draw_value_scale,
                    root_dirichlet_alpha=root_dirichlet_alpha,
                    root_noise_fraction=root_noise_fraction,
                    mcts_batch_size=mcts_batch_size,
                    inference_cache_size=inference_cache_size,
                    seed=base_seed + offset * 100_003,
                    device=device,
                    initial_checkpoint=None if initial_checkpoint is None else str(initial_checkpoint),
                )
                for offset, count in enumerate(assignments)
                if count > 0
            ]
            for future in as_completed(futures):
                results.append(future.result())

    examples: list[AlphaZeroExample] = []
    wins = [0, 0]
    draws = 0
    for worker_examples, worker_stats in results:
        examples.extend(worker_examples)
        wins[0] += worker_stats.wins[0]
        wins[1] += worker_stats.wins[1]
        draws += worker_stats.draws

    value_abs_total = sum(abs(item.value) for item in examples)
    value_nonzero = sum(1 for item in examples if abs(item.value) > 1e-9)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return examples, AlphaZeroSelfPlayStats(
        games=games,
        examples=len(examples),
        updates=0,
        wins=(wins[0], wins[1]),
        draws=draws,
        device=str(selected_device),
        elapsed_seconds=time.perf_counter() - started,
        mcts_batch_size=mcts_batch_size,
        value_mean_abs=value_abs_total / len(examples) if examples else 0.0,
        value_nonzero_examples=value_nonzero,
    )


def save_alphazero_checkpoint(
    model: AlphaZeroNet,
    path: str | Path,
    *,
    hidden_size: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "obs_dim": model.trunk[0].in_features,
        "action_size": ACTION_SIZE,
        "hidden_size": hidden_size if hidden_size is not None else model.trunk[0].out_features,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_alphazero_checkpoint(path: str | Path, device: str | torch.device | None = None) -> AlphaZeroCheckpoint:
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(Path(path), map_location=selected_device)
    obs_dim = int(payload.get("obs_dim", default_obs_dim()))
    hidden_size = int(payload.get("hidden_size", 256))
    action_size = int(payload.get("action_size", ACTION_SIZE))
    model = AlphaZeroNet(obs_dim, action_size=action_size, hidden_size=hidden_size).to(selected_device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return AlphaZeroCheckpoint(model=model, metadata=dict(payload.get("metadata", {})), device=selected_device)


def _flat_observation(state: object) -> list[float]:
    wrapper = DiscreteQuoridorEnv()
    wrapper.env.state = state  # type: ignore[assignment]
    return wrapper.flat_observation()


def _model_policy_prior(
    model: AlphaZeroNet,
    device: torch.device,
    state: object,
    actions: Sequence[Action],
) -> dict[Action, float]:
    obs = torch.tensor(_flat_observation(state), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(obs)
    return _policy_from_logits(logits.squeeze(0).detach().cpu(), actions)


def _policy_from_logits(logits: torch.Tensor, actions: Sequence[Action]) -> dict[Action, float]:
    if not actions:
        return {}
    legal_ids = [action_to_id(action) for action in actions]
    selected = logits[torch.tensor(legal_ids, dtype=torch.long)]
    probs = torch.softmax(selected, dim=0).detach().cpu().tolist()
    return {action: float(prob) for action, prob in zip(actions, probs)}


def _model_value(model: AlphaZeroNet, device: torch.device, state: object, root_player: int) -> float:
    obs = torch.tensor(_flat_observation(state), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, value = model(obs)
    current_value = float(value.squeeze(0).item())
    current_player = getattr(state, "current_player")
    return current_value if current_player == root_player else -current_value


def _draw_value_target(state: object, player: int, scale: float) -> float:
    opponent = 1 - player
    self_dist = _path_distance(state, player)
    opp_dist = _path_distance(state, opponent)
    wall_balance = getattr(state, "remaining_walls")[player] - getattr(state, "remaining_walls")[opponent]
    progress = _goal_progress(state, player) - _goal_progress(state, opponent)
    mobility = len(legal_pawn_moves(state, player)) - len(legal_pawn_moves(state, opponent))  # type: ignore[arg-type]
    tempo = 0.25 if getattr(state, "current_player") == player else -0.25
    score = 10.0 * (opp_dist - self_dist) + 0.75 * wall_balance + 0.5 * progress + 0.75 * mobility + tempo
    return tanh(score / scale)


def _path_distance(state: object, player: int) -> int:
    distance = shortest_path_length(state, player)  # type: ignore[arg-type]
    return UNREACHABLE_DISTANCE if distance is None else distance


def _goal_progress(state: object, player: int) -> int:
    row, _ = getattr(state, "pawn_positions")[player]
    board_size = getattr(state, "board_size")
    if player == 0:
        return board_size - 1 - row
    return row


def _sample_policy_action(policy: Mapping[Action, float], rng: random.Random) -> Action:
    actions = list(policy)
    weights = [max(0.0, policy[action]) for action in actions]
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(actions)
    return rng.choices(actions, weights=weights, k=1)[0]


def _train_replay_epochs(
    model: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    replay: list[AlphaZeroExample],
    batch_size: int,
    epochs: int,
    device: torch.device,
    rng: random.Random,
) -> int:
    if len(replay) < batch_size:
        return 0

    updates = 0
    model.train()
    for _ in range(epochs):
        batch = rng.sample(replay, batch_size)
        obs = torch.tensor([item.observation for item in batch], dtype=torch.float32, device=device)
        target_policy = torch.tensor([item.policy for item in batch], dtype=torch.float32, device=device)
        target_value = torch.tensor([item.value for item in batch], dtype=torch.float32, device=device)
        policy_logits, values = model(obs)
        loss = alphazero_loss(policy_logits, values, target_policy, target_value)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        updates += 1
    model.eval()
    return updates


def _generate_self_play_worker(
    *,
    games: int,
    simulations: int,
    max_turns: int,
    hidden_size: int,
    action_limit: int,
    wall_limit: int,
    temperature_turns: int,
    draw_value_mode: str,
    draw_value_scale: float,
    root_dirichlet_alpha: float,
    root_noise_fraction: float,
    mcts_batch_size: int,
    inference_cache_size: int,
    seed: int,
    device: str | None,
    initial_checkpoint: str | None,
) -> tuple[list[AlphaZeroExample], AlphaZeroSelfPlayStats]:
    if draw_value_mode not in {"zero", "heuristic"}:
        raise ValueError("draw_value_mode must be 'zero' or 'heuristic'")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if initial_checkpoint is not None:
        model = load_alphazero_checkpoint(initial_checkpoint, device=selected_device).model
    else:
        model = AlphaZeroNet(default_obs_dim(), hidden_size=hidden_size).to(selected_device)
    model.eval()

    started = time.perf_counter()
    examples: list[AlphaZeroExample] = []
    wins = [0, 0]
    draws = 0
    for _ in range(games):
        game_examples, winner = _play_self_play_game(
            model=model,
            device=selected_device,
            rng=rng,
            simulations=simulations,
            max_turns=max_turns,
            action_limit=action_limit,
            wall_limit=wall_limit,
            temperature_turns=temperature_turns,
            draw_value_mode=draw_value_mode,
            draw_value_scale=draw_value_scale,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_noise_fraction=root_noise_fraction,
            mcts_batch_size=mcts_batch_size,
            inference_cache_size=inference_cache_size,
        )
        examples.extend(game_examples)
        if winner is None:
            draws += 1
        else:
            wins[winner] += 1

    value_abs_total = sum(abs(item.value) for item in examples)
    value_nonzero = sum(1 for item in examples if abs(item.value) > 1e-9)
    stats = AlphaZeroSelfPlayStats(
        games=games,
        examples=len(examples),
        updates=0,
        wins=(wins[0], wins[1]),
        draws=draws,
        device=str(selected_device),
        elapsed_seconds=time.perf_counter() - started,
        mcts_batch_size=mcts_batch_size,
        value_mean_abs=value_abs_total / len(examples) if examples else 0.0,
        value_nonzero_examples=value_nonzero,
    )
    return examples, stats


def _play_self_play_game(
    *,
    model: AlphaZeroNet,
    device: torch.device,
    rng: random.Random,
    simulations: int,
    max_turns: int,
    action_limit: int,
    wall_limit: int,
    temperature_turns: int,
    draw_value_mode: str,
    draw_value_scale: float,
    root_dirichlet_alpha: float,
    root_noise_fraction: float,
    mcts_batch_size: int,
    inference_cache_size: int,
) -> tuple[list[AlphaZeroExample], int | None]:
    from quoridor.agents.puct import PUCTAgent

    env = QuoridorEnv()
    pending: list[tuple[list[float], list[float], int]] = []
    evaluator = AlphaZeroBatchEvaluator(model, device, cache_size=inference_cache_size)
    while not env.state.done and env.state.turn_count < max_turns:
        player = env.state.current_player
        legal_actions = env.legal_actions()
        search = PUCTAgent(
            simulations=simulations,
            action_limit=action_limit,
            wall_limit=wall_limit,
            prior_fn=lambda state, actions: evaluator.evaluate([(state, actions, state.current_player)])[0][0],
            value_fn=lambda state, root_player: evaluator.evaluate([(state, (), root_player)])[0][1],
            policy_value_batch_fn=evaluator.evaluate,
            inference_batch_size=mcts_batch_size,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_noise_fraction=root_noise_fraction,
            seed=rng.randrange(2**31),
        )
        temperature = 1.0 if env.state.turn_count < temperature_turns else 0.0
        visit_policy = search.search_policy(env.state, legal_actions, temperature=temperature)
        pending.append((_flat_observation(env.state), policy_vector(visit_policy), player))
        env.step(_sample_policy_action(visit_policy, rng))

    examples: list[AlphaZeroExample] = []
    for observation, policy, player in pending:
        value = 0.0
        if env.state.winner is not None:
            value = 1.0 if env.state.winner == player else -1.0
        elif draw_value_mode == "heuristic":
            value = _draw_value_target(env.state, player, draw_value_scale)
        examples.append(AlphaZeroExample(observation=observation, policy=policy, value=value))
    return examples, env.state.winner


def _game_assignments(games: int, workers: int) -> list[int]:
    base = games // workers
    extra = games % workers
    return [base + (1 if index < extra else 0) for index in range(workers)]
