"""AlphaZero-style policy/value network utilities for Quoridor."""

from __future__ import annotations

import random
import time
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
        while not env.state.done and env.state.turn_count < max_turns:
            player = env.state.current_player
            legal_actions = env.legal_actions()
            search = PUCTAgent(
                simulations=simulations,
                action_limit=action_limit,
                wall_limit=wall_limit,
                prior_fn=lambda state, actions: _model_policy_prior(model, selected_device, state, actions),
                value_fn=lambda state, root_player: _model_value(model, selected_device, state, root_player),
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
        value_mean_abs=value_abs_total / total_examples if total_examples else 0.0,
        value_nonzero_examples=value_nonzero_examples,
    )
    return model, stats


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
    legal_ids = [action_to_id(action) for action in actions]
    obs = torch.tensor(_flat_observation(state), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(obs)
        selected = logits.squeeze(0)[torch.tensor(legal_ids, dtype=torch.long, device=device)]
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
