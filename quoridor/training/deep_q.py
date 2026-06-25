"""GPU-capable Deep Q-learning for Quoridor."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, NamedTuple

import torch
from torch import nn

from quoridor.agents.heuristics import evaluate_state
from quoridor.core.actions import WallAction
from quoridor.training.discrete_env import ACTION_SIZE, DiscreteQuoridorEnv


class DQN(nn.Module):
    def __init__(self, obs_dim: int, action_size: int = ACTION_SIZE, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transition(NamedTuple):
    obs: list[float]
    action_id: int
    reward: float
    next_obs: list[float]
    next_mask: list[int]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.items: Deque[Transition] = deque(maxlen=capacity)

    def append(self, transition: Transition) -> None:
        self.items.append(transition)

    def sample(self, batch_size: int, rng: random.Random) -> list[Transition]:
        return rng.sample(list(self.items), batch_size)

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True, slots=True)
class DeepQStats:
    episodes: int
    wins: tuple[int, int]
    draws: int
    updates: int
    final_epsilon: float
    device: str
    elapsed_seconds: float = 0.0


def train_deep_q(
    *,
    episodes: int = 100,
    max_turns: int = 120,
    hidden_size: int = 256,
    batch_size: int = 128,
    replay_capacity: int = 20_000,
    warmup_steps: int = 256,
    gamma: float = 0.95,
    lr: float = 1e-3,
    epsilon: float = 0.6,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.05,
    target_update_interval: int = 200,
    terminal_reward: float = 1.0,
    shaping_scale: float = 0.05,
    wall_step_penalty: float = 0.02,
    seed: int | None = None,
    device: str | None = None,
) -> tuple[DQN, DeepQStats]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if replay_capacity < batch_size:
        raise ValueError("replay_capacity must be at least batch_size")

    rng = random.Random(seed)
    torch.manual_seed(seed or 0)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    probe_env = DiscreteQuoridorEnv()
    probe_env.reset()
    obs_dim = len(probe_env.flat_observation())
    policy = DQN(obs_dim, ACTION_SIZE, hidden_size=hidden_size).to(selected_device)
    target = DQN(obs_dim, ACTION_SIZE, hidden_size=hidden_size).to(selected_device)
    target.load_state_dict(policy.state_dict())
    target.eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, amsgrad=True)
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(replay_capacity)

    wins = [0, 0]
    draws = 0
    updates = 0
    step_count = 0

    for _ in range(episodes):
        env = DiscreteQuoridorEnv()
        env.reset()
        while not env.state.done and env.state.turn_count < max_turns:
            player = env.state.current_player
            obs = env.flat_observation()
            legal_ids = env.legal_action_ids()
            action_id = _select_action(policy, obs, legal_ids, epsilon, selected_device, rng)
            action = env.info_action(action_id) if hasattr(env, "info_action") else None
            before_score = evaluate_state(env.state, player)
            result = env.step(action_id)
            reward = terminal_reward * result.reward[player]
            if not result.done:
                reward += shaping_scale * _clip((evaluate_state(env.state, player) - before_score) / 100.0, -1.0, 1.0)
            if action is None:
                action = result.info.get("action")
            if isinstance(action, WallAction):
                reward -= wall_step_penalty

            replay.append(
                Transition(
                    obs=obs,
                    action_id=action_id,
                    reward=float(reward),
                    next_obs=env.flat_observation(),
                    next_mask=env.legal_action_mask(),
                    done=result.done or env.state.turn_count >= max_turns,
                )
            )
            step_count += 1

            if len(replay) >= max(batch_size, warmup_steps):
                updates += _optimize_batch(policy, target, optimizer, loss_fn, replay, batch_size, gamma, selected_device, rng)
            if step_count % target_update_interval == 0:
                target.load_state_dict(policy.state_dict())

        if env.state.winner is None:
            draws += 1
        else:
            wins[env.state.winner] += 1
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return policy, DeepQStats(
        episodes=episodes,
        wins=(wins[0], wins[1]),
        draws=draws,
        updates=updates,
        final_epsilon=epsilon,
        device=str(selected_device),
    )


def save_deep_q_checkpoint(
    model: DQN,
    path: str | Path,
    stats: DeepQStats,
    *,
    hidden_size: int = 256,
    metadata: dict[str, object] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obs_dim = model.net[0].in_features
    payload = {
        "model_state": model.state_dict(),
        "obs_dim": obs_dim,
        "action_size": ACTION_SIZE,
        "hidden_size": hidden_size,
        "metadata": {
            "episodes": stats.episodes,
            "wins": list(stats.wins),
            "draws": stats.draws,
            "updates": stats.updates,
            "final_epsilon": stats.final_epsilon,
            "device": stats.device,
            **(metadata or {}),
        },
    }
    torch.save(payload, path)


def _select_action(
    policy: DQN,
    obs: list[float],
    legal_ids: list[int],
    epsilon: float,
    device: torch.device,
    rng: random.Random,
) -> int:
    if rng.random() < epsilon:
        return rng.choice(legal_ids)
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy(obs_tensor).squeeze(0)
        mask = torch.full((ACTION_SIZE,), -torch.inf, dtype=torch.float32, device=device)
        mask[torch.tensor(legal_ids, dtype=torch.long, device=device)] = 0.0
        return int(torch.argmax(q_values + mask).item())


def _optimize_batch(
    policy: DQN,
    target: DQN,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    rng: random.Random,
) -> int:
    batch = replay.sample(batch_size, rng)
    obs = torch.tensor([item.obs for item in batch], dtype=torch.float32, device=device)
    actions = torch.tensor([[item.action_id] for item in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=device)
    next_obs = torch.tensor([item.next_obs for item in batch], dtype=torch.float32, device=device)
    next_masks = torch.tensor([item.next_mask for item in batch], dtype=torch.bool, device=device)
    done = torch.tensor([item.done for item in batch], dtype=torch.bool, device=device)

    predicted = policy(obs).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q = target(next_obs)
        next_q = next_q.masked_fill(~next_masks, -torch.inf)
        next_best = next_q.max(dim=1).values
        next_best = torch.where(done, torch.zeros_like(next_best), next_best)
        expected = rewards - gamma * next_best

    loss = loss_fn(predicted, expected)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy.parameters(), 100.0)
    optimizer.step()
    return 1


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
