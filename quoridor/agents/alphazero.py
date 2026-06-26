"""AlphaZero-style PUCT agent wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch

from quoridor.agents.puct import PUCTAgent
from quoridor.core.actions import Action
from quoridor.core.state import QuoridorState
from quoridor.training.alphazero import AlphaZeroCheckpoint, load_alphazero_checkpoint
from quoridor.training.discrete_env import DiscreteQuoridorEnv, action_to_id


class AlphaZeroAgent:
    """PUCT search backed by a policy/value network when a checkpoint exists."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = Path("experiments/results/alphazero_policy_value.pt"),
        simulations: int = 32,
        action_limit: int = 16,
        wall_limit: int = 8,
        c_puct: float = 1.5,
        device: str | None = None,
        allow_heuristic_fallback: bool = False,
        seed: int | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint: AlphaZeroCheckpoint | None = None
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            self.checkpoint = load_alphazero_checkpoint(self.checkpoint_path, device=device)
        elif not allow_heuristic_fallback:
            path_label = "None" if self.checkpoint_path is None else str(self.checkpoint_path)
            raise FileNotFoundError(f"AlphaZero checkpoint is not available: {path_label}")

        self.search = PUCTAgent(
            simulations=simulations,
            c_puct=c_puct,
            action_limit=action_limit,
            wall_limit=wall_limit,
            prior_fn=self._policy_prior if self.checkpoint is not None else None,
            value_fn=self._value_estimate if self.checkpoint is not None else None,
            seed=seed,
        )

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        return self.search.choose_action(state, legal_actions)

    def _policy_prior(self, state: QuoridorState, actions: Sequence[Action]) -> Mapping[Action, float]:
        if self.checkpoint is None:
            return {}

        legal_ids = [action_to_id(action) for action in actions]
        obs = self._obs_tensor(state)
        with torch.no_grad():
            logits, _ = self.checkpoint.model(obs)
            masked_logits = logits.squeeze(0)[torch.tensor(legal_ids, dtype=torch.long, device=self.checkpoint.device)]
            probs = torch.softmax(masked_logits, dim=0).detach().cpu().tolist()
        return {action: float(prob) for action, prob in zip(actions, probs)}

    def _value_estimate(self, state: QuoridorState, root_player: int) -> float:
        if self.checkpoint is None:
            return 0.0

        obs = self._obs_tensor(state)
        with torch.no_grad():
            _, value = self.checkpoint.model(obs)
        current_player_value = float(value.squeeze(0).item())
        if state.current_player == root_player:
            return current_player_value
        return -current_player_value

    def _obs_tensor(self, state: QuoridorState) -> torch.Tensor:
        if self.checkpoint is None:
            raise RuntimeError("AlphaZero checkpoint is not loaded")

        wrapper = DiscreteQuoridorEnv()
        wrapper.env.state = state
        obs = torch.tensor(wrapper.flat_observation(), dtype=torch.float32, device=self.checkpoint.device)
        if obs.numel() != self.checkpoint.model.trunk[0].in_features:
            raise ValueError("checkpoint observation dimension does not match DiscreteQuoridorEnv")
        return obs.unsqueeze(0)
