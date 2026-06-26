"""Deep Q-Network policy agent."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from quoridor.agents.heuristics import choose_best, evaluate_action
from quoridor.core.actions import Action
from quoridor.core.state import QuoridorState
from quoridor.training.discrete_env import ACTION_SIZE, DiscreteQuoridorEnv, action_to_id


class DeepQAgent:
    """Choose legal actions from a trained PyTorch DQN checkpoint.

    Without a checkpoint the agent falls back to the existing one-ply heuristic,
    which keeps the UI and tournament harness usable before training.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        heuristic_margin: float = 5.0,
        seed: int | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.device_name = device
        self.heuristic_margin = heuristic_margin
        self.seed = seed
        self._model = None
        self._device = None
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            self._load_checkpoint(self.checkpoint_path)

    def choose_action(self, state: QuoridorState, legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("DeepQAgent received no legal actions")
        if self._model is None:
            return choose_best(legal_actions, state, state.current_player)

        import torch

        wrapper = DiscreteQuoridorEnv()
        wrapper.env.state = state
        obs = torch.tensor(wrapper.flat_observation(), dtype=torch.float32, device=self._device).unsqueeze(0)
        legal_ids = [action_to_id(action) for action in legal_actions]
        with torch.no_grad():
            q_values = self._model(obs).squeeze(0)
            mask = torch.full((ACTION_SIZE,), -torch.inf, dtype=torch.float32, device=self._device)
            mask[torch.tensor(legal_ids, dtype=torch.long, device=self._device)] = 0.0
            action_id = int(torch.argmax(q_values + mask).item())
        for action in legal_actions:
            if action_to_id(action) == action_id:
                heuristic_choice = choose_best(legal_actions, state, state.current_player)
                if (
                    evaluate_action(state, action, state.current_player)
                    < evaluate_action(state, heuristic_choice, state.current_player) - self.heuristic_margin
                ):
                    return heuristic_choice
                return action
        return choose_best(legal_actions, state, state.current_player)

    def _load_checkpoint(self, path: Path) -> None:
        import torch

        from quoridor.training.deep_q import DQN

        device = torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(path, map_location=device)
        obs_dim = int(payload.get("obs_dim", 293))
        action_size = int(payload.get("action_size", ACTION_SIZE))
        hidden_size = int(payload.get("hidden_size", 256))
        model = DQN(obs_dim, action_size, hidden_size=hidden_size).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        self._model = model
        self._device = device
