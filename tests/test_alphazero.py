import tempfile
import unittest
from pathlib import Path

import torch

from quoridor import QuoridorEnv
from quoridor.agents import AlphaZeroAgent
from quoridor.training.alphazero import (
    AlphaZeroNet,
    alphazero_loss,
    default_obs_dim,
    policy_vector,
    save_alphazero_checkpoint,
    train_alphazero_self_play,
)


class AlphaZeroTests(unittest.TestCase):
    def test_alphazero_agent_requires_checkpoint_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.pt"

            with self.assertRaises(FileNotFoundError):
                AlphaZeroAgent(checkpoint_path=missing, simulations=2, action_limit=6, wall_limit=3, seed=0)

    def test_alphazero_agent_loads_policy_value_checkpoint(self):
        env = QuoridorEnv()
        model = AlphaZeroNet(default_obs_dim(), hidden_size=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "az.pt"
            save_alphazero_checkpoint(model, path, hidden_size=32, metadata={"test": True})
            agent = AlphaZeroAgent(checkpoint_path=path, simulations=2, action_limit=6, wall_limit=3, seed=0)

            action = agent.choose_action(env.state, env.legal_actions())

        self.assertIn(action, env.legal_actions())

    def test_alphazero_network_shapes(self):
        model = AlphaZeroNet(default_obs_dim(), hidden_size=32)
        obs = torch.zeros((2, default_obs_dim()), dtype=torch.float32)

        policy_logits, values = model(obs)

        self.assertEqual(policy_logits.shape, (2, 209))
        self.assertEqual(values.shape, (2,))

    def test_policy_vector_and_loss(self):
        env = QuoridorEnv()
        legal = env.legal_actions()
        vector = policy_vector({legal[0]: 0.25, legal[1]: 0.75})
        model = AlphaZeroNet(default_obs_dim(), hidden_size=32)
        obs = torch.zeros((1, default_obs_dim()), dtype=torch.float32)

        policy_logits, values = model(obs)
        loss = alphazero_loss(
            policy_logits,
            values,
            torch.tensor([vector], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
        )

        self.assertAlmostEqual(sum(vector), 1.0)
        self.assertGreater(float(loss.item()), 0.0)

    def test_self_play_training_smoke(self):
        model, stats = train_alphazero_self_play(
            games=1,
            simulations=1,
            max_turns=4,
            hidden_size=32,
            action_limit=4,
            wall_limit=2,
            batch_size=2,
            epochs_per_game=1,
            seed=0,
            device="cpu",
        )

        self.assertIsInstance(model, AlphaZeroNet)
        self.assertEqual(stats.games, 1)
        self.assertGreater(stats.examples, 0)

    def test_self_play_training_can_resume_from_checkpoint(self):
        model = AlphaZeroNet(default_obs_dim(), hidden_size=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "seed.pt"
            save_alphazero_checkpoint(model, checkpoint)
            resumed, stats = train_alphazero_self_play(
                games=1,
                simulations=1,
                max_turns=4,
                hidden_size=32,
                action_limit=4,
                wall_limit=2,
                batch_size=2,
                epochs_per_game=1,
                replay_capacity=4,
                seed=1,
                device="cpu",
                initial_checkpoint=checkpoint,
            )

        self.assertIsInstance(resumed, AlphaZeroNet)
        self.assertEqual(stats.games, 1)
        self.assertGreater(stats.examples, 0)

    def test_replay_capacity_must_cover_batch_size(self):
        with self.assertRaises(ValueError):
            train_alphazero_self_play(
                games=1,
                simulations=1,
                batch_size=4,
                replay_capacity=2,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
