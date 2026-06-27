import tempfile
import unittest
from pathlib import Path

import torch

from quoridor import QuoridorEnv
from quoridor.agents import AlphaZeroAgent
from quoridor.training.alphazero import (
    AlphaZeroBatchEvaluator,
    AlphaZeroNet,
    alphazero_loss,
    default_obs_dim,
    generate_alphazero_self_play_examples,
    policy_vector,
    save_alphazero_checkpoint,
    train_alphazero_examples,
    train_alphazero_self_play,
)
from quoridor.training.teacher_bootstrap import generate_teacher_bootstrap_examples


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

    def test_batch_evaluator_returns_priors_and_values(self):
        env = QuoridorEnv()
        model = AlphaZeroNet(default_obs_dim(), hidden_size=32)
        evaluator = AlphaZeroBatchEvaluator(model, torch.device("cpu"), cache_size=8)
        legal = env.legal_actions()

        results = evaluator.evaluate([(env.state, legal[:4], env.state.current_player)])

        self.assertEqual(len(results), 1)
        priors, value = results[0]
        self.assertEqual(set(priors), set(legal[:4]))
        self.assertAlmostEqual(sum(priors.values()), 1.0, places=6)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

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

    def test_draw_shaping_produces_nonzero_value_targets(self):
        _, stats = train_alphazero_self_play(
            games=1,
            simulations=1,
            max_turns=1,
            hidden_size=32,
            action_limit=4,
            wall_limit=2,
            batch_size=1,
            epochs_per_game=1,
            draw_value_mode="heuristic",
            seed=2,
            device="cpu",
        )

        self.assertEqual(stats.draws, 1)
        self.assertGreater(stats.value_nonzero_examples, 0)
        self.assertGreater(stats.value_mean_abs, 0.0)

    def test_batched_example_training_smoke(self):
        env = QuoridorEnv()
        legal = env.legal_actions()
        examples = [
            self._example(env, legal[0], 0.5),
            self._example(env, legal[1], -0.5),
        ]

        model, stats = train_alphazero_examples(
            examples,
            hidden_size=32,
            batch_size=2,
            epochs=1,
            seed=0,
            device="cpu",
        )

        self.assertIsInstance(model, AlphaZeroNet)
        self.assertEqual(stats.examples, 2)
        self.assertEqual(stats.batch_size, 2)
        self.assertEqual(stats.updates, 1)

    def test_batched_self_play_generation_smoke(self):
        examples, stats = generate_alphazero_self_play_examples(
            games=1,
            simulations=2,
            max_turns=3,
            hidden_size=32,
            action_limit=4,
            wall_limit=2,
            mcts_batch_size=2,
            seed=0,
            device="cpu",
            workers=1,
        )

        self.assertGreater(len(examples), 0)
        self.assertEqual(stats.games, 1)
        self.assertEqual(stats.examples, len(examples))
        self.assertEqual(stats.mcts_batch_size, 2)

    def test_teacher_bootstrap_generation_smoke(self):
        examples, stats = generate_teacher_bootstrap_examples(
            games=1,
            max_turns=2,
            seed=0,
            workers=1,
            teacher_profile="fast",
        )

        self.assertGreater(len(examples), 0)
        self.assertEqual(stats.games, 1)
        self.assertEqual(stats.examples, len(examples))

    def _example(self, env: QuoridorEnv, action, value: float):
        from quoridor.training.alphazero import AlphaZeroExample
        from quoridor.training.discrete_env import DiscreteQuoridorEnv

        wrapper = DiscreteQuoridorEnv()
        wrapper.env.state = env.state
        return AlphaZeroExample(
            observation=wrapper.flat_observation(),
            policy=policy_vector({action: 1.0}),
            value=value,
        )


if __name__ == "__main__":
    unittest.main()
