import tempfile
import unittest
from pathlib import Path

from quoridor import MoveAction, QuoridorEnv
from quoridor.agents import ApproxQLearningAgent
from quoridor.agents.approx_q import load_weights, q_value, save_weights
from quoridor.training.approx_q_learning import train_approx_q_learning


class ApproxQLearningTests(unittest.TestCase):
    def test_agent_returns_legal_action_without_weights(self):
        env = QuoridorEnv()
        agent = ApproxQLearningAgent(seed=0)

        action = agent.choose_action(env.state, env.legal_actions())

        self.assertIn(action, env.legal_actions())

    def test_agent_uses_weighted_features(self):
        env = QuoridorEnv()
        agent = ApproxQLearningAgent(weights={"move_forward": 5.0, "move_sideways": -1.0}, seed=0)

        action = agent.choose_action(env.state, env.legal_actions())

        self.assertEqual(action, MoveAction((7, 4)))
        self.assertGreater(q_value(agent.weights, env.state, action), 0.0)

    def test_weights_save_and_load_round_trip(self):
        weights = {"bias": 1.0, "move_forward": 2.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            save_weights(weights, path, {"episodes": 1})
            loaded = load_weights(path)

        self.assertEqual(loaded, weights)

    def test_training_smoke_updates_weights(self):
        weights, stats = train_approx_q_learning(episodes=2, max_turns=6, epsilon=0.1, seed=0)

        self.assertEqual(stats.episodes, 2)
        self.assertGreater(stats.nonzero_weights, 0)
        self.assertTrue(any(abs(value) > 0 for value in weights.values()))


if __name__ == "__main__":
    unittest.main()
