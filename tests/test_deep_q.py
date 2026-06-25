import tempfile
import unittest
from pathlib import Path

from quoridor import QuoridorEnv
from quoridor.agents.deep_q import DeepQAgent
from quoridor.training.deep_q import save_deep_q_checkpoint, train_deep_q


class DeepQTests(unittest.TestCase):
    def test_deep_q_agent_falls_back_without_checkpoint(self):
        env = QuoridorEnv()
        agent = DeepQAgent()

        action = agent.choose_action(env.state, env.legal_actions())

        self.assertIn(action, env.legal_actions())

    def test_deep_q_training_smoke_cpu(self):
        model, stats = train_deep_q(
            episodes=1,
            max_turns=4,
            batch_size=2,
            warmup_steps=2,
            replay_capacity=16,
            hidden_size=16,
            device="cpu",
            seed=0,
        )

        self.assertEqual(stats.episodes, 1)
        self.assertGreaterEqual(stats.updates, 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep_q.pt"
            save_deep_q_checkpoint(model, path, stats, hidden_size=16)
            agent = DeepQAgent(checkpoint_path=path, device="cpu")
            env = QuoridorEnv()
            action = agent.choose_action(env.state, env.legal_actions())

        self.assertIn(action, env.legal_actions())


if __name__ == "__main__":
    unittest.main()
