import tempfile
import unittest
from pathlib import Path

from quoridor import MoveAction, QuoridorEnv
from quoridor.agents import QLearningAgent
from quoridor.agents.q_learning import load_q_table, save_q_table, state_key
from quoridor.training.discrete_env import action_to_id
from quoridor.training.q_learning import train_q_learning


class QLearningTests(unittest.TestCase):
    def test_q_learning_agent_returns_legal_action_without_table(self):
        env = QuoridorEnv()
        agent = QLearningAgent(seed=0)

        action = agent.choose_action(env.state, env.legal_actions())

        self.assertIn(action, env.legal_actions())

    def test_q_learning_agent_prefers_highest_known_q_value(self):
        env = QuoridorEnv()
        preferred = MoveAction((8, 3))
        key = state_key(env.state)
        table = {key: {action_to_id(preferred): 5.0}}
        agent = QLearningAgent(q_table=table, heuristic_margin=1_000.0, seed=0)

        action = agent.choose_action(env.state, env.legal_actions())

        self.assertEqual(action, preferred)

    def test_q_table_save_and_load_round_trip(self):
        env = QuoridorEnv()
        key = state_key(env.state)
        table = {key: {action_to_id(MoveAction((7, 4))): 1.25}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.json"
            save_q_table(table, path, {"episodes": 1})
            loaded = load_q_table(path)

        self.assertEqual(loaded, table)

    def test_q_learning_training_smoke(self):
        q_table, stats = train_q_learning(episodes=2, max_turns=6, epsilon=0.1, seed=0)

        self.assertEqual(stats.episodes, 2)
        self.assertGreater(stats.q_states, 0)
        self.assertGreater(sum(len(row) for row in q_table.values()), 0)


if __name__ == "__main__":
    unittest.main()
