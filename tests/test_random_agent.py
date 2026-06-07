import unittest

from quoridor import QuoridorEnv
from quoridor.agents import RandomAgent
from quoridor.core.actions import MoveAction, WallAction


class RandomAgentTests(unittest.TestCase):
    def test_random_agent_returns_legal_action(self):
        env = QuoridorEnv()
        agent = RandomAgent(seed=1)
        legal_actions = env.legal_actions()

        action = agent.choose_action(env.state, legal_actions)

        self.assertIn(action, legal_actions)
        self.assertIsInstance(action, (MoveAction, WallAction))

    def test_random_agent_can_step_environment(self):
        env = QuoridorEnv()
        agents = (RandomAgent(seed=2), RandomAgent(seed=3))

        for _ in range(5):
            player = env.state.current_player
            action = agents[player].choose_action(env.state, env.legal_actions())
            result = env.step(action)
            self.assertEqual(result.state, env.state)

        self.assertEqual(env.state.turn_count, 5)

    def test_two_random_agents_can_play_some_turns(self):
        env = QuoridorEnv()
        agents = (RandomAgent(seed=4), RandomAgent(seed=5))

        for _ in range(10):
            if env.state.done:
                break
            player = env.state.current_player
            action = agents[player].choose_action(env.state, env.legal_actions())
            env.step(action)

        self.assertLessEqual(env.state.turn_count, 10)
        self.assertIn(env.state.winner, {0, 1, None})


if __name__ == "__main__":
    unittest.main()
