import unittest

from quoridor import QuoridorEnv
from quoridor.agents import GreedyBFSAgent, MCTSAgent, MinimaxAgent, PathLureAgent
from quoridor.agents.heuristics import path_distance
from quoridor.core.state import QuoridorState


class SearchAgentTests(unittest.TestCase):
    def assert_agent_returns_legal_action(self, agent):
        env = QuoridorEnv()
        legal_actions = env.legal_actions()

        action = agent.choose_action(env.state, legal_actions)

        self.assertIn(action, legal_actions)
        result = env.step(action)
        self.assertEqual(result.state.turn_count, 1)

    def test_greedy_bfs_returns_legal_action(self):
        self.assert_agent_returns_legal_action(GreedyBFSAgent(seed=0))

    def test_minimax_returns_legal_action(self):
        self.assert_agent_returns_legal_action(MinimaxAgent(depth=1, action_limit=8, wall_limit=4, seed=0))

    def test_mcts_returns_legal_action(self):
        self.assert_agent_returns_legal_action(
            MCTSAgent(iterations=4, rollout_depth=3, action_limit=6, wall_limit=3, seed=0)
        )

    def test_path_lure_returns_legal_action(self):
        self.assert_agent_returns_legal_action(
            PathLureAgent(seed=0, action_limit=6, wall_limit=3, victim_action_limit=6)
        )

    def test_path_distance_allows_zero_at_goal(self):
        state = QuoridorState(pawn_positions=((0, 4), (8, 4)), winner=0)

        self.assertEqual(path_distance(state, 0), 0)


if __name__ == "__main__":
    unittest.main()
