import unittest

from quoridor import MoveAction, QuoridorEnv
from quoridor.agents import PUCTAgent
from quoridor.core.state import QuoridorState


class PUCTAgentTests(unittest.TestCase):
    def test_puct_returns_legal_action(self):
        env = QuoridorEnv()
        agent = PUCTAgent(simulations=4, action_limit=6, wall_limit=3)

        action = agent.choose_action(env.state, env.legal_actions())

        self.assertIn(action, env.legal_actions())

    def test_puct_expands_best_opening_move_with_tiny_budget(self):
        env = QuoridorEnv()
        agent = PUCTAgent(simulations=1, action_limit=6, wall_limit=3)

        action = agent.choose_action(env.state, env.legal_actions())

        self.assertEqual(action, MoveAction((7, 4)))

    def test_puct_prefers_immediate_winning_move(self):
        state = QuoridorState(pawn_positions=((1, 4), (8, 4)), current_player=0)
        env = QuoridorEnv()
        env.state = state
        agent = PUCTAgent(simulations=2, action_limit=6, wall_limit=3)

        action = agent.choose_action(state, env.legal_actions())

        self.assertEqual(action, MoveAction((0, 4)))


if __name__ == "__main__":
    unittest.main()
