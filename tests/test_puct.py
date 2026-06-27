import unittest

from quoridor import MoveAction, QuoridorEnv
from quoridor.agents import PUCTAgent
from quoridor.core.actions import WallAction
from quoridor.core.state import QuoridorState


class PUCTAgentTests(unittest.TestCase):
    def test_puct_returns_legal_action(self):
        env = QuoridorEnv()
        agent = PUCTAgent(simulations=4, action_limit=6, wall_limit=3)

        action = agent.choose_action(env.state, env.legal_actions())

        self.assertIn(action, env.legal_actions())

    def test_puct_blocks_losing_opening_race_with_tiny_budget(self):
        env = QuoridorEnv()
        agent = PUCTAgent(simulations=1, action_limit=6, wall_limit=3)

        action = agent.choose_action(env.state, env.legal_actions())

        self.assertIsInstance(action, WallAction)

    def test_puct_prefers_immediate_winning_move(self):
        state = QuoridorState(pawn_positions=((1, 4), (8, 4)), current_player=0)
        env = QuoridorEnv()
        env.state = state
        agent = PUCTAgent(simulations=2, action_limit=6, wall_limit=3)

        action = agent.choose_action(state, env.legal_actions())

        self.assertEqual(action, MoveAction((0, 4)))

    def test_puct_search_policy_is_probability_distribution(self):
        env = QuoridorEnv()
        agent = PUCTAgent(simulations=2, action_limit=6, wall_limit=3)

        policy = agent.search_policy(env.state, env.legal_actions())

        self.assertEqual(set(policy), set(env.legal_actions()))
        self.assertAlmostEqual(sum(policy.values()), 1.0)

    def test_puct_can_batch_leaf_evaluations(self):
        env = QuoridorEnv()
        batch_sizes = []

        def evaluate(requests):
            batch_sizes.append(len(requests))
            output = []
            for _, actions, _ in requests:
                prior = 1.0 / len(actions)
                output.append(({action: prior for action in actions}, 0.0))
            return output

        agent = PUCTAgent(
            simulations=4,
            action_limit=6,
            wall_limit=3,
            tactical_shortcut_margin=999.0,
            policy_value_batch_fn=evaluate,
            inference_batch_size=2,
        )

        policy = agent.search_policy(env.state, env.legal_actions())

        self.assertAlmostEqual(sum(policy.values()), 1.0)
        self.assertTrue(any(size > 1 for size in batch_sizes))


if __name__ == "__main__":
    unittest.main()
