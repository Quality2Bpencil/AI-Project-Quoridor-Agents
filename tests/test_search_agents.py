import unittest

from quoridor import QuoridorEnv
from quoridor.agents import (
    ArgmaxQTrapAgent,
    CounterfactualTrapAgent,
    DepthTrapAgent,
    GreedyBFSAgent,
    MCTSAgent,
    MinimaxAgent,
    PathLureAgent,
    RolloutPoisonAgent,
)
from quoridor.agents.heuristics import choose_best, evaluate_state_terms, path_distance, pawn_race_winner
from quoridor.core.actions import MoveAction, WallAction
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

    def test_mcts_expands_best_ranked_action_first_with_tiny_budget(self):
        env = QuoridorEnv()
        agent = MCTSAgent(iterations=1, rollout_depth=1, action_limit=6, wall_limit=3, seed=0)

        action = agent.choose_action(env.state, env.legal_actions())

        self.assertIsInstance(action, WallAction)

    def test_mcts_prefers_immediate_winning_move_with_tiny_budget(self):
        state = QuoridorState(pawn_positions=((1, 4), (8, 4)), current_player=0)
        env = QuoridorEnv()
        env.state = state
        agent = MCTSAgent(iterations=1, rollout_depth=1, action_limit=6, wall_limit=3, seed=0)

        action = agent.choose_action(state, env.legal_actions())

        self.assertEqual(action, MoveAction((0, 4)))

    def test_mcts_avoids_obvious_opening_side_step(self):
        env = QuoridorEnv()
        agent = MCTSAgent(iterations=4, rollout_depth=3, action_limit=6, wall_limit=3, seed=3)

        action = agent.choose_action(env.state, env.legal_actions())

        self.assertNotIn(action, [MoveAction((8, 3)), MoveAction((8, 5))])

    def test_opening_heuristic_blocks_losing_pawn_race(self):
        env = QuoridorEnv()

        self.assertEqual(pawn_race_winner(env.state), 1)
        self.assertIsInstance(choose_best(env.legal_actions(), env.state, 0), WallAction)

    def test_heuristic_terms_are_explainable(self):
        env = QuoridorEnv()

        terms = evaluate_state_terms(env.state, 0)

        self.assertIn("path_distance", terms)
        self.assertIn("path_diversity", terms)
        self.assertIn("pawn_mobility", terms)
        self.assertIn("pawn_race", terms)

    def test_path_lure_returns_legal_action(self):
        self.assert_agent_returns_legal_action(
            PathLureAgent(seed=0, action_limit=6, wall_limit=3, victim_action_limit=6)
        )

    def test_depth_trap_returns_legal_action(self):
        self.assert_agent_returns_legal_action(
            DepthTrapAgent(seed=0, action_limit=6, wall_limit=3, victim_action_limit=4, followup_limit=4)
        )

    def test_rollout_poison_returns_legal_action(self):
        self.assert_agent_returns_legal_action(
            RolloutPoisonAgent(seed=0, action_limit=4, wall_limit=2, victim_action_limit=3, rollout_depth=1)
        )

    def test_counterfactual_trap_returns_legal_action(self):
        self.assert_agent_returns_legal_action(
            CounterfactualTrapAgent(
                seed=0,
                action_limit=4,
                wall_limit=2,
                victim_action_limit=3,
                response_width=2,
                followup_limit=3,
            )
        )

    def test_argmax_q_trap_returns_legal_action(self):
        self.assert_agent_returns_legal_action(
            ArgmaxQTrapAgent(
                seed=0,
                action_limit=4,
                wall_limit=2,
                victim_action_limit=3,
                response_width=2,
                followup_limit=3,
            )
        )

    def test_path_distance_allows_zero_at_goal(self):
        state = QuoridorState(pawn_positions=((0, 4), (8, 4)), winner=0)

        self.assertEqual(path_distance(state, 0), 0)


if __name__ == "__main__":
    unittest.main()
