import unittest

from quoridor import MoveAction, QuoridorEnv, QuoridorState, WallAction
from quoridor.core.rules import (
    has_path_to_goal,
    is_legal_action,
    legal_pawn_moves,
    legal_wall_action,
    shortest_path_length,
)


def move_targets(state):
    return {action.target for action in legal_pawn_moves(state)}


class RuleTests(unittest.TestCase):
    def test_initial_state_has_expected_actions(self):
        env = QuoridorEnv()

        self.assertEqual(env.state.pawn_positions, ((8, 4), (0, 4)))
        self.assertEqual(env.state.remaining_walls, (10, 10))
        self.assertEqual(move_targets(env.state), {(7, 4), (8, 3), (8, 5)})
        self.assertEqual(len(env.legal_actions()), 3 + 128)


    def test_simple_move_switches_turn(self):
        env = QuoridorEnv()
        result = env.step(MoveAction((7, 4)))

        self.assertEqual(result.state.pawn_positions[0], (7, 4))
        self.assertEqual(result.state.current_player, 1)
        self.assertEqual(result.reward, (0, 0))

    def test_single_action_legality_check(self):
        env = QuoridorEnv()

        self.assertTrue(is_legal_action(env.state, MoveAction((7, 4))))
        self.assertTrue(is_legal_action(env.state, WallAction("H", 1, 4)))
        self.assertFalse(is_legal_action(env.state, MoveAction((6, 4))))
        self.assertFalse(is_legal_action(env.state, WallAction("H", 8, 0)))


    def test_wall_blocks_movement_between_two_cells(self):
        state = QuoridorState(walls=frozenset({("H", 7, 4)}))

        self.assertNotIn((7, 4), move_targets(state))
        self.assertIn((8, 3), move_targets(state))
        self.assertIn((8, 5), move_targets(state))


    def test_direct_jump_when_opponent_is_adjacent_and_behind_is_open(self):
        state = QuoridorState(pawn_positions=((4, 4), (3, 4)))

        self.assertIn((2, 4), move_targets(state))
        self.assertNotIn((3, 4), move_targets(state))


    def test_side_jump_when_opponent_has_wall_behind(self):
        state = QuoridorState(
            pawn_positions=((4, 4), (3, 4)),
            walls=frozenset({("H", 2, 4)}),
        )

        self.assertGreaterEqual(move_targets(state), {(3, 3), (3, 5)})
        self.assertNotIn((2, 4), move_targets(state))


    def test_wall_placement_consumes_wall_and_switches_turn(self):
        env = QuoridorEnv()
        result = env.step(WallAction("H", 1, 4))

        self.assertIn(("H", 1, 4), result.state.walls)
        self.assertEqual(result.state.remaining_walls, (9, 10))
        self.assertEqual(result.state.current_player, 1)


    def test_overlapping_or_crossing_wall_is_illegal(self):
        state = QuoridorState(walls=frozenset({("H", 3, 3)}))

        self.assertFalse(legal_wall_action(state, WallAction("H", 3, 3)))
        self.assertFalse(legal_wall_action(state, WallAction("H", 3, 2)))
        self.assertFalse(legal_wall_action(state, WallAction("H", 3, 4)))
        self.assertFalse(legal_wall_action(state, WallAction("V", 3, 3)))


    def test_wall_cannot_remove_all_paths(self):
        walls = frozenset({("H", 0, 0), ("H", 0, 2), ("H", 0, 4), ("H", 0, 7)})
        state = QuoridorState(pawn_positions=((8, 6), (0, 6)), walls=walls)

        self.assertTrue(has_path_to_goal(state, 1))
        self.assertFalse(legal_wall_action(state, WallAction("H", 0, 5)))


    def test_shortest_path_length_initial_board(self):
        state = QuoridorEnv().state

        self.assertEqual(shortest_path_length(state, 0), 8)
        self.assertEqual(shortest_path_length(state, 1), 8)


    def test_illegal_action_raises(self):
        env = QuoridorEnv()

        with self.assertRaises(ValueError):
            env.step(MoveAction((6, 4)))


if __name__ == "__main__":
    unittest.main()
