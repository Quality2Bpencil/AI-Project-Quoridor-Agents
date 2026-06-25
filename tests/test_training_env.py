import unittest

from quoridor import MoveAction, WallAction
from quoridor.training.discrete_env import ACTION_SIZE, DiscreteQuoridorEnv, action_to_id, id_to_action


class TrainingEnvTests(unittest.TestCase):
    def test_action_encoding_round_trip(self):
        actions = [
            MoveAction((0, 0)),
            MoveAction((8, 8)),
            WallAction("H", 0, 0),
            WallAction("H", 7, 7),
            WallAction("V", 0, 0),
            WallAction("V", 7, 7),
        ]

        for action in actions:
            self.assertEqual(id_to_action(action_to_id(action)), action)

    def test_action_encoding_rejects_out_of_bounds_actions(self):
        actions = [
            MoveAction((-1, 8)),
            MoveAction((9, 0)),
            MoveAction((0, 9)),
            WallAction("H", -1, 0),
            WallAction("H", 8, 0),
            WallAction("V", 0, 8),
        ]

        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    action_to_id(action)

    def test_id_to_action_rejects_non_integer_ids(self):
        for action_id in (1.5, "1", True):
            with self.subTest(action_id=action_id):
                with self.assertRaises(TypeError):
                    id_to_action(action_id)

    def test_id_to_action_rejects_out_of_range_ids(self):
        for action_id in (-1, ACTION_SIZE):
            with self.subTest(action_id=action_id):
                with self.assertRaises(ValueError):
                    id_to_action(action_id)

    def test_initial_legal_mask_matches_engine_actions(self):
        env = DiscreteQuoridorEnv()
        obs = env.reset()

        self.assertEqual(len(obs["legal_action_mask"]), ACTION_SIZE)
        self.assertEqual(sum(obs["legal_action_mask"]), len(env.env.legal_actions()))

    def test_step_with_legal_discrete_action(self):
        env = DiscreteQuoridorEnv()
        obs = env.reset()
        action_id = next(idx for idx, is_legal in enumerate(obs["legal_action_mask"]) if is_legal)

        result = env.step(action_id)

        self.assertFalse(result.info["invalid_action"])
        self.assertEqual(env.state.turn_count, 1)

    def test_invalid_action_can_raise(self):
        env = DiscreteQuoridorEnv()
        env.reset()

        with self.assertRaises(ValueError):
            env.step(action_to_id(MoveAction((0, 0))))

    def test_invalid_action_can_return_penalty(self):
        env = DiscreteQuoridorEnv(invalid_action_penalty=-1)
        env.reset()

        result = env.step(action_to_id(MoveAction((0, 0))))

        self.assertTrue(result.info["invalid_action"])
        self.assertEqual(result.reward, (-1, 0))
        self.assertEqual(env.state.turn_count, 0)

    def test_flat_observation_size(self):
        env = DiscreteQuoridorEnv()
        env.reset()

        self.assertEqual(len(env.flat_observation()), 2 * 81 + 2 * 64 + 2 + 1)


if __name__ == "__main__":
    unittest.main()
