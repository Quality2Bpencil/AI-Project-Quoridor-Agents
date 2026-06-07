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
