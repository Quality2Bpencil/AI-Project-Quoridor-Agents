import tempfile
import unittest
from pathlib import Path

from quoridor.agents import GreedyBFSAgent, RandomAgent
from quoridor.evaluation import AgentSpec, play_game, run_round_robin, update_elo


class EvaluationTests(unittest.TestCase):
    def test_play_game_returns_record(self):
        record = play_game(
            RandomAgent(seed=0),
            GreedyBFSAgent(seed=1),
            agent0_name="random",
            agent1_name="greedy",
            max_turns=20,
        )

        self.assertEqual(record.agent0, "random")
        self.assertEqual(record.agent1, "greedy")
        self.assertLessEqual(record.turns, 20)
        self.assertIn(record.winner, {0, 1, None})

    def test_update_elo_moves_winner_up(self):
        winner, loser = update_elo(1000, 1000, 1.0)

        self.assertGreater(winner, 1000)
        self.assertLess(loser, 1000)

    def test_round_robin_and_csv_export(self):
        result = run_round_robin(
            [
                AgentSpec("random", lambda: RandomAgent(seed=0)),
                AgentSpec("greedy", lambda: GreedyBFSAgent(seed=1)),
            ],
            games_per_pair=2,
            max_turns=20,
        )

        standings = result.standings()
        self.assertEqual(len(result.records), 2)
        self.assertEqual({row["agent"] for row in standings}, {"random", "greedy"})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "games.csv"
            result.write_games_csv(path)
            self.assertTrue(path.exists())
            self.assertIn("winner_name", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
