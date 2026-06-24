import tempfile
import unittest
from pathlib import Path

from experiments.run_ablation import summarize_rows
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
        self.assertEqual(len(record.final_path_lengths), 2)
        self.assertEqual(len(record.min_path_diversity), 2)
        self.assertEqual(sum(record.move_actions) + sum(record.wall_actions), record.turns)

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
            text = path.read_text(encoding="utf-8")
            self.assertIn("winner_name", text)
            self.assertIn("trap_events_0", text)
            self.assertIn("final_path_delta_1", text)

    def test_ablation_summary_rows(self):
        rows = [
            {
                "condition": "path_lure_weight_0",
                "adversary_won": True,
                "trap_events": 2,
                "opponent_min_diversity": 1,
                "opponent_path_delta": 3,
            },
            {
                "condition": "path_lure_weight_0",
                "adversary_won": False,
                "trap_events": 0,
                "opponent_min_diversity": 2,
                "opponent_path_delta": 1,
            },
        ]

        summary = summarize_rows(rows)

        self.assertEqual(summary[0]["condition"], "path_lure_weight_0")
        self.assertEqual(summary[0]["games"], 2)
        self.assertEqual(summary[0]["adversary_win_rate"], 0.5)
        self.assertEqual(summary[0]["avg_trap_events"], 1.0)


if __name__ == "__main__":
    unittest.main()
