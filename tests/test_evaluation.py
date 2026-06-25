import tempfile
import unittest
from pathlib import Path

from experiments.run_ablation import summarize_rows
from experiments.summarize_trap_effectiveness import summarize_trap_effectiveness
from quoridor.agents import GreedyBFSAgent, RandomAgent
from quoridor.evaluation import (
    AgentSpec,
    build_round_robin_tasks,
    matchup_matrix_rows,
    play_arena_task,
    play_game,
    read_arena_rows,
    run_arena,
    run_round_robin,
    update_elo,
    write_matchup_matrix_csv,
    write_score_matrix_csv,
)
from quoridor.evaluation.metrics import _is_trap_condition, _update_trap_metrics


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

    def test_arena_tasks_rotate_first_player(self):
        specs = [
            AgentSpec("random", lambda: RandomAgent(seed=0)),
            AgentSpec("greedy", lambda: GreedyBFSAgent(seed=1)),
        ]

        tasks = build_round_robin_tasks(specs, games_per_pair=2, max_turns=20)

        self.assertEqual(len(tasks), 2)
        self.assertEqual((tasks[0].agent0, tasks[0].agent1), ("random", "greedy"))
        self.assertEqual((tasks[1].agent0, tasks[1].agent1), ("greedy", "random"))
        self.assertNotEqual(tasks[0].game_id, tasks[1].game_id)

    def test_resumable_arena_skips_completed_games_and_writes_matrix(self):
        specs = [
            AgentSpec("random", lambda: RandomAgent(seed=0)),
            AgentSpec("greedy", lambda: GreedyBFSAgent(seed=1)),
        ]
        specs_by_name = {spec.name: spec for spec in specs}
        tasks = build_round_robin_tasks(specs, games_per_pair=2, max_turns=6)

        with tempfile.TemporaryDirectory() as tmpdir:
            games_path = Path(tmpdir) / "arena_games.csv"
            matrix_path = Path(tmpdir) / "matchup_matrix.csv"
            score_path = Path(tmpdir) / "score_matrix.csv"

            first_rows = run_arena(
                tasks,
                lambda task: play_arena_task(task, specs_by_name),
                output=games_path,
                workers=1,
            )
            second_rows = run_arena(
                tasks,
                lambda task: play_arena_task(task, specs_by_name),
                output=games_path,
                workers=1,
                resume=True,
            )
            rows = read_arena_rows(games_path)
            matrix = matchup_matrix_rows(rows)
            write_matchup_matrix_csv(rows, matrix_path)
            write_score_matrix_csv(rows, score_path)

            self.assertEqual(len(first_rows), 2)
            self.assertEqual(len(second_rows), 0)
            self.assertEqual(len(rows), 2)
            self.assertTrue(matrix_path.exists())
            self.assertTrue(score_path.exists())
            self.assertEqual({row["agent"] for row in matrix}, {"random", "greedy"})
            self.assertIn("score_rate", matrix_path.read_text(encoding="utf-8"))

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

    def test_trap_effectiveness_summary_uses_trap_perspective(self):
        rows = [
            {
                "agent0": "greedy_bfs",
                "agent1": "path_lure",
                "winner": "1",
                "turns": "12",
                "trap_events_0": "0",
                "trap_events_1": "2",
                "wall_actions_0": "0",
                "wall_actions_1": "3",
                "final_path_delta_0": "4",
                "final_path_delta_1": "0",
                "min_diversity_0": "1",
                "min_diversity_1": "2",
                "status": "ok",
            }
        ]

        summary = summarize_trap_effectiveness(rows, {"path_lure": "greedy_bfs"})

        self.assertEqual(summary[0]["games"], "1")
        self.assertEqual(summary[0]["wins"], "1")
        self.assertEqual(summary[0]["score_rate"], "1.0000")
        self.assertEqual(summary[0]["avg_trap_events"], "2.000")
        self.assertEqual(summary[0]["avg_target_path_delta"], "4.000")
        self.assertEqual(summary[0]["avg_target_min_diversity"], "1.000")

    def test_trap_condition_requires_low_diversity_and_path_increase(self):
        initial_paths = (8, 8)

        self.assertFalse(_is_trap_condition((8, 8), (2, 1), initial_paths, 1))
        self.assertFalse(_is_trap_condition((8, 10), (2, 2), initial_paths, 1))
        self.assertTrue(_is_trap_condition((8, 10), (2, 1), initial_paths, 1))

    def test_trap_events_only_count_new_trap_transitions(self):
        min_diversity = [2, 2]
        trap_counts = [0, 0]
        initial_paths = (8, 8)

        _update_trap_metrics(
            acting_player=0,
            initial_paths=initial_paths,
            previous_paths=(8, 8),
            previous_diversity=(2, 2),
            current_paths=(8, 10),
            current_diversity=(2, 1),
            min_diversity=min_diversity,
            trap_counts=trap_counts,
        )
        self.assertEqual(trap_counts, [1, 0])
        self.assertEqual(min_diversity, [2, 1])

        _update_trap_metrics(
            acting_player=0,
            initial_paths=initial_paths,
            previous_paths=(8, 10),
            previous_diversity=(2, 1),
            current_paths=(8, 11),
            current_diversity=(2, 1),
            min_diversity=min_diversity,
            trap_counts=trap_counts,
        )
        self.assertEqual(trap_counts, [1, 0])


if __name__ == "__main__":
    unittest.main()
