import unittest
from pathlib import Path
from unittest.mock import patch

from quoridor.core.actions import MoveAction
from quoridor.core.rules import apply_action
from quoridor.web.server import AGENT_FACTORIES, WebGameSession, agent_status


class WebServerTests(unittest.TestCase):
    def test_state_payload_contains_visualizer_fields(self):
        session = WebGameSession()

        payload = session.state_payload()

        self.assertEqual(payload["boardSize"], 9)
        self.assertIn("Human", payload["agentOptions"])
        self.assertIn("PathLure", payload["agentOptions"])
        self.assertIn("agentStatus", payload)
        self.assertEqual(len(payload["legalMoves"]), 3)
        self.assertGreater(len(payload["legalWalls"]), 0)
        self.assertEqual(len(payload["paths"]), 2)

    def test_can_configure_and_step_agent(self):
        session = WebGameSession()
        session.set_players(["Random", "Random"])

        payload = session.step_agent()

        self.assertEqual(payload["turnCount"], 1)
        self.assertIsNotNone(payload["lastAction"])

    def test_all_agent_names_have_factories_except_human(self):
        for name, factory in AGENT_FACTORIES.items():
            with self.subTest(name=name):
                if name == "Human":
                    self.assertIsNone(factory)
                else:
                    self.assertIsNotNone(factory)

    def test_agent_status_contains_enabled_flags(self):
        status = agent_status()

        self.assertIn("AlphaZero", status)
        self.assertIn("enabled", status["AlphaZero"])

    def test_unavailable_agent_cannot_be_configured(self):
        session = WebGameSession()
        with patch("quoridor.web.server.AGENT_REQUIRED_FILES", {"Random": Path("missing-policy.bin")}):
            with self.assertRaises(ValueError):
                session.set_players(["Random", "Human"])

    def test_agent_seeds_change_between_creations(self):
        session = WebGameSession()

        self.assertNotEqual(session._next_seed(0), session._next_seed(0))

    def test_agent_step_avoids_repeating_seen_position(self):
        session = WebGameSession()
        action = MoveAction((7, 4))
        session.repetition_counts[session._repetition_key(apply_action(session.env.state, action))] = 1

        replacement = session._avoid_repetition(action, session.env.legal_actions())

        self.assertNotEqual(replacement, action)


if __name__ == "__main__":
    unittest.main()
