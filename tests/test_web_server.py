import unittest

from quoridor.web.server import AGENT_FACTORIES, WebGameSession


class WebServerTests(unittest.TestCase):
    def test_state_payload_contains_visualizer_fields(self):
        session = WebGameSession()

        payload = session.state_payload()

        self.assertEqual(payload["boardSize"], 9)
        self.assertIn("Human", payload["agentOptions"])
        self.assertIn("PathLure", payload["agentOptions"])
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


if __name__ == "__main__":
    unittest.main()
