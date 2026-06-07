"""Tiny Quoridor engine demo."""

from quoridor import MoveAction, QuoridorEnv, WallAction
from quoridor.core.ascii_board import render_ascii


def main() -> None:
    env = QuoridorEnv()
    print(render_ascii(env.state))
    print(f"Current player: {env.state.current_player}")
    print(f"Legal actions at start: {len(env.legal_actions())}")

    for action in (MoveAction((7, 4)), WallAction("H", 1, 4), MoveAction((6, 4))):
        result = env.step(action)
        print()
        print(f"Action: {action}")
        print(render_ascii(result.state))
        print(f"Current player: {result.state.current_player}")


if __name__ == "__main__":
    main()
