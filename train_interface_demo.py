"""Small example showing how a model would use the training interface."""

from __future__ import annotations

import random

from quoridor.training import DiscreteQuoridorEnv


def main() -> None:
    rng = random.Random(0)
    env = DiscreteQuoridorEnv()
    obs = env.reset()

    max_turns = 1000
    while not obs["done"] and env.state.turn_count < max_turns:
        legal_ids = [idx for idx, is_legal in enumerate(obs["legal_action_mask"]) if is_legal]
        action_id = rng.choice(legal_ids)
        result = env.step(action_id)
        obs = result.observation

    if obs["winner"] is None:
        print("demo reached max_turns without a winner")
    else:
        print(f"winner: player {obs['winner']}")


if __name__ == "__main__":
    main()
