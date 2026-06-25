"""Example agent interfaces for the Quoridor engine."""

from .adversarial import CounterfactualTrapAgent, DepthTrapAgent, PathLureAgent, RolloutPoisonAgent
from .base import Agent
from .greedy_bfs import GreedyBFSAgent
from .mcts import MCTSAgent
from .minimax import MinimaxAgent
from .random_agent import RandomAgent

__all__ = [
    "Agent",
    "CounterfactualTrapAgent",
    "DepthTrapAgent",
    "GreedyBFSAgent",
    "MCTSAgent",
    "MinimaxAgent",
    "PathLureAgent",
    "RandomAgent",
    "RolloutPoisonAgent",
]
