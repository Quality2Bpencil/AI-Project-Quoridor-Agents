"""Example agent interfaces for the Quoridor engine."""

from .adversarial import ArgmaxQTrapAgent, CounterfactualTrapAgent, DepthTrapAgent, PathLureAgent, RolloutPoisonAgent
from .alphazero import AlphaZeroAgent
from .approx_q import ApproxQLearningAgent
from .base import Agent
from .deep_q import DeepQAgent
from .greedy_bfs import GreedyBFSAgent
from .mcts import MCTSAgent
from .minimax import MinimaxAgent
from .puct import PUCTAgent
from .q_learning import QLearningAgent
from .random_agent import RandomAgent

__all__ = [
    "Agent",
    "AlphaZeroAgent",
    "ApproxQLearningAgent",
    "ArgmaxQTrapAgent",
    "CounterfactualTrapAgent",
    "DeepQAgent",
    "DepthTrapAgent",
    "GreedyBFSAgent",
    "MCTSAgent",
    "MinimaxAgent",
    "PathLureAgent",
    "PUCTAgent",
    "QLearningAgent",
    "RandomAgent",
    "RolloutPoisonAgent",
]
