"""Standard-library web server for visualizing Quoridor agent games."""

from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import threading
from collections import Counter, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from quoridor import MoveAction, QuoridorEnv, WallAction
from quoridor.agents import (
    AlphaZeroAgent,
    ApproxQLearningAgent,
    ArgmaxQTrapAgent,
    CounterfactualTrapAgent,
    DeepQAgent,
    DepthTrapAgent,
    GreedyBFSAgent,
    MCTSAgent,
    MinimaxAgent,
    PathLureAgent,
    PUCTAgent,
    QLearningAgent,
    RandomAgent,
    RolloutPoisonAgent,
)
from quoridor.agents.heuristics import path_distance, path_diversity, ranked_actions
from quoridor.core.actions import Action
from quoridor.core.rules import adjacent_reachable, apply_action
from quoridor.core.state import QuoridorState

STATIC_DIR = Path(__file__).with_name("static")

AgentFactory = Callable[[int, int], object]
ALPHAZERO_CHECKPOINT = Path("experiments/results/alphazero_policy_value.pt")


def _factory(factory: Callable[..., object], **kwargs: Any) -> AgentFactory:
    return lambda player, seed: factory(seed=seed, **kwargs)


AGENT_FACTORIES: dict[str, AgentFactory | None] = {
    "Human": None,
    "Random": _factory(RandomAgent),
    "Greedy BFS": _factory(GreedyBFSAgent, action_limit=20, wall_limit=12),
    "Minimax d1": _factory(MinimaxAgent, depth=1, action_limit=10, wall_limit=6),
    "MCTS 16": _factory(MCTSAgent, iterations=16, rollout_depth=5, action_limit=8, wall_limit=4),
    "PUCT 16": _factory(PUCTAgent, simulations=16, action_limit=8, wall_limit=4),
    "AlphaZero": _factory(
        AlphaZeroAgent,
        checkpoint_path=ALPHAZERO_CHECKPOINT,
        simulations=16,
        action_limit=8,
        wall_limit=4,
    ),
    "Q-Learning": _factory(QLearningAgent, table_path=Path("experiments/results/q_learning_policy.json")),
    "Approx-Q": _factory(ApproxQLearningAgent, weights_path=Path("experiments/results/approx_q_policy.json")),
    "Deep-Q": _factory(DeepQAgent, checkpoint_path=Path("experiments/results/deep_q_policy.pt")),
    "PathLure": _factory(PathLureAgent, action_limit=8, wall_limit=4, victim_action_limit=8),
    "DepthTrap": _factory(DepthTrapAgent, action_limit=6, wall_limit=3, victim_action_limit=4, followup_limit=4),
    "RolloutPoison": _factory(
        RolloutPoisonAgent,
        action_limit=6,
        wall_limit=3,
        victim_action_limit=4,
        rollout_depth=2,
    ),
    "CounterTrap": _factory(
        CounterfactualTrapAgent,
        action_limit=6,
        wall_limit=3,
        victim_action_limit=4,
        response_width=2,
        followup_limit=4,
    ),
    "ArgmaxQTrap": _factory(
        ArgmaxQTrapAgent,
        action_limit=4,
        wall_limit=2,
        victim_action_limit=3,
        response_width=1,
        followup_limit=3,
    ),
}

AGENT_REQUIRED_FILES: dict[str, Path] = {
    "Q-Learning": Path("experiments/results/q_learning_policy.json"),
    "Approx-Q": Path("experiments/results/approx_q_policy.json"),
    "Deep-Q": Path("experiments/results/deep_q_policy.pt"),
    "AlphaZero": ALPHAZERO_CHECKPOINT,
}


def agent_status() -> dict[str, dict[str, object]]:
    status: dict[str, dict[str, object]] = {}
    for name in AGENT_FACTORIES:
        required = AGENT_REQUIRED_FILES.get(name)
        enabled = required is None or required.exists()
        status[name] = {
            "enabled": enabled,
            "reason": "" if enabled else f"missing {required}",
        }
    return status


def agent_enabled(name: str) -> bool:
    return bool(agent_status()[name]["enabled"])


class WebGameSession:
    def __init__(self) -> None:
        self.env = QuoridorEnv()
        self.player_types = ["Human", "Random"]
        self.seed_nonce = 0
        self.agents = [self._make_agent(0), self._make_agent(1)]
        self.history: deque[str] = deque(maxlen=80)
        self.last_action: str | None = None
        self.wall_owners: dict[tuple[str, int, int], int] = {}
        self.repetition_counts: Counter[tuple[object, ...]] = Counter()
        self._mark_repetition_state(self.env.state)

    def reset(self) -> dict[str, Any]:
        self.env.reset()
        self.agents = [self._make_agent(0), self._make_agent(1)]
        self.history.clear()
        self.last_action = None
        self.wall_owners.clear()
        self.repetition_counts.clear()
        self._mark_repetition_state(self.env.state)
        return self.state_payload()

    def set_players(self, players: list[str]) -> dict[str, Any]:
        if len(players) != 2:
            raise ValueError("players must contain exactly two entries")
        for player_type in players:
            if player_type not in AGENT_FACTORIES:
                raise ValueError(f"unknown player type: {player_type}")
            if not agent_enabled(player_type):
                reason = agent_status()[player_type]["reason"]
                raise ValueError(f"agent unavailable: {player_type} ({reason})")
        self.player_types = list(players)
        self.agents = [self._make_agent(0), self._make_agent(1)]
        return self.state_payload()

    def apply_human_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.env.state.done:
            raise ValueError("game is already over")
        player = self.env.state.current_player
        if self.player_types[player] != "Human":
            raise ValueError("current player is controlled by an agent")
        action = _action_from_payload(payload)
        self._step(action)
        return self.state_payload()

    def step_agent(self) -> dict[str, Any]:
        if self.env.state.done:
            return self.state_payload()
        player = self.env.state.current_player
        if self.player_types[player] == "Human":
            raise ValueError("current player is Human")
        agent = self.agents[player]
        if agent is None:
            raise ValueError("missing agent instance")
        legal_actions = self.env.legal_actions()
        action = agent.choose_action(self.env.state, legal_actions)  # type: ignore[attr-defined]
        action = self._avoid_repetition(action, legal_actions)
        self._step(action)
        return self.state_payload()

    def state_payload(self) -> dict[str, Any]:
        state = self.env.state
        legal_actions = self.env.legal_actions()
        legal_moves = [action.target for action in legal_actions if isinstance(action, MoveAction)]
        legal_walls = [
            {"orientation": action.orientation, "row": action.row, "col": action.col}
            for action in legal_actions
            if isinstance(action, WallAction)
        ]
        return {
            "boardSize": state.board_size,
            "currentPlayer": state.current_player,
            "winner": state.winner,
            "done": state.done,
            "turnCount": state.turn_count,
            "pawns": [list(pos) for pos in state.pawn_positions],
            "walls": [
                {"orientation": orientation, "row": row, "col": col,
                 "owner": self.wall_owners.get((orientation, row, col), 0)}
                for orientation, row, col in sorted(state.walls)
            ],
            "remainingWalls": list(state.remaining_walls),
            "playerTypes": list(self.player_types),
            "agentOptions": list(AGENT_FACTORIES.keys()),
            "agentStatus": agent_status(),
            "legalMoves": [list(pos) for pos in legal_moves],
            "legalWalls": legal_walls,
            "pathLengths": [path_distance(state, 0), path_distance(state, 1)],
            "pathDiversity": [path_diversity(state, 0), path_diversity(state, 1)],
            "paths": [_shortest_path(state, 0), _shortest_path(state, 1)],
            "lastAction": self.last_action,
            "history": list(self.history)[-60:],
        }

    def _make_agent(self, player: int) -> object | None:
        factory = AGENT_FACTORIES[self.player_types[player]]
        if not agent_enabled(self.player_types[player]):
            raise ValueError(f"agent unavailable: {self.player_types[player]}")
        return None if factory is None else factory(player, self._next_seed(player))

    def _step(self, action: Action) -> None:
        player = self.env.state.current_player
        if isinstance(action, WallAction):
            self.wall_owners[(action.orientation, action.row, action.col)] = player
        self.env.step(action)
        self._mark_repetition_state(self.env.state)
        label = _format_action(player, action)
        self.last_action = label
        self.history.append(label)

    def _next_seed(self, player: int) -> int:
        self.seed_nonce += 1
        return secrets.randbits(63) ^ (self.seed_nonce << 8) ^ player

    def _avoid_repetition(self, action: Action, legal_actions: list[Action]) -> Action:
        if self._next_repetition_count(action) == 0:
            return action

        legal_set = set(legal_actions)
        candidates = [
            candidate
            for candidate in ranked_actions(
                self.env.state,
                max_actions=32,
                wall_limit=16,
                wall_radius=3,
            )
            if candidate in legal_set
        ]
        candidates.extend(candidate for candidate in legal_actions if candidate not in candidates)
        for candidate in candidates:
            if candidate != action and self._next_repetition_count(candidate) == 0:
                return candidate
        return action

    def _next_repetition_count(self, action: Action) -> int:
        try:
            next_state = apply_action(self.env.state, action)
        except ValueError:
            return 0
        return self.repetition_counts[self._repetition_key(next_state)]

    def _mark_repetition_state(self, state: QuoridorState) -> None:
        self.repetition_counts[self._repetition_key(state)] += 1

    @staticmethod
    def _repetition_key(state: QuoridorState) -> tuple[object, ...]:
        return (
            state.current_player,
            state.pawn_positions,
            state.walls,
            state.remaining_walls,
        )


def _action_from_payload(payload: dict[str, Any]) -> Action:
    action_type = payload.get("type")
    if action_type == "move":
        return MoveAction((int(payload["row"]), int(payload["col"])))
    if action_type == "wall":
        return WallAction(str(payload["orientation"]), int(payload["row"]), int(payload["col"]))
    raise ValueError("action type must be 'move' or 'wall'")


def _format_action(player: int, action: Action) -> str:
    if isinstance(action, MoveAction):
        row, col = action.target
        return f"P{player} move ({row}, {col})"
    return f"P{player} wall {action.orientation} ({action.row}, {action.col})"


def _shortest_path(state: QuoridorState, player: int) -> list[list[int]]:
    start = state.pawn_positions[player]
    goal = state.goal_row(player)
    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while queue:
        pos = queue.popleft()
        if pos[0] == goal:
            path: list[list[int]] = []
            cur: tuple[int, int] | None = pos
            while cur is not None:
                path.append([cur[0], cur[1]])
                cur = parent[cur]
            return list(reversed(path))
        for nxt in adjacent_reachable(state, pos):
            if nxt not in parent:
                parent[nxt] = pos
                queue.append(nxt)
    return []


class QuoridorWebHandler(BaseHTTPRequestHandler):
    session = WebGameSession()
    lock = threading.RLock()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(self._with_session(lambda session: session.state_payload()))
            return
        if path == "/api/agents":
            self._send_json({"agents": list(AGENT_FACTORIES.keys()), "agentStatus": agent_status()})
            return
        if path == "/":
            path = "/index.html"
        self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/reset":
                self._send_json(self._with_session(lambda session: session.reset()))
                return
            if path == "/api/config":
                self._send_json(self._with_session(lambda session: session.set_players(payload["players"])))
                return
            if path == "/api/human-action":
                self._send_json(self._with_session(lambda session: session.apply_human_action(payload)))
                return
            if path == "/api/agent-step":
                self._send_json(self._with_session(lambda session: session.step_agent()))
                return
            self._send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
        except Exception as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _with_session(self, fn: Callable[[WebGameSession], dict[str, Any]]) -> dict[str, Any]:
        with self.lock:
            return fn(self.session)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, request_path: str) -> None:
        relative = request_path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "file not found")
            return

        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), QuoridorWebHandler)
    print(f"Quoridor web UI: http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()
