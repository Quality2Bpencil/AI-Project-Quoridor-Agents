"""Standard-library web server for visualizing Quoridor agent games."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from quoridor import MoveAction, QuoridorEnv, WallAction
from quoridor.agents import (
    DepthTrapAgent,
    GreedyBFSAgent,
    MCTSAgent,
    MinimaxAgent,
    PathLureAgent,
    RandomAgent,
    RolloutPoisonAgent,
)
from quoridor.agents.heuristics import path_distance, path_diversity
from quoridor.core.actions import Action
from quoridor.core.rules import adjacent_reachable
from quoridor.core.state import QuoridorState

STATIC_DIR = Path(__file__).with_name("static")

AgentFactory = Callable[[int], object]


def _factory(factory: Callable[..., object], **kwargs: Any) -> AgentFactory:
    return lambda player: factory(seed=player, **kwargs)


AGENT_FACTORIES: dict[str, AgentFactory | None] = {
    "Human": None,
    "Random": _factory(RandomAgent),
    "Greedy BFS": _factory(GreedyBFSAgent, action_limit=20, wall_limit=12),
    "Minimax d1": _factory(MinimaxAgent, depth=1, action_limit=10, wall_limit=6),
    "MCTS 8": _factory(MCTSAgent, iterations=8, rollout_depth=5, action_limit=8, wall_limit=4),
    "PathLure": _factory(PathLureAgent, action_limit=8, wall_limit=4, victim_action_limit=8),
    "DepthTrap": _factory(DepthTrapAgent, action_limit=8, wall_limit=4, victim_action_limit=6, followup_limit=6),
    "RolloutPoison": _factory(
        RolloutPoisonAgent,
        action_limit=6,
        wall_limit=3,
        victim_action_limit=4,
        rollout_depth=2,
    ),
}


class WebGameSession:
    def __init__(self) -> None:
        self.env = QuoridorEnv()
        self.player_types = ["Human", "Random"]
        self.agents = [self._make_agent(0), self._make_agent(1)]
        self.history: deque[str] = deque(maxlen=80)
        self.last_action: str | None = None
        self.wall_owners: dict[tuple[str, int, int], int] = {}

    def reset(self) -> dict[str, Any]:
        self.env.reset()
        self.agents = [self._make_agent(0), self._make_agent(1)]
        self.history.clear()
        self.last_action = None
        self.wall_owners.clear()
        return self.state_payload()

    def set_players(self, players: list[str]) -> dict[str, Any]:
        if len(players) != 2:
            raise ValueError("players must contain exactly two entries")
        for player_type in players:
            if player_type not in AGENT_FACTORIES:
                raise ValueError(f"unknown player type: {player_type}")
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
        action = agent.choose_action(self.env.state, self.env.legal_actions())  # type: ignore[attr-defined]
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
        return None if factory is None else factory(player)

    def _step(self, action: Action) -> None:
        player = self.env.state.current_player
        if isinstance(action, WallAction):
            self.wall_owners[(action.orientation, action.row, action.col)] = player
        self.env.step(action)
        label = _format_action(player, action)
        self.last_action = label
        self.history.append(label)


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
            self._send_json({"agents": list(AGENT_FACTORIES.keys())})
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
