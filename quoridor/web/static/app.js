const state = {
  data: null,
  mode: "move",
  playing: false,
  busy: false,
};

const els = {
  board: document.getElementById("boardGrid"),
  axisTop: document.getElementById("axisTop"),
  axisLeft: document.getElementById("axisLeft"),
  player0: document.getElementById("player0"),
  player1: document.getElementById("player1"),
  status: document.getElementById("matchStatus"),
  turn: document.getElementById("turnMetric"),
  current: document.getElementById("currentMetric"),
  p0Path: document.getElementById("p0Path"),
  p1Path: document.getElementById("p1Path"),
  p0Diversity: document.getElementById("p0Diversity"),
  p1Diversity: document.getElementById("p1Diversity"),
  p0Walls: document.getElementById("p0Walls"),
  p1Walls: document.getElementById("p1Walls"),
  lastAction: document.getElementById("lastAction"),
  actionLog: document.getElementById("actionLog"),
  timeline: document.getElementById("timelineTrack"),
  step: document.getElementById("stepButton"),
  play: document.getElementById("playButton"),
  reset: document.getElementById("resetButton"),
  speed: document.getElementById("speedSlider"),
};

const hasGsap = () => window.gsap && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

async function api(path, body = null) {
  const options = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const res = await fetch(path, options);
  const payload = await res.json();
  if (!res.ok) throw new Error(payload.error || "request failed");
  return payload;
}

function buildBoard() {
  const tracks = Array.from({ length: 17 }, (_, index) => (index % 2 === 0 ? "1fr" : "12px")).join(" ");
  els.board.style.gridTemplateColumns = tracks;
  els.board.style.gridTemplateRows = tracks;
  els.axisTop.innerHTML = "ABCDEFGHI".split("").map((label) => `<span>${label}</span>`).join("");
  els.axisLeft.innerHTML = Array.from({ length: 9 }, (_, index) => `<span>${index}</span>`).join("");

  for (let row = 0; row < 17; row += 1) {
    for (let col = 0; col < 17; col += 1) {
      const node = document.createElement("button");
      node.type = "button";
      node.style.gridRow = String(row + 1);
      node.style.gridColumn = String(col + 1);
      if (row % 2 === 0 && col % 2 === 0) {
        node.className = "cell";
        node.dataset.row = String(row / 2);
        node.dataset.col = String(col / 2);
        node.setAttribute("aria-label", `cell ${row / 2}, ${col / 2}`);
        node.addEventListener("click", () => onCellClick(row / 2, col / 2));
      } else if (row % 2 === 1 && col % 2 === 1) {
        node.className = "intersection";
        node.tabIndex = -1;
      } else {
        node.className = "wall-slot";
        if (row % 2 === 1) {
          node.dataset.orientation = "H";
          node.dataset.row = String((row - 1) / 2);
          node.dataset.col = String(col / 2);
          node.style.gridColumn = `${col + 1} / span 3`;
          node.addEventListener("click", () => onWallClick("H", (row - 1) / 2, col / 2));
        } else {
          node.dataset.orientation = "V";
          node.dataset.row = String(row / 2);
          node.dataset.col = String((col - 1) / 2);
          node.style.gridRow = `${row + 1} / span 3`;
          node.addEventListener("click", () => onWallClick("V", row / 2, (col - 1) / 2));
        }
      }
      els.board.appendChild(node);
    }
  }

  for (const player of [0, 1]) {
    const pawn = document.createElement("div");
    pawn.className = `pawn p${player}`;
    pawn.dataset.player = String(player);
    els.board.appendChild(pawn);
  }
}

function render(data) {
  const previous = state.data;
  state.data = data;

  syncSelects(data);
  clearBoardClasses();
  renderPaths(data);
  renderLegalHints(data);
  renderWalls(data, previous);
  renderPawns(data, previous);
  renderConsole(data);
}

function syncSelects(data) {
  for (const select of [els.player0, els.player1]) {
    if (select.options.length === data.agentOptions.length) continue;
    select.innerHTML = data.agentOptions.map((name) => `<option value="${name}">${name}</option>`).join("");
  }
  els.player0.value = data.playerTypes[0];
  els.player1.value = data.playerTypes[1];
}

function clearBoardClasses() {
  els.board.querySelectorAll(".cell").forEach((node) => {
    node.classList.remove("legal", "path-p0", "path-p1");
    node.querySelector(".pawn")?.remove();
  });
  els.board.querySelectorAll(".wall-slot").forEach((node) => {
    node.classList.remove("legal", "placed");
  });
}

function renderPaths(data) {
  data.paths.forEach((path, player) => {
    path.forEach(([row, col]) => {
      cellAt(row, col)?.classList.add(`path-p${player}`);
    });
  });
}

function renderLegalHints(data) {
  if (data.done || data.playerTypes[data.currentPlayer] !== "Human") return;
  if (state.mode === "move") {
    data.legalMoves.forEach(([row, col]) => cellAt(row, col)?.classList.add("legal"));
    return;
  }
  data.legalWalls
    .filter((wall) => wall.orientation === state.mode)
    .forEach((wall) => wallAt(wall.orientation, wall.row, wall.col)?.classList.add("legal"));
}

function renderWalls(data, previous) {
  const oldWalls = new Set((previous?.walls || []).map(wallKey));
  data.walls.forEach((wall) => {
    const slot = wallAt(wall.orientation, wall.row, wall.col);
    if (!slot) return;
    slot.classList.add("placed");
    if (!oldWalls.has(wallKey(wall)) && hasGsap()) {
      gsap.fromTo(slot, { scale: 0.45, autoAlpha: 0 }, { scale: 1, autoAlpha: 1, duration: 0.24, ease: "back.out(1.8)" });
    }
  });
}

function renderPawns(data, previous) {
  data.pawns.forEach(([row, col], player) => {
    const cell = cellAt(row, col);
    const pawn = els.board.querySelector(`.pawn.p${player}`);
    if (!cell || !pawn) return;
    cell.appendChild(pawn);
    const old = previous?.pawns?.[player];
    if (hasGsap() && (!old || old[0] !== row || old[1] !== col)) {
      gsap.fromTo(pawn, { scale: 0.72, y: -8 }, { scale: 1, y: 0, duration: 0.26, ease: "back.out(1.9)" });
    }
  });
}

function renderConsole(data) {
  els.turn.textContent = data.turnCount;
  els.current.textContent = data.done ? "Done" : `P${data.currentPlayer}`;
  els.p0Path.textContent = data.pathLengths[0];
  els.p1Path.textContent = data.pathLengths[1];
  els.p0Diversity.textContent = data.pathDiversity[0];
  els.p1Diversity.textContent = data.pathDiversity[1];
  els.p0Walls.textContent = data.remainingWalls[0];
  els.p1Walls.textContent = data.remainingWalls[1];
  els.status.textContent = data.done
    ? `Player ${data.winner} wins`
    : `P${data.currentPlayer}: ${data.playerTypes[data.currentPlayer]}`;
  els.lastAction.textContent = data.lastAction || "No moves yet";
  els.actionLog.innerHTML = data.history
    .slice()
    .reverse()
    .map((item) => `<li>${item}</li>`)
    .join("");
  els.timeline.innerHTML = data.history
    .slice(-36)
    .map((item) => `<span class="tick ${item.startsWith("P0") ? "p0" : "p1"}" title="${item}"></span>`)
    .join("");
}

function cellAt(row, col) {
  return els.board.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
}

function wallAt(orientation, row, col) {
  return els.board.querySelector(
    `.wall-slot[data-orientation="${orientation}"][data-row="${row}"][data-col="${col}"]`,
  );
}

function wallKey(wall) {
  return `${wall.orientation}:${wall.row}:${wall.col}`;
}

async function onCellClick(row, col) {
  if (!canHumanAct("move", row, col)) return;
  await submitHumanAction({ type: "move", row, col });
}

async function onWallClick(orientation, row, col) {
  if (!canHumanAct(orientation, row, col)) return;
  await submitHumanAction({ type: "wall", orientation, row, col });
}

function canHumanAct(mode, row, col) {
  const data = state.data;
  if (!data || data.done || data.playerTypes[data.currentPlayer] !== "Human") return false;
  if (state.mode !== mode) return false;
  if (mode === "move") {
    return data.legalMoves.some(([r, c]) => r === row && c === col);
  }
  return data.legalWalls.some((wall) => wall.orientation === mode && wall.row === row && wall.col === col);
}

async function submitHumanAction(action) {
  await guarded(async () => {
    render(await api("/api/human-action", action));
  });
  await maybeAutoStep(false);
}

async function stepAgent() {
  await guarded(async () => {
    render(await api("/api/agent-step", {}));
  });
}

async function maybeAutoStep(chain = state.playing) {
  if ((chain && !state.playing) || !state.data || state.data.done) return;
  if (state.data.playerTypes[state.data.currentPlayer] === "Human") return;
  window.setTimeout(async () => {
    await stepAgent();
    if (chain && state.playing) await maybeAutoStep(true);
  }, Number(els.speed.value));
}

async function guarded(fn) {
  if (state.busy) return;
  state.busy = true;
  try {
    await fn();
  } catch (error) {
    els.status.textContent = error.message;
  } finally {
    state.busy = false;
  }
}

function bindControls() {
  els.player0.addEventListener("change", updatePlayers);
  els.player1.addEventListener("change", updatePlayers);
  els.step.addEventListener("click", stepAgent);
  els.reset.addEventListener("click", async () => {
    state.playing = false;
    setPlayButton();
    render(await api("/api/reset", {}));
  });
  els.play.addEventListener("click", async () => {
    state.playing = !state.playing;
    setPlayButton();
    await maybeAutoStep(true);
  });
  document.querySelectorAll(".mode").forEach((button) => {
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      document.querySelectorAll(".mode").forEach((node) => node.classList.toggle("active", node === button));
      render(state.data);
    });
  });
}

async function updatePlayers() {
  await guarded(async () => {
    render(await api("/api/config", { players: [els.player0.value, els.player1.value] }));
  });
  await maybeAutoStep(true);
}

function setPlayButton() {
  els.play.innerHTML = state.playing
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h4v14H7z" /><path d="M13 5h4v14h-4z" /></svg>Pause'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7-11-7Z" /></svg>Play';
}

async function init() {
  buildBoard();
  bindControls();
  render(await api("/api/state"));
  if (hasGsap()) {
    gsap.from(".topbar, .console-section, .board-frame, .timeline", {
      y: 14,
      autoAlpha: 0,
      duration: 0.55,
      stagger: 0.04,
      ease: "power2.out",
    });
  }
}

init();
