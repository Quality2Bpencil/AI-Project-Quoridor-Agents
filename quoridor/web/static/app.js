import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";

/* ═══════════════════════════════════════════════════
   Constants
   ═══════════════════════════════════════════════════ */
const N = 9, TILE = 0.95, GAP = 0.12, SP = TILE + GAP, HALF = 4;
const FRAME_PAD = 0.55, BOARD_SPAN = N * SP, FRAME_SZ = BOARD_SPAN + FRAME_PAD * 2;
const TILE_H = 0.14, FRAME_H = 0.38, WALL_H = 0.55;
const WALL_LEN = 2 * TILE + GAP, WALL_THICK = 0.13, PAWN_SCALE = 0.1;

const C = {
  bg: 0x122033,
  frameDark: 0x6f3b1d, frameRim: 0xa96728, groove: 0x2d2015,
  tile: 0xdcc596, tileDark: 0x948064, joint: 0x44382b,
  wallP0: 0xd88a22, wallP1: 0xd88a22,
  pawnP0: 0xf0d49a, pawnP1: 0x242934,
  legal: 0x58c2ab, pathP0: 0xf2c868, pathP1: 0x737f96,
};

function g2w(r, c) { return [(c - HALF) * SP, (r - HALF) * SP]; }

/* ═══════════════════════════════════════════════════
   Three.js globals
   ═══════════════════════════════════════════════════ */
let scene, camera, renderer, composer, controls;
let boardGroup, wallGroup, hintGroup, tileMeshes = [], wallMeshes = {};
let pawnMeshes = [null, null], pawnGeo = null;
let legalOvl = [], pathOvl = [], wallGhosts = [], currentState = null;
let hoverTarget = null;

/* ═══════════════════════════════════════════════════
   Procedural wood texture
   ═══════════════════════════════════════════════════ */
function woodTex(base, grain, w = 256, h = 256) {
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const cx = cv.getContext("2d");
  cx.fillStyle = base; cx.fillRect(0, 0, w, h);
  cx.globalAlpha = 0.22; cx.strokeStyle = grain;
  for (let y = 0; y < h; y += 2 + Math.random() * 5) {
    cx.lineWidth = 0.5 + Math.random() * 0.8; cx.beginPath(); cx.moveTo(0, y);
    let x = 0; while (x < w) { x += 8 + Math.random() * 18; cx.lineTo(x, y + (Math.random() - .5) * 3); }
    cx.stroke();
  }
  cx.globalAlpha = 1;
  const id = cx.getImageData(0, 0, w, h);
  for (let i = 0; i < id.data.length; i += 4) {
    const n = (Math.random() - .5) * 10;
    id.data[i] = Math.min(255, Math.max(0, id.data[i] + n));
    id.data[i + 1] = Math.min(255, Math.max(0, id.data[i + 1] + n));
    id.data[i + 2] = Math.min(255, Math.max(0, id.data[i + 2] + n));
  }
  cx.putImageData(id, 0, 0);
  const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t;
}

/* ═══════════════════════════════════════════════════
   Post-processing: pixel blocks, quantized color, screen-space edges
   ═══════════════════════════════════════════════════ */
const PixelBoardShader = {
  uniforms: {
    tDiffuse: { value: null },
    resolution: { value: new THREE.Vector2() },
    pixelSize: { value: 1.65 },
    colorSteps: { value: 16.0 },
    outlineStrength: { value: 0.22 },
  },
  vertexShader: `varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform vec2 resolution;
    uniform float pixelSize;
    uniform float colorSteps;
    uniform float outlineStrength;
    varying vec2 vUv;
    float lum(vec3 c){return dot(c,vec3(0.299,0.587,0.114));}
    void main(){
      vec2 dxy=pixelSize/resolution;
      vec2 c=dxy*floor(vUv/dxy)+dxy*0.5;
      vec3 col=texture2D(tDiffuse,c).rgb;
      col=pow(max(col,0.0),vec3(1.0/2.2));
      float l=lum(col);
      float e=0.0;
      e=max(e,abs(l-lum(pow(max(texture2D(tDiffuse,c+vec2(dxy.x,0.0)).rgb,0.0),vec3(1.0/2.2)))));
      e=max(e,abs(l-lum(pow(max(texture2D(tDiffuse,c-vec2(dxy.x,0.0)).rgb,0.0),vec3(1.0/2.2)))));
      e=max(e,abs(l-lum(pow(max(texture2D(tDiffuse,c+vec2(0.0,dxy.y)).rgb,0.0),vec3(1.0/2.2)))));
      e=max(e,abs(l-lum(pow(max(texture2D(tDiffuse,c-vec2(0.0,dxy.y)).rgb,0.0),vec3(1.0/2.2)))));
      float edge=smoothstep(0.14,0.34,e)*outlineStrength;
      col=floor(col*colorSteps+0.5)/colorSteps;
      col=clamp(col*1.025,0.0,1.0);
      float d=distance(vUv,vec2(0.5));
      col*=smoothstep(1.05,0.38,d);
      col=mix(col,vec3(0.025,0.025,0.03),edge);
      gl_FragColor=vec4(col,1.0);
    }`,
};

/* ═══════════════════════════════════════════════════
   Scene init
   ═══════════════════════════════════════════════════ */
function initScene(container) {
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(C.bg);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(C.bg);

  camera = new THREE.PerspectiveCamera(32, container.clientWidth / container.clientHeight, 0.1, 100);
  camera.position.set(0, 13.8, 9.2);
  camera.lookAt(0, 0, 0);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minPolarAngle = Math.PI * 0.1;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.minDistance = 8;
  controls.maxDistance = 28;
  controls.enablePan = false;

  scene.add(new THREE.AmbientLight(0xfff1d8, 0.64));
  scene.add(new THREE.HemisphereLight(0xffedcf, 0x101a2a, 0.35));

  const sun = new THREE.DirectionalLight(0xffedd0, 1.62);
  sun.position.set(-5, 11, 5.5);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const sc = sun.shadow.camera;
  sc.left = -7; sc.right = 7; sc.top = 7; sc.bottom = -7; sc.near = 1; sc.far = 30;
  sun.shadow.bias = -0.002; sun.shadow.normalBias = 0.02;
  scene.add(sun);

  const fill = new THREE.DirectionalLight(0x7099c0, 0.28);
  fill.position.set(5, 6, -4);
  scene.add(fill);

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const pixelPass = new ShaderPass(PixelBoardShader);
  pixelPass.uniforms.resolution.value.set(container.clientWidth, container.clientHeight);
  composer.addPass(pixelPass);

  buildBoard();
  loadPawn();
  setupInteraction(container);

  window.addEventListener("resize", () => {
    const w = container.clientWidth, h = container.clientHeight;
    camera.aspect = w / h; camera.updateProjectionMatrix();
    renderer.setSize(w, h); composer.setSize(w, h);
    pixelPass.uniforms.resolution.value.set(w, h);
  });

  animate();
}

/* ═══════════════════════════════════════════════════
   Board
   ═══════════════════════════════════════════════════ */
function buildBoard() {
  boardGroup = new THREE.Group(); scene.add(boardGroup);
  wallGroup = new THREE.Group(); scene.add(wallGroup);
  hintGroup = new THREE.Group(); scene.add(hintGroup);

  const frameMat = new THREE.MeshStandardMaterial({ color: C.frameDark, roughness: 0.82, metalness: 0.02, flatShading: true });
  const frame = new THREE.Mesh(new THREE.BoxGeometry(FRAME_SZ, FRAME_H, FRAME_SZ), frameMat);
  frame.position.y = -FRAME_H / 2; frame.receiveShadow = true; boardGroup.add(frame);

  const rimMat = new THREE.MeshStandardMaterial({ color: C.frameRim, roughness: 0.62, metalness: 0.04, flatShading: true });
  const rim = new THREE.Mesh(new THREE.BoxGeometry(FRAME_SZ + 0.1, 0.06, FRAME_SZ + 0.1), rimMat);
  rim.position.y = 0.01; rim.receiveShadow = true; boardGroup.add(rim);

  const grooveMat = new THREE.MeshStandardMaterial({ color: 0x4a3824, roughness: 0.96, flatShading: true });
  const groove = new THREE.Mesh(new THREE.BoxGeometry(BOARD_SPAN + 0.06, 0.02, BOARD_SPAN + 0.06), grooveMat);
  groove.position.y = 0.011; groove.receiveShadow = true; boardGroup.add(groove);

  const tileGeo = new THREE.BoxGeometry(TILE, TILE_H, TILE);
  const wt = woodTex("#dcc596", "#a89168");
  tileMeshes = [];
  for (let r = 0; r < N; r++) {
    tileMeshes[r] = [];
    for (let c = 0; c < N; c++) {
      const mat = new THREE.MeshStandardMaterial({ map: wt.clone(), color: 0xffffff, roughness: 0.9, metalness: 0.0, flatShading: true });
      const m = new THREE.Mesh(tileGeo, mat);
      const [x, z] = g2w(r, c);
      m.position.set(x, TILE_H / 2 + 0.02, z);
      m.receiveShadow = true;
      m.userData = { row: r, col: c };
      boardGroup.add(m);
      tileMeshes[r][c] = m;
    }
  }

  const jointGeo = new THREE.BoxGeometry(0.13, 0.16, 0.13);
  const jointMat = new THREE.MeshStandardMaterial({ color: C.joint, roughness: 0.96, metalness: 0.0, flatShading: true });
  for (let r = 0; r < N - 1; r++) {
    for (let c = 0; c < N - 1; c++) {
      const [x, z] = g2w(r + 0.5, c + 0.5);
      const joint = new THREE.Mesh(jointGeo, jointMat);
      joint.position.set(x, TILE_H + 0.08, z);
      joint.castShadow = true;
      joint.receiveShadow = true;
      boardGroup.add(joint);
    }
  }

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 40),
    new THREE.MeshStandardMaterial({ color: 0x111f33, roughness: 0.98 }),
  );
  ground.rotation.x = -Math.PI / 2; ground.position.y = -FRAME_H - 0.01;
  ground.receiveShadow = true; scene.add(ground);
}

/* ═══════════════════════════════════════════════════
   Pawn
   ═══════════════════════════════════════════════════ */
function loadPawn() {
  new OBJLoader().load("/models/pawn/pawn.obj", (obj) => {
    obj.traverse((child) => {
      if (child.isMesh) {
        const geo = child.geometry.clone();
        geo.computeBoundingBox();
        const bb = geo.boundingBox;
        geo.translate(-(bb.min.x + bb.max.x) / 2, -(bb.min.y + bb.max.y) / 2, -bb.min.z);
        geo.rotateX(-Math.PI / 2);
        geo.scale(PAWN_SCALE, PAWN_SCALE, PAWN_SCALE);
        geo.computeVertexNormals();
        pawnGeo = geo;
      }
    });
    if (pawnMeshes[0]) { pawnMeshes[0].geometry.dispose(); pawnMeshes[0].geometry = pawnGeo; }
    if (pawnMeshes[1]) { pawnMeshes[1].geometry.dispose(); pawnMeshes[1].geometry = pawnGeo; }
    if (currentState) syncPawns(currentState);
  });
}

function ensurePawn(p) {
  if (pawnMeshes[p]) return pawnMeshes[p];
  const geo = pawnGeo || new THREE.SphereGeometry(0.28, 16, 12);
  const mat = new THREE.MeshStandardMaterial({
    color: p === 0 ? C.pawnP0 : C.pawnP1,
    roughness: p === 0 ? 0.62 : 0.35,
    metalness: p === 0 ? 0.0 : 0.18,
    emissive: p === 1 ? 0x0a0d14 : 0x000000,
    flatShading: true,
  });
  const m = new THREE.Mesh(geo, mat);
  m.castShadow = true; m.receiveShadow = true;
  scene.add(m); pawnMeshes[p] = m; return m;
}

/* ═══════════════════════════════════════════════════
   Walls
   ═══════════════════════════════════════════════════ */
const wallGeo = new THREE.BoxGeometry(WALL_LEN, WALL_H, WALL_THICK);
const wTexP0 = woodTex("#d88a22", "#f0b94e");
const wTexP1 = woodTex("#d88a22", "#9c5518");

function wKey(w) { return `${w.orientation}:${w.row}:${w.col}`; }

function syncWalls(data) {
  const cur = new Set(data.walls.map(wKey));
  Object.keys(wallMeshes).forEach((k) => {
    if (!cur.has(k)) { wallGroup.remove(wallMeshes[k]); delete wallMeshes[k]; }
  });
  data.walls.forEach((w) => {
    const k = wKey(w);
    if (wallMeshes[k]) return;
    const tex = w.owner === 0 ? wTexP0 : wTexP1;
    const color = w.owner === 0 ? C.wallP0 : C.wallP1;
    const mat = new THREE.MeshStandardMaterial({ map: tex.clone(), color, roughness: 0.72, metalness: 0.02, flatShading: true });
    const m = new THREE.Mesh(wallGeo, mat);
    const [x, z] = g2w(w.row + .5, w.col + .5);
    m.position.set(x, TILE_H + 0.02 + WALL_H / 2, z);
    if (w.orientation === "V") m.rotation.y = Math.PI / 2;
    m.castShadow = true; m.receiveShadow = true;
    m.scale.y = 0; wallGroup.add(m); wallMeshes[k] = m;
    animWall(m);
  });
}

function animWall(m) {
  const t0 = performance.now(), dur = 350, ty = m.position.y;
  (function tick() {
    const t = Math.min(1, (performance.now() - t0) / dur);
    const e = 1 - Math.pow(1 - t, 3);
    m.scale.y = e; m.position.y = (TILE_H + 0.02) + (WALL_H / 2) * e;
    if (t < 1) requestAnimationFrame(tick);
  })();
}

/* ═══════════════════════════════════════════════════
   Pawns sync & anim
   ═══════════════════════════════════════════════════ */
const pawnTgt = [null, null];

function syncPawns(data) {
  for (let p = 0; p < 2; p++) {
    const m = ensurePawn(p);
    const [r, c] = data.pawns[p];
    const [x, z] = g2w(r, c);
    const tgt = new THREE.Vector3(x, TILE_H + 0.04, z);
    if (!pawnTgt[p]) { m.position.copy(tgt); pawnTgt[p] = tgt.clone(); }
    else if (!pawnTgt[p].equals(tgt)) { pawnTgt[p].copy(tgt); animPawn(m, tgt); }
  }
}

function animPawn(m, tgt) {
  const s = m.position.clone(), t0 = performance.now(), dur = 400;
  const pk = Math.max(s.y, tgt.y) + 0.3;
  (function tick() {
    const t = Math.min(1, (performance.now() - t0) / dur);
    const e = t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    m.position.x = s.x + (tgt.x - s.x) * e;
    m.position.z = s.z + (tgt.z - s.z) * e;
    m.position.y = s.y + (pk - s.y) * Math.sin(t * Math.PI) + (tgt.y - s.y) * e;
    if (t < 1) requestAnimationFrame(tick);
  })();
}

/* ═══════════════════════════════════════════════════
   Overlays (legal, paths)
   ═══════════════════════════════════════════════════ */
function clearOvl() {
  legalOvl.forEach((m) => { boardGroup.remove(m); m.geometry.dispose(); m.material.dispose(); });
  legalOvl = [];
  pathOvl.forEach((m) => { boardGroup.remove(m); m.geometry.dispose(); m.material.dispose(); });
  pathOvl = [];
  wallGhosts.forEach((m) => { hintGroup.remove(m); m.geometry.dispose(); m.material.dispose(); });
  wallGhosts = [];
  hoverTarget = null;
}

function showLegal(data) {
  if (!el.toggleLegal.checked) return;
  if (ui.mode !== "move") {
    showWallHints(data);
    return;
  }
  const geo = new THREE.PlaneGeometry(TILE * 0.58, TILE * 0.58);
  data.legalMoves.forEach(([r, c]) => {
    const mat = new THREE.MeshBasicMaterial({ color: C.legal, transparent: true, opacity: 0.42, side: THREE.DoubleSide });
    const m = new THREE.Mesh(geo, mat);
    const [x, z] = g2w(r, c);
    m.position.set(x, TILE_H + 0.055, z); m.rotation.x = -Math.PI / 2;
    m.userData = { row: r, col: c, type: "legal" };
    boardGroup.add(m); legalOvl.push(m);
  });
}

function showWallHints(data) {
  const orientation = ui.mode;
  data.legalWalls
    .filter((w) => w.orientation === orientation)
    .forEach((w) => {
      const mat = new THREE.MeshStandardMaterial({
        color: C.legal,
        transparent: true,
        opacity: 0.2,
        roughness: 0.76,
        metalness: 0.0,
        flatShading: true,
        depthWrite: false,
      });
      const m = new THREE.Mesh(wallGeo.clone(), mat);
      const [x, z] = g2w(w.row + 0.5, w.col + 0.5);
      m.position.set(x, TILE_H + 0.06 + WALL_H / 2, z);
      if (w.orientation === "V") m.rotation.y = Math.PI / 2;
      m.scale.y = 0.28;
      m.userData = { type: "wall-ghost", orientation: w.orientation, row: w.row, col: w.col };
      hintGroup.add(m);
      wallGhosts.push(m);
    });
}

function showPaths(data) {
  if (!el.togglePaths.checked) return;
  const geo = new THREE.SphereGeometry(0.06, 8, 6);
  data.paths.forEach((path, p) => {
    const color = p === 0 ? C.pathP0 : C.pathP1;
    path.forEach(([r, c]) => {
      const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6 });
      const m = new THREE.Mesh(geo, mat);
      const [x, z] = g2w(r, c);
      const off = p === 0 ? -0.12 : 0.12;
      m.position.set(x + off, TILE_H + 0.12, z + off);
      boardGroup.add(m); pathOvl.push(m);
    });
  });
}

/* ═══════════════════════════════════════════════════
   Interaction (click with orbit-aware debounce)
   ═══════════════════════════════════════════════════ */
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let pointerStart = null;

function setupInteraction(container) {
  container.addEventListener("pointerdown", (e) => { pointerStart = { x: e.clientX, y: e.clientY }; });
  container.addEventListener("pointermove", (e) => updateHover(e, container));
  container.addEventListener("pointerleave", () => setHover(null, container));
  container.addEventListener("pointerup", (e) => {
    if (!pointerStart) return;
    const dx = e.clientX - pointerStart.x, dy = e.clientY - pointerStart.y;
    if (dx * dx + dy * dy < 25) handleClick(e, container);
    pointerStart = null;
  });
}

function pickInteractive(event, container) {
  if (!currentState || currentState.done) return null;
  const p = currentState.currentPlayer;
  if (currentState.playerTypes[p] !== "Human") return null;

  const rect = container.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  if (ui.mode === "move") {
    const hits = raycaster.intersectObjects(legalOvl, false);
    return hits.length ? hits[0].object : null;
  }
  const hits = raycaster.intersectObjects(wallGhosts, false);
  return hits.length ? hits[0].object : null;
}

function updateHover(event, container) {
  setHover(pickInteractive(event, container), container);
}

function setHover(target, container) {
  if (hoverTarget === target) return;
  if (hoverTarget?.material) {
    hoverTarget.material.opacity = hoverTarget.userData.type === "wall-ghost" ? 0.2 : 0.42;
    hoverTarget.scale.set(hoverTarget.scale.x, hoverTarget.userData.type === "wall-ghost" ? 0.28 : 1, hoverTarget.scale.z);
  }
  hoverTarget = target;
  if (hoverTarget?.material) {
    hoverTarget.material.opacity = 0.72;
    if (hoverTarget.userData.type === "wall-ghost") hoverTarget.scale.y = 0.74;
  }
  container.style.cursor = target ? "pointer" : "";
}

function handleClick(event, container) {
  if (!currentState || currentState.done) return;
  const p = currentState.currentPlayer;
  if (currentState.playerTypes[p] !== "Human") return;
  const target = pickInteractive(event, container);

  if (ui.mode === "move") {
    if (target) {
      const { row, col } = target.userData;
      apiPost("/api/human-action", { type: "move", row, col }).then(handleState);
    }
  } else {
    if (target) {
      const { orientation, row, col } = target.userData;
      apiPost("/api/human-action", { type: "wall", orientation, row, col }).then(handleState);
      return;
    }
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -(TILE_H + 0.02));
    const pt = new THREE.Vector3();
    raycaster.ray.intersectPlane(plane, pt);
    if (!pt) return;
    const wr = Math.round(pt.z / SP + HALF - 0.5);
    const wc = Math.round(pt.x / SP + HALF - 0.5);
    const ori = ui.mode;
    if (wr >= 0 && wr < N - 1 && wc >= 0 && wc < N - 1) {
      if (currentState.legalWalls.some((w) => w.orientation === ori && w.row === wr && w.col === wc)) {
        apiPost("/api/human-action", { type: "wall", orientation: ori, row: wr, col: wc }).then(handleState);
      }
    }
  }
}

/* ═══════════════════════════════════════════════════
   Animate
   ═══════════════════════════════════════════════════ */
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  composer.render();
}

/* ═══════════════════════════════════════════════════
   API
   ═══════════════════════════════════════════════════ */
const apiGet = (u) => fetch(u).then((r) => r.json());
const apiPost = (u, b = {}) => fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) }).then((r) => r.json());

function handleState(data) {
  if (data.error) {
    flashStatus(data.error);
    return;
  }
  currentState = data;
  render(data);
}

function flashStatus(message) {
  if (el.boardHint) {
    el.boardHint.textContent = message;
    el.boardHint.classList.add("flash");
  }
  setBadge("Action rejected", "var(--danger)");
}

/* ═══════════════════════════════════════════════════
   UI state
   ═══════════════════════════════════════════════════ */
const ui = { mode: "move", speed: 420, playing: false, timer: null, t0: null, tInt: null };

/* ═══════════════════════════════════════════════════
   DOM cache
   ═══════════════════════════════════════════════════ */
const el = {};
function cacheEls() {
  ["player0","player1","p0Turn","p1Turn","p0Rack","p1Rack",
   "p0PanelPath","p1PanelPath","p0PanelDiv","p1PanelDiv","p0WallCount","p1WallCount",
   "stepButton","playButton","resetButton","speedPresets",
   "turnMetric","turnMetric2","currentMetric","p0Path","p1Path","p0PathD","p1PathD","p0Diversity","p1Diversity",
   "timelineTrack","lastAction","timerValue","matchStatus",
   "winnerOverlay","winnerCrown","winnerTitle","winnerDetail","winnerReplay",
   "togglePaths","toggleLegal","toggleWalls","boardContainer",
   "boardHint",
   "drawer","drawerToggle","drawerClose","actionLog",
  ].forEach((id) => { el[id] = document.getElementById(id); });
}

/* ═══════════════════════════════════════════════════
   Render
   ═══════════════════════════════════════════════════ */
function render(data) {
  clearOvl();
  syncWalls(data);
  wallGroup.visible = el.toggleWalls.checked;
  hintGroup.visible = el.toggleLegal.checked;
  syncPawns(data);
  showLegal(data);
  showPaths(data);
  renderDOM(data);
}

function modeLabel() {
  if (ui.mode === "move") return "move target";
  return ui.mode === "H" ? "horizontal wall" : "vertical wall";
}

function setBadge(text, color) {
  const badge = el.matchStatus;
  if (!badge) return;
  badge.querySelector(".badge-pip").style.background = color;
  badge.lastChild.textContent = text;
}

function renderDOM(data) {
  el.turnMetric.textContent = data.turnCount;
  if (el.currentMetric) el.currentMetric.textContent = `P${data.currentPlayer}`;
  el.p0Path.textContent = data.pathLengths[0];
  el.p1Path.textContent = data.pathLengths[1];
  if (el.p0PanelPath) el.p0PanelPath.textContent = data.pathLengths[0];
  if (el.p1PanelPath) el.p1PanelPath.textContent = data.pathLengths[1];
  if (el.p0PanelDiv) el.p0PanelDiv.textContent = data.pathDiversity[0];
  if (el.p1PanelDiv) el.p1PanelDiv.textContent = data.pathDiversity[1];
  if (el.p0WallCount) el.p0WallCount.textContent = data.remainingWalls[0];
  if (el.p1WallCount) el.p1WallCount.textContent = data.remainingWalls[1];
  if (el.p0Diversity) el.p0Diversity.textContent = data.pathDiversity[0];
  if (el.p1Diversity) el.p1Diversity.textContent = data.pathDiversity[1];

  el.p0Turn.classList.toggle("active", data.currentPlayer === 0);
  el.p0Turn.textContent = data.currentPlayer === 0 ? "To move" : "Standby";
  el.p1Turn.classList.toggle("active", data.currentPlayer === 1);
  el.p1Turn.textContent = data.currentPlayer === 1 ? "To move" : "Standby";

  if (data.playerTypes) {
    if (el.player0.value !== data.playerTypes[0]) el.player0.value = data.playerTypes[0];
    if (el.player1.value !== data.playerTypes[1]) el.player1.value = data.playerTypes[1];
  }

  wallRack(el.p0Rack, data.remainingWalls[0]);
  wallRack(el.p1Rack, data.remainingWalls[1]);

  if (data.lastAction && el.lastAction) el.lastAction.textContent = data.lastAction;

  if (data.done) {
    setBadge(data.winner !== null ? `P${data.winner} wins` : "Draw", "var(--gold-hi)");
    if (el.boardHint) {
      el.boardHint.classList.remove("flash");
      el.boardHint.textContent = data.winner !== null ? `Player ${data.winner} wins` : "Draw";
    }
  } else {
    const playerType = data.playerTypes[data.currentPlayer];
    setBadge(`P${data.currentPlayer}: ${playerType}`, "var(--hint)");
    if (el.boardHint) {
      el.boardHint.classList.remove("flash");
      el.boardHint.textContent = playerType === "Human" ? `P${data.currentPlayer} ${modeLabel()}` : `P${data.currentPlayer} agent ready`;
    }
  }

  if (el.turnMetric2) el.turnMetric2.textContent = data.turnCount;
  if (el.p0PathD) el.p0PathD.textContent = data.pathLengths[0];
  if (el.p1PathD) el.p1PathD.textContent = data.pathLengths[1];

  renderTimeline(data);
  renderLog(data);
  if (data.done && data.winner !== null) showWinner(data);
}

function wallRack(c, rem) {
  if (c.children.length !== 10) {
    c.innerHTML = "";
    for (let i = 0; i < 10; i++) { const b = document.createElement("div"); b.className = "brick"; c.appendChild(b); }
  }
  Array.from(c.children).forEach((b, i) => { b.className = "brick " + (i < rem ? "active" : "spent"); });
}

function renderTimeline(data) {
  const tr = el.timelineTrack, hist = data.history || [];
  while (tr.children.length < hist.length) { const p = document.createElement("div"); p.className = "tl-pip"; tr.appendChild(p); }
  while (tr.children.length > hist.length) tr.removeChild(tr.lastChild);
  hist.forEach((e, i) => {
    const p = tr.children[i], cl = e.startsWith("P0") ? "p0" : "p1";
    p.className = `tl-pip ${cl}` + (i === hist.length - 1 ? " current" : "");
    p.textContent = i + 1; p.title = e;
  });
  if (tr.lastChild) tr.lastChild.scrollIntoView({ inline: "end", behavior: "smooth" });
}

function renderLog(data) {
  if (!el.actionLog) return;
  const ol = el.actionLog, hist = data.history || [];
  ol.innerHTML = "";
  for (let i = hist.length - 1; i >= Math.max(0, hist.length - 40); i--) {
    const li = document.createElement("li");
    const idx = document.createElement("span");
    const p = hist[i].startsWith("P0") ? 0 : 1;
    idx.className = `log-idx lp${p}`; idx.textContent = i + 1;
    li.appendChild(idx); li.appendChild(document.createTextNode(hist[i]));
    ol.appendChild(li);
  }
}

function showWinner(data) {
  const w = data.winner;
  el.winnerCrown.textContent = "WIN";
  el.winnerTitle.textContent = `Player ${w} Wins!`;
  el.winnerTitle.style.color = w === 0 ? "var(--p0)" : "var(--p1)";
  el.winnerDetail.textContent = `Completed in ${data.turnCount} turns`;
  el.winnerOverlay.classList.add("visible");
  stopPlay();
}

/* ═══════════════════════════════════════════════════
   Controls
   ═══════════════════════════════════════════════════ */
function initControls() {
  el.stepButton.addEventListener("click", doStep);
  el.playButton.addEventListener("click", togglePlay);
  el.resetButton.addEventListener("click", doReset);
  el.winnerReplay.addEventListener("click", doReset);

  el.speedPresets.querySelectorAll(".sp-btn").forEach((btn) => {
    if (btn.hasAttribute("data-default")) btn.classList.add("active");
    btn.addEventListener("click", () => {
      ui.speed = parseInt(btn.dataset.ms);
      el.speedPresets.querySelectorAll(".sp-btn").forEach((b) => b.classList.toggle("active", b === btn));
      if (ui.playing) { stopPlay(); startPlay(); }
    });
  });

  document.querySelectorAll(".mode").forEach((btn) => {
    btn.addEventListener("click", () => {
      setMode(btn.dataset.mode);
    });
  });

  [el.togglePaths, el.toggleLegal, el.toggleWalls].forEach((cb) => {
    cb.addEventListener("change", () => { if (currentState) render(currentState); });
  });

  el.drawerToggle.addEventListener("click", () => el.drawer.classList.toggle("open"));
  el.drawerClose.addEventListener("click", () => el.drawer.classList.remove("open"));

  initAgents();
}

async function initAgents() {
  const data = await apiGet("/api/agents");
  [el.player0, el.player1].forEach((sel, i) => {
    sel.innerHTML = "";
    (data.agents || []).forEach((name) => {
      const o = document.createElement("option"); o.value = name; o.textContent = name; sel.appendChild(o);
    });
    sel.value = i === 0 ? "Human" : "Random";
  });
  const onChange = () => apiPost("/api/config", { players: [el.player0.value, el.player1.value] }).then(handleState);
  el.player0.addEventListener("change", onChange);
  el.player1.addEventListener("change", onChange);
}

async function doStep() {
  if (!currentState || currentState.done) return;
  if (currentState.playerTypes[currentState.currentPlayer] === "Human") {
    flashStatus("Human turn");
    return;
  }
  handleState(await apiPost("/api/agent-step"));
}

function togglePlay() { ui.playing ? stopPlay() : startPlay(); }

function startPlay() {
  if (!currentState || currentState.done) return;
  if (currentState.playerTypes[currentState.currentPlayer] === "Human") {
    flashStatus("Human turn");
    return;
  }
  ui.playing = true;
  el.playButton.textContent = "Pause"; el.playButton.classList.add("playing");
  if (!ui.t0) { ui.t0 = Date.now(); ui.tInt = setInterval(updateTimer, 1000); }
  tick();
}

function stopPlay() {
  ui.playing = false;
  el.playButton.textContent = "Play"; el.playButton.classList.remove("playing");
  if (ui.timer) { clearTimeout(ui.timer); ui.timer = null; }
}

async function tick() {
  if (!ui.playing || !currentState || currentState.done) { stopPlay(); return; }
  if (currentState.playerTypes[currentState.currentPlayer] === "Human") { stopPlay(); return; }
  handleState(await apiPost("/api/agent-step"));
  if (currentState.done) { stopPlay(); return; }
  ui.timer = setTimeout(tick, ui.speed);
}

async function doReset() {
  stopPlay();
  el.winnerOverlay.classList.remove("visible");
  if (ui.tInt) { clearInterval(ui.tInt); ui.tInt = null; }
  ui.t0 = null; el.timerValue.textContent = "00:00";

  Object.values(wallMeshes).forEach((m) => { wallGroup.remove(m); m.geometry.dispose(); m.material.dispose(); });
  wallMeshes = {}; pawnTgt[0] = null; pawnTgt[1] = null;

  handleState(await apiPost("/api/reset"));
}

function updateTimer() {
  if (!ui.t0) return;
  const s = Math.floor((Date.now() - ui.t0) / 1000);
  el.timerValue.textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

/* ═══════════════════════════════════════════════════
   Keyboard
   ═══════════════════════════════════════════════════ */
function initKeyboard() {
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    switch (e.key) {
      case " ": e.preventDefault(); togglePlay(); break;
      case "ArrowRight": doStep(); break;
      case "r": case "R": doReset(); break;
      case "m": case "M": setMode("move"); break;
      case "h": setMode("H"); break;
      case "v": setMode("V"); break;
      case "l": case "L": el.drawer.classList.toggle("open"); break;
    }
  });
}

function setMode(mode) {
  ui.mode = mode;
  document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  if (currentState) render(currentState);
}

/* ═══════════════════════════════════════════════════
   Init
   ═══════════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", async () => {
  cacheEls();
  initScene(el.boardContainer);
  initControls();
  initKeyboard();
  handleState(await apiGet("/api/state"));
});
