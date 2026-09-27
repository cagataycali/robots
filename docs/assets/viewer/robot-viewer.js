/*
 * <robot-viewer name="so101" [autoload]></robot-viewer>
 *
 * Renders any robot in the strands-robots registry in the browser, on MuJoCo
 * itself (the official @mujoco/mujoco WebAssembly build) with three.js for the
 * pixels. The MJCF and meshes stream from jsDelivr in front of the model's
 * public git repository at view time, so the docs ship no mesh files.
 *
 * One finger orbits, two fingers pan and zoom. Sliders move joints through
 * mj_forward; the Physics switch steps mj_step at real time with the sliders as
 * actuator targets. The code panel mirrors the sliders as robot.act({...}).
 *
 * Dependencies resolve through the import map in overrides/main.html:
 *   three, three/addons/, @mujoco/mujoco  (all pinned on cdn.jsdelivr.net)
 *
 * No build step. ES2022. Every embind handle is deleted on unload.
 */

const MANIFEST_URL = new URL("robots.json", import.meta.url);
const HIDDEN_GROUPS = new Set([3, 4, 5]); // MuJoCo convention: 3+ is collision geometry
const MAX_GEOMS = 20000;

let mujocoPromise = null;
let threePromise = null;
let manifestPromise = null;

function loadMujoco() {
  if (!mujocoPromise) {
    mujocoPromise = import("@mujoco/mujoco").then((m) => (m.default || m)());
  }
  return mujocoPromise;
}
function loadThree() {
  if (!threePromise) {
    threePromise = Promise.all([import("three"), import("three/addons/controls/OrbitControls.js")]).then(
      ([THREE, { OrbitControls }]) => ({ THREE, OrbitControls })
    );
  }
  return threePromise;
}
function loadManifest() {
  if (!manifestPromise) {
    manifestPromise = fetch(MANIFEST_URL).then((r) => {
      if (!r.ok) throw new Error(`robots.json ${r.status}`);
      return r.json();
    });
  }
  return manifestPromise;
}

const fmtMB = (n) => `${(n / 1e6).toFixed(1)} MB`;

/** Parse MJCF text for includes and asset files. Regex is enough: attribute order varies, tags do not. */
function scanXml(xml) {
  const attr = (tag, name) => {
    const m = xml.match(new RegExp(`<${tag}[^>]*\\b${name}="([^"]*)"`));
    return m ? m[1] : null;
  };
  const includes = [...xml.matchAll(/<include\s+[^>]*file="([^"]+)"/g)].map((m) => m[1]);
  const meshdir = attr("compiler", "meshdir") ?? attr("compiler", "assetdir") ?? "";
  const texturedir = attr("compiler", "texturedir") ?? attr("compiler", "assetdir") ?? "";
  const join = (dir, f) => (dir && !f.startsWith("/") ? `${dir.replace(/\/?$/, "/")}${f}` : f);
  const files = [];
  for (const m of xml.matchAll(/<(mesh|hfield|skin)\b[^>]*\bfile="([^"]+)"/g)) files.push(join(meshdir, m[2]));
  for (const m of xml.matchAll(/<texture\b[^>]*\bfile="([^"]+)"/g)) files.push(join(texturedir, m[1]));
  for (const m of xml.matchAll(/<texture\b[^>]*\bfile(?:right|left|up|down|front|back)="([^"]+)"/g)) files.push(join(texturedir, m[1]));
  return { includes, files };
}

const TEMPLATE = `
<style>
  :host { display:block; position:relative; font-family: Inter, system-ui, sans-serif; color: var(--sr-fg, #1b1b1b); }
  canvas { display:block; width:100%; height:100%; outline:none; }
  .poster, .status { position:absolute; inset:0; display:grid; place-items:center; text-align:center; padding:1rem; }
  .poster[hidden], .status[hidden], .chrome[hidden] { display:none; }
  .poster img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; opacity:.55; filter:saturate(.6); }
  .poster .card, .status .card { position:relative; background: var(--sr-card-bg, #fff); border:1px solid var(--sr-card-border, #e3ded2); border-radius:12px; padding:.9rem 1.1rem; max-width:22rem; box-shadow: 0 8px 30px rgba(0,0,0,.08); }
  .poster h4, .status h4 { margin:0 0 .25rem; font-size:.95rem; }
  .poster p, .status p { margin:0 0 .6rem; font-size:.75rem; color: var(--sr-muted, #6b6760); }
  button { font: inherit; font-size:.75rem; font-weight:600; padding:.45rem .9rem; border-radius:8px; border:1px solid #1b1b1b; background:#1b1b1b; color:#fff; cursor:pointer; }
  button:hover { border-color:#ff6a00; background:#ff6a00; color:#fff; }
  .bar { height:4px; border-radius:2px; background: var(--sr-card-border, #e3ded2); overflow:hidden; margin-top:.5rem; }
  .bar i { display:block; height:100%; width:0; background:#ff6a00; transition: width 120ms linear; }
  .chrome { position:absolute; left:0; right:0; bottom:0; display:flex; gap:.4rem; align-items:center; padding:.5rem .6rem; pointer-events:none; }
  .chrome > * { pointer-events:auto; }
  .chrome .spacer { flex:1; }
  .pill { font-size:.65rem; font-weight:500; padding:.3rem .6rem; border-radius:999px; border:1px solid var(--sr-card-border, #c9c6bf); background: var(--sr-card-bg, #fff); color: inherit; cursor:pointer; }
  .pill[aria-pressed="true"] { border-color:#ff6a00; color:#ff6a00; }
  .pill:hover { border-color:#ff6a00; background: var(--sr-card-bg, #fff); color:#ff6a00; }
  .joints { position:absolute; top:.6rem; right:.6rem; width: 15rem; max-height: calc(100% - 3.6rem); overflow:auto; background: color-mix(in srgb, var(--sr-card-bg, #fff) 88%, transparent); backdrop-filter: blur(6px); border:1px solid var(--sr-card-border, #e3ded2); border-radius:10px; padding:.55rem .7rem .5rem; font-size:.68rem; }
  .joints[hidden] { display:none; }
  .joints h5 { margin:0 0 .35rem; font-size:.66rem; letter-spacing:.06em; text-transform:uppercase; color: var(--sr-muted, #6b6760); font-weight:600; }
  .joint { display:grid; gap:.1rem; margin-bottom:.3rem; }
  .joint label { display:flex; justify-content:space-between; font-family: "JetBrains Mono", ui-monospace, monospace; font-size:.62rem; }
  .joint label output { color: var(--sr-muted, #6b6760); }
  .joint input[type=range] { width:100%; accent-color:#ff6a00; margin:0; height: 1rem; }
  .code { position:absolute; top:.6rem; left:.6rem; max-width: calc(100% - 17rem); font-family: "JetBrains Mono", ui-monospace, monospace; font-size:.62rem; line-height:1.45; background: color-mix(in srgb, var(--sr-card-bg, #fff) 88%, transparent); backdrop-filter: blur(6px); border:1px solid var(--sr-card-border, #e3ded2); border-radius:10px; padding:.5rem .7rem; white-space:pre; overflow:auto; max-height: 40%; }
  .code[hidden] { display:none; }
  .code b { color:#ff6a00; font-weight:600; }
  @media (max-width: 40em) { .joints { width: 11rem; } .code { max-width: calc(100% - 12.5rem); font-size:.56rem; } }
</style>
<canvas tabindex="0" aria-label="3D robot viewer"></canvas>
<div class="poster" part="poster"></div>
<div class="code" hidden></div>
<div class="joints" hidden></div>
<div class="chrome" hidden>
  <button class="pill" data-act="reset" title="Reset pose (R)">Reset</button>
  <button class="pill" data-act="physics" aria-pressed="false" title="Step MuJoCo at real time">Physics</button>
  <button class="pill" data-act="collision" aria-pressed="false" title="Show collision geometry">Collision</button>
  <span class="spacer"></span>
  <button class="pill" data-act="joints" aria-pressed="true">Joints</button>
  <button class="pill" data-act="code" aria-pressed="true">Code</button>
  <button class="pill" data-act="full" title="Fullscreen">Full</button>
</div>
`;

class RobotViewer extends HTMLElement {
  static get observedAttributes() { return ["name"]; }

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).innerHTML = TEMPLATE;
    this.$ = (s) => this.shadowRoot.querySelector(s);
    this._state = "idle";
    this._physics = false;
    this._showCollision = false;
    this._handles = [];
    this._three = null;
    this._raf = null;
    this._joints = [];
    this._onKey = (e) => { if (e.key === "r" || e.key === "R") this.resetPose(); };
  }

  connectedCallback() {
    this._entry = null;
    loadManifest()
      .then((m) => {
        this._entry = m.robots[this.getAttribute("name")] ?? null;
        this._renderPoster();
        if (this.hasAttribute("autoload")) this._whenVisible(() => this.load());
      })
      .catch((e) => this._fail(`Could not read the robot manifest (${e.message}).`));
    this.shadowRoot.addEventListener("click", (e) => {
      const b = e.target.closest("button[data-act]");
      if (b) this._action(b.dataset.act, b);
    });
    this.$("canvas").addEventListener("keydown", this._onKey);
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(this);
  }

  disconnectedCallback() {
    this._ro?.disconnect();
    this.unload();
  }

  attributeChangedCallback(n, oldV, newV) {
    if (n === "name" && oldV && oldV !== newV) { this.unload(); this.connectedCallback(); }
  }

  _whenVisible(fn) {
    const io = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
    }, { rootMargin: "200px" });
    io.observe(this);
  }

  _renderPoster() {
    const p = this.$(".poster");
    const e = this._entry;
    if (this._state !== "idle") return;
    if (!e) return this._fail(`No robot named "${this.getAttribute("name")}" in the registry.`);
    if (!e.sim) return this._fail(`${e.description} has no simulation model, so there is nothing to render.`);
    if (!e.viewer) return this._fail(`${e.description} simulates locally, but its model has no public source to stream from.`);
    const thumb = e.thumbnail ? `<img alt="" src="${new URL("../../" + e.thumbnail, import.meta.url)}">` : "";
    p.innerHTML = `${thumb}<div class="card"><h4>${e.description}</h4><p>${e.joints ?? "?"} joints. Runs MuJoCo in your browser. Meshes stream from ${this._sourceLabel()}.</p><button data-act="load">Load 3D</button></div>`;
    p.hidden = false;
  }

  _sourceLabel() {
    const m = (this._entry?.base_url || "").match(/\/gh\/([^/]+\/[^/@]+)@/);
    return m ? m[1] : "jsDelivr";
  }

  _status(title, text, progress) {
    let s = this.$(".status");
    if (!s) { s = document.createElement("div"); s.className = "status"; this.shadowRoot.appendChild(s); }
    s.innerHTML = `<div class="card"><h4>${title}</h4><p>${text}</p>${progress == null ? "" : `<div class="bar"><i style="width:${(progress * 100).toFixed(1)}%"></i></div>`}</div>`;
  }
  _clearStatus() { this.$(".status")?.remove(); }

  _fail(msg) {
    this._state = "error";
    this.$(".poster").hidden = true;
    this._status("Viewer unavailable", msg);
  }

  _action(act, btn) {
    switch (act) {
      case "load": this.load(); return;
      case "reset": this.resetPose(); return;
      case "physics":
        this._physics = !this._physics; btn.setAttribute("aria-pressed", String(this._physics));
        if (this._physics) for (const jt of this._joints) if (jt.act >= 0) this._data.ctrl[jt.act] = this._data.qpos[jt.qadr];
        return;
      case "collision":
        this._showCollision = !this._showCollision; btn.setAttribute("aria-pressed", String(this._showCollision)); this._applyVisibility(); return;
      case "joints": { const el = this.$(".joints"); el.hidden = !el.hidden; btn.setAttribute("aria-pressed", String(!el.hidden)); return; }
      case "code": { const el = this.$(".code"); el.hidden = !el.hidden; btn.setAttribute("aria-pressed", String(!el.hidden)); return; }
      case "full":
        if (document.fullscreenElement === this) document.exitFullscreen(); else this.requestFullscreen?.();
        return;
    }
  }

  /** Fetch the model, compile it in MuJoCo, build the three.js scene. Idempotent. */
  async load() {
    if (this._state !== "idle") return;
    this._state = "loading";
    const gen = (this._gen = (this._gen || 0) + 1);
    const stale = () => gen !== this._gen || this._state !== "loading";
    try {
      if (!this._entry) {
        const m = await loadManifest();
        this._entry = m.robots[this.getAttribute("name")] ?? null;
        if (!this._entry?.viewer) throw new Error(`no streamable model for "${this.getAttribute("name")}"`);
      }
      const e = this._entry;
      this.$(".poster").hidden = true;
      this._status("Loading MuJoCo", "The WebAssembly engine is 10 MB and cached after the first robot.");
      const [mujoco, three] = await Promise.all([loadMujoco(), loadThree()]);
      if (stale()) return;
      this._mujoco = mujoco;
      const files = await this._fetchAssets(e);
      if (stale()) return;
      this._status("Compiling model", `${files.count} files, ${fmtMB(files.bytes)}`);
      await new Promise((r) => setTimeout(r, 0));
      if (stale()) return;
      this._compile(files);
      this._buildScene(three);
      this._buildJoints();
      this._clearStatus();
      this.$(".poster").hidden = true;
      this.$(".chrome").hidden = false;
      const narrow = this.clientWidth < 640 || this.hasAttribute("compact");
      this.$(".joints").hidden = narrow;
      this.$(".code").hidden = narrow;
      this.$('[data-act="joints"]').setAttribute("aria-pressed", String(!narrow));
      this.$('[data-act="code"]').setAttribute("aria-pressed", String(!narrow));
      this._state = "ready";
      this._loop();
      this.dispatchEvent(new CustomEvent("robot-loaded", { detail: { name: this._entry.name } }));
    } catch (err) {
      if (stale()) return;
      console.error(err);
      this._fail(this._explain(err));
    }
  }

  _explain(err) {
    const m = String(err?.message || err);
    if (/Failed to fetch|NetworkError|Load failed/.test(m)) return "cdn.jsdelivr.net is unreachable from this network. The viewer needs it for the engine and the meshes.";
    if (/ 404/.test(m)) return `A model file is missing upstream: ${m}.`;
    return `MuJoCo refused the model: ${m.slice(0, 240)}`;
  }

  async _fetchAssets(e) {
    const base = e.base_url;
    const dec = new TextDecoder();
    const fetched = new Map();
    let bytes = 0, done = 0, total = 1;
    const progress = (label) => this._status("Streaming meshes", `${label} ${fmtMB(bytes)}, ${done}/${total} files`, done / total);
    const LFS = new Uint8Array([118, 101, 114, 115, 105, 111, 110, 32, 104, 116, 116, 112, 115, 58, 47, 47, 103, 105, 116, 45, 108, 102, 115]); // "version https://git-lfs"
    const isLfsPointer = (b) => b.length < 400 && LFS.every((c, i) => b[i] === c);
    const get = async (path) => {
      let r = await fetch(new URL(path, base).href);
      if (!r.ok) throw new Error(`${path} ${r.status}`);
      let buf = new Uint8Array(await r.arrayBuffer());
      if (isLfsPointer(buf) && e.lfs_url) {
        // jsDelivr serves the Git LFS pointer; the blob itself lives on GitHub's media host.
        r = await fetch(new URL(path, e.lfs_url).href);
        if (!r.ok) throw new Error(`${path} (lfs) ${r.status}`);
        buf = new Uint8Array(await r.arrayBuffer());
      }
      bytes += buf.length; done += 1; progress(path.split("/").pop());
      return buf;
    };
    // The scene, its includes (recursively), then every asset those name.
    const xmlQueue = [e.scene];
    const assets = new Set();
    const seen = new Set();
    let sceneXml = null;
    while (xmlQueue.length) {
      const f = xmlQueue.shift();
      if (seen.has(f)) continue;
      seen.add(f);
      total += 1;
      const buf = await get(f);
      const text = dec.decode(buf);
      if (sceneXml === null) sceneXml = text;
      fetched.set(f, buf);
      const dir = f.includes("/") ? f.slice(0, f.lastIndexOf("/") + 1) : "";
      const { includes, files } = scanXml(text);
      for (const inc of includes) xmlQueue.push(dir + inc);
      for (const a of files) assets.add(dir + a);
    }
    total = seen.size + assets.size;
    progress("");
    await Promise.all([...assets].map(async (a) => fetched.set(a, await get(a))));
    return { sceneName: e.scene, sceneXml, fetched, bytes, count: fetched.size };
  }

  _compile({ sceneName, sceneXml, fetched }) {
    const mj = this._mujoco;
    const vfs = new mj.MjVFS();
    this._handles.push(vfs);
    // MJCF resolves paths relative to the scene file's own directory.
    const sceneDir = sceneName.includes("/") ? sceneName.slice(0, sceneName.lastIndexOf("/") + 1) : "";
    for (const [path, buf] of fetched) {
      const rel = sceneDir && path.startsWith(sceneDir) ? path.slice(sceneDir.length) : path;
      vfs.addBuffer(rel, buf);
    }
    const model = mj.MjModel.from_xml_string(sceneXml, vfs);
    if (!model) throw new Error("MuJoCo returned no model");
    const data = new mj.MjData(model);
    this._handles.push(model, data);
    mj.mj_forward(model, data);
    this._model = model;
    this._data = data;
    this._qpos0 = Float64Array.from(data.qpos);
  }

  _buildScene({ THREE, OrbitControls }) {
    const canvas = this.$("canvas");
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 4 / 3, 0.01, 200);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    controls.maxPolarAngle = Math.PI * 0.52;

    scene.add(new THREE.HemisphereLight(0xffffff, 0xb8b0a0, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(1.5, -2, 3);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.bias = -0.0002;
    key.shadow.normalBias = 0.02;
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.6);
    fill.position.set(-2, 1.5, 1.5);
    scene.add(fill);

    const mj = this._mujoco, m = this._model;
    const G = mj.mjtGeom;
    const geomMeshes = [];
    const meshCache = new Map();
    const planeColor = getComputedStyle(this).getPropertyValue("--sr-viewer-bg").trim() || "#f3f0e9";
    for (let g = 0; g < m.ngeom; g++) {
      const type = m.geom_type[g];
      const size = [m.geom_size[3 * g], m.geom_size[3 * g + 1], m.geom_size[3 * g + 2]];
      const group = m.geom_group[g];
      let geometry;
      if (type === G.mjGEOM_MESH.value) {
        const id = m.geom_dataid[g];
        geometry = meshCache.get(id) ?? this._meshGeometry(THREE, id);
        meshCache.set(id, geometry);
      } else if (type === G.mjGEOM_PLANE.value) {
        geometry = new THREE.PlaneGeometry(size[0] ? 2 * size[0] : 40, size[1] ? 2 * size[1] : 40);
      } else if (type === G.mjGEOM_SPHERE.value) {
        geometry = new THREE.SphereGeometry(size[0], 24, 16);
      } else if (type === G.mjGEOM_CAPSULE.value) {
        geometry = new THREE.CapsuleGeometry(size[0], 2 * size[1], 8, 20); geometry.rotateX(Math.PI / 2);
      } else if (type === G.mjGEOM_CYLINDER.value) {
        geometry = new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 32); geometry.rotateX(Math.PI / 2);
      } else if (type === G.mjGEOM_BOX.value) {
        geometry = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
      } else if (type === G.mjGEOM_ELLIPSOID.value) {
        geometry = new THREE.SphereGeometry(1, 24, 16); geometry.scale(size[0], size[1], size[2]);
      } else {
        continue; // hfield, sdf: not drawn in v1
      }
      let rgba = [m.geom_rgba[4 * g], m.geom_rgba[4 * g + 1], m.geom_rgba[4 * g + 2], m.geom_rgba[4 * g + 3]];
      const matid = m.geom_matid[g];
      if (matid >= 0) rgba = [m.mat_rgba[4 * matid], m.mat_rgba[4 * matid + 1], m.mat_rgba[4 * matid + 2], m.mat_rgba[4 * matid + 3]];
      const isPlane = type === G.mjGEOM_PLANE.value;
      const material = new THREE.MeshStandardMaterial({
        color: isPlane ? new THREE.Color(planeColor) : new THREE.Color(rgba[0], rgba[1], rgba[2]),
        roughness: isPlane ? 0.95 : 0.55,
        metalness: isPlane ? 0 : 0.08,
        transparent: rgba[3] < 1,
        opacity: rgba[3],
        side: isPlane ? THREE.DoubleSide : THREE.FrontSide,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = !isPlane;
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      mesh.userData = { geom: g, group, collision: HIDDEN_GROUPS.has(group), plane: isPlane };
      scene.add(mesh);
      geomMeshes.push(mesh);
    }
    this._three = { THREE, renderer, scene, camera, controls, geomMeshes };
    this._syncPoses();
    // Frame the robot: bounding box of every visible non-plane geom at the home pose.
    const bbox = new THREE.Box3();
    for (const mesh of geomMeshes) {
      if (mesh.userData.plane || mesh.userData.collision) continue;
      mesh.geometry.computeBoundingBox();
      bbox.union(mesh.geometry.boundingBox.clone().applyMatrix4(mesh.matrix));
    }
    if (bbox.isEmpty()) bbox.set(new THREE.Vector3(-0.5, -0.5, 0), new THREE.Vector3(0.5, 0.5, 1));
    const center = bbox.getCenter(new THREE.Vector3());
    const radius = Math.max(bbox.getSize(new THREE.Vector3()).length() / 2, 0.15);
    controls.target.copy(center);
    camera.position.set(center.x + radius * 1.6, center.y - radius * 1.8, center.z + radius * 0.9);
    camera.near = radius / 100; camera.far = radius * 100; camera.updateProjectionMatrix();
    key.shadow.camera.left = key.shadow.camera.bottom = -radius * 1.6;
    key.shadow.camera.right = key.shadow.camera.top = radius * 1.6;
    key.shadow.camera.near = radius * 0.5; key.shadow.camera.far = radius * 8;
    key.shadow.camera.updateProjectionMatrix();
    key.position.copy(center).add(new THREE.Vector3(radius * 1.5, -radius * 2, radius * 3));
    key.target.position.copy(center); scene.add(key.target);
    controls.minDistance = radius * 0.5; controls.maxDistance = radius * 12;
    // A faint grid on the floor for depth; hidden when the model has no ground plane.
    if (geomMeshes.some((x) => x.userData.plane)) {
      const grid = new THREE.GridHelper(Math.max(2, radius * 8), Math.max(8, Math.round(radius * 8 / 0.1)), 0xd8d2c4, 0xe6e1d5);
      grid.rotation.x = Math.PI / 2;
      grid.position.z = 0.0015;
      grid.material.transparent = true; grid.material.opacity = 0.7;
      scene.add(grid);
      this._grid = grid;
    }
    this._applyVisibility();
    this._resize();
  }

  _meshGeometry(THREE, id) {
    const m = this._model;
    const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id];
    const fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
    const pos = new Float32Array(m.mesh_vert.subarray(3 * va, 3 * (va + vn)));
    const idx = new Uint32Array(m.mesh_face.subarray(3 * fa, 3 * (fa + fn)));
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    return geo;
  }

  _applyVisibility() {
    if (!this._three) return;
    for (const mesh of this._three.geomMeshes) {
      if (!mesh.userData.collision) continue;
      mesh.visible = this._showCollision;
      mesh.material.wireframe = true;
      mesh.material.transparent = true;
      mesh.material.opacity = 0.5;
    }
  }

  _syncPoses() {
    const xpos = this._data.geom_xpos, xmat = this._data.geom_xmat;
    for (const mesh of this._three.geomMeshes) {
      const g = mesh.userData.geom, p = 3 * g, r = 9 * g;
      mesh.matrix.set(
        xmat[r], xmat[r + 1], xmat[r + 2], xpos[p],
        xmat[r + 3], xmat[r + 4], xmat[r + 5], xpos[p + 1],
        xmat[r + 6], xmat[r + 7], xmat[r + 8], xpos[p + 2],
        0, 0, 0, 1
      );
    }
  }

  _buildJoints() {
    const mj = this._mujoco, m = this._model, d = this._data;
    const el = this.$(".joints");
    const rows = [];
    this._joints = [];
    const HINGE = mj.mjtJoint.mjJNT_HINGE.value, SLIDE = mj.mjtJoint.mjJNT_SLIDE.value;
    for (let j = 0; j < m.njnt; j++) {
      const type = m.jnt_type[j];
      if (type !== HINGE && type !== SLIDE) continue;
      const name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT.value, j) || `joint_${j}`;
      let lo = m.jnt_range[2 * j], hi = m.jnt_range[2 * j + 1];
      // mjtByte arrays (jnt_limited) are not readable in the 3.14 bindings; an unlimited joint has range 0 0.
      if (lo === hi) { lo = -Math.PI; hi = Math.PI; }
      const qadr = m.jnt_qposadr[j];
      let act = -1;
      for (let u = 0; u < m.nu; u++) {
        if (m.actuator_trntype[u] === mj.mjtTrn.mjTRN_JOINT.value && m.actuator_trnid[2 * u] === j) { act = u; break; }
      }
      this._joints.push({ j, name, lo, hi, qadr, act });
      const v = d.qpos[qadr];
      rows.push(`<div class="joint"><label for="j${j}"><span>${name}</span><output id="o${j}">${v.toFixed(2)}</output></label><input id="j${j}" type="range" min="${lo}" max="${hi}" step="${(hi - lo) / 400}" value="${v}" data-j="${j}" aria-label="${name}"></div>`);
    }
    el.innerHTML = `<h5>${this._joints.length} joints</h5>${rows.join("")}`;
    el.oninput = (e) => {
      const inp = e.target.closest("input[data-j]");
      if (!inp) return;
      const jt = this._joints.find((x) => x.j === Number(inp.dataset.j));
      const v = Number(inp.value);
      d.qpos[jt.qadr] = v;
      if (jt.act >= 0) d.ctrl[jt.act] = v;
      this.shadowRoot.getElementById(`o${jt.j}`).textContent = v.toFixed(2);
      if (!this._physics) { d.qvel.fill(0); mj.mj_forward(m, d); }
      this._renderCode();
    };
    this._renderCode();
  }

  _renderCode() {
    const d = this._data;
    const moved = this._joints.filter((jt) => Math.abs(d.qpos[jt.qadr] - this._qpos0[jt.qadr]) > 1e-3);
    const body = moved.length
      ? moved.map((jt) => `    <b>"${jt.name}"</b>: ${d.qpos[jt.qadr].toFixed(3)},`).join("\n")
      : `    <span style="opacity:.55"># move a slider</span>`;
    this.$(".code").innerHTML = `from strands_robots import Robot\n\nrobot = Robot(<b>"${this._entry.name}"</b>)\nrobot.act({\n${body}\n})`;
  }

  resetPose() {
    if (!this._data) return;
    const mj = this._mujoco, m = this._model, d = this._data;
    mj.mj_resetData(m, d);
    mj.mj_forward(m, d);
    for (const jt of this._joints) {
      const inp = this.shadowRoot.getElementById(`j${jt.j}`);
      if (inp) { inp.value = d.qpos[jt.qadr]; this.shadowRoot.getElementById(`o${jt.j}`).textContent = d.qpos[jt.qadr].toFixed(2); }
      if (jt.act >= 0) d.ctrl[jt.act] = d.qpos[jt.qadr];
    }
    this._renderCode();
  }

  _resize() {
    if (!this._three) return;
    const { renderer, camera } = this._three;
    const w = this.clientWidth, h = this.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  _loop() {
    const mj = this._mujoco, m = this._model, d = this._data;
    let last = performance.now();
    const tick = (now) => {
      if (this._state !== "ready") return;
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      if (this._physics) {
        const target = d.time + dt;
        let n = 0;
        while (d.time < target && n++ < 200) mj.mj_step(m, d);
        for (const jt of this._joints) {
          const o = this.shadowRoot.getElementById(`o${jt.j}`);
          if (o) o.textContent = d.qpos[jt.qadr].toFixed(2);
        }
      }
      this._three.controls.update();
      this._syncPoses();
      this._three.renderer.render(this._three.scene, this._three.camera);
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  /** Free every embind handle and GPU resource. Safe to call twice. */
  unload() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
    this._state = "idle";
    if (this._three) {
      for (const mesh of this._three.geomMeshes) { mesh.geometry.dispose(); mesh.material.dispose(); }
      this._grid?.geometry.dispose(); this._grid?.material.dispose(); this._grid = null;
      this._three.controls.dispose();
      this._three.renderer.dispose();
      this._three = null;
    }
    for (const h of this._handles.reverse()) { try { h.delete(); } catch { /* already freed */ } }
    this._handles = [];
    this._model = this._data = null;
    this._joints = [];
    this.$(".chrome").hidden = true;
    this.$(".joints").hidden = true;
    this.$(".code").hidden = true;
  }
}

if (!customElements.get("robot-viewer")) customElements.define("robot-viewer", RobotViewer);

// Catalog filter chips (robots/index.md): .sr-filter button[data-family] toggles .sr-robot[data-family].
document.addEventListener("click", (e) => {
  const b = e.target.closest(".sr-filter button[data-family], .sr-filter-btn[data-family]");
  if (!b) return;
  const fam = b.dataset.family;
  for (const x of b.parentElement.querySelectorAll("button")) x.setAttribute("aria-pressed", String(x === b));
  for (const card of document.querySelectorAll(".sr-robot[data-family]")) card.hidden = fam !== "all" && card.dataset.family !== fam;
});
