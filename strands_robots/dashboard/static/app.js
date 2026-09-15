/* strands robots dashboard - app shell. ES modules, no bundler.
   Every request goes through api(); a 401 anywhere routes to the login view. */

const $ = (sel, root = document) => root.querySelector(sel);
const views = ["fleet", "sim", "agent", "settings"];
let authenticated = false;

export async function api(path, init = {}) {
  const res = await fetch(path, { credentials: "same-origin", headers: { "content-type": "application/json", ...(init.headers || {}) }, ...init });
  if (res.status === 401) { showLogin(); throw new Error("sign in required"); }
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(body?.error || res.statusText);
  return body;
}

/* ---- WebAuthn helpers: py_webauthn hands out base64url, the browser wants ArrayBuffers ---- */
const b64uToBuf = (s) => Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(s.length / 4) * 4, "=")), c => c.charCodeAt(0)).buffer;
const bufToB64u = (b) => btoa(String.fromCharCode(...new Uint8Array(b))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

function toCreationOptions(o) {
  return { ...o, challenge: b64uToBuf(o.challenge), user: { ...o.user, id: b64uToBuf(o.user.id) },
    excludeCredentials: (o.excludeCredentials || []).map(c => ({ ...c, id: b64uToBuf(c.id) })) };
}
function toRequestOptions(o) {
  return { ...o, challenge: b64uToBuf(o.challenge), allowCredentials: (o.allowCredentials || []).map(c => ({ ...c, id: b64uToBuf(c.id) })) };
}
function serializeCredential(cred) {
  const r = cred.response;
  const out = { id: cred.id, rawId: bufToB64u(cred.rawId), type: cred.type, response: { clientDataJSON: bufToB64u(r.clientDataJSON) } };
  if (r.attestationObject) out.response.attestationObject = bufToB64u(r.attestationObject);
  if (r.authenticatorData) out.response.authenticatorData = bufToB64u(r.authenticatorData);
  if (r.signature) out.response.signature = bufToB64u(r.signature);
  if (r.userHandle) out.response.userHandle = bufToB64u(r.userHandle);
  if (cred.getClientExtensionResults) out.clientExtensionResults = cred.getClientExtensionResults();
  return out;
}

/* ---- views ---- */
function show(view) {
  for (const v of [...views, "login"]) $(`#view-${v}`).hidden = v !== view;
  for (const b of $("#tabs").children) b.classList.toggle("on", b.dataset.view === view);
  if (view !== "login") location.hash = view;
}

function showLogin() { authenticated = false; $("#who").textContent = "signed out"; show("login"); }

async function loadStatus() {
  const s = await fetch("/api/auth/status", { credentials: "same-origin" }).then(r => r.json());
  const hint = $("#login-hint"), enrol = $("#enrol"), login = $("#login");
  enrol.hidden = true; login.hidden = true;
  if (s.warning) hint.textContent = s.warning;
  if (s.setup_required) {
    hint.textContent = s.bootstrap_source === "env"
      ? "First passkey. Paste the STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN you started the dashboard with."
      : "First passkey. Paste the token from the enrol_token file beside ~/.strands_dashboard/auth.json on the machine running the dashboard.";
    enrol.hidden = false;
  } else if (!s.authenticated && !s.open_posture) {
    hint.textContent = "This dashboard is sealed by a passkey.";
    login.hidden = false;
  }
  authenticated = s.authenticated || s.open_posture;
  return s;
}

async function loadWho() {
  try {
    const w = await api("/api/whoami");
    $("#who").textContent = w.via === "passkey" ? `passkey · ${w.name || w.sub}` : w.via === "token" ? "token" : "this machine · no passkey yet";
  } catch { /* routed to login */ }
}

async function loadFleet() {
  const box = $("#fleet"); box.textContent = "";
  try {
    const f = await api("/api/fleet");
    $("#fleet-count").textContent = `${f.robots.length} robots`;
    for (const r of f.robots) {
      const el = document.createElement("article"); el.className = "robot";
      el.innerHTML = `<div class="name">${r.name}</div><div class="desc">${r.description || ""}</div>
        <div class="meta"><span class="pill">${r.category || ""}</span><span class="pill">${r.joints ?? "?"} dof</span>
        ${r.has_sim ? '<span class="pill sim">sim</span>' : ""}${r.has_real ? '<span class="pill real">real</span>' : ""}</div>`;
      box.appendChild(el);
    }
  } catch (e) {
    if (e.message !== "sign in required") box.innerHTML = `<p class="muted">${e.message === "Not Found" ? "The fleet route arrives with the next slice." : e.message}</p>`;
  }
}

async function loadSettings() {
  const form = $("#settings-form"); form.textContent = "";
  const { settings, file } = await api("/api/settings");
  $("#settings-file").textContent = file;
  for (const [section, values] of Object.entries(settings)) {
    const h = document.createElement("h2"); h.textContent = section; form.appendChild(h);
    for (const [key, value] of Object.entries(values)) {
      const label = document.createElement("label");
      const isSecret = section === "security" && key === "auth_token";
      const shown = isSecret ? (value ? "(set)" : "(unset)") : Array.isArray(value) ? value.join(", ") : value ?? "";
      label.innerHTML = `<span class="mono">${section}.${key}</span>`;
      const input = document.createElement("input"); input.name = `${section}.${key}`; input.value = shown; input.placeholder = isSecret ? "leave to keep" : "";
      label.appendChild(input); form.appendChild(label);
    }
  }
  const btn = document.createElement("button"); btn.type = "submit"; btn.textContent = "Save"; form.appendChild(btn);
}

async function saveSettings(ev) {
  ev.preventDefault();
  const patch = {};
  for (const input of ev.target.querySelectorAll("input")) {
    const [section, key] = input.name.split(".");
    let v = input.value.trim();
    if (section === "security" && key === "auth_token" && (v === "(set)" || v === "(unset)" || v === "")) continue;
    if (v === "") v = null;
    (patch[section] ||= {})[key] = v;
  }
  try {
    const r = await api("/api/settings", { method: "POST", body: JSON.stringify(patch) });
    $("#settings-msg").textContent = r.changed.length ? `saved: ${r.changed.join(", ")}` : "nothing changed";
  } catch (e) { $("#settings-msg").textContent = e.message; }
}


/* ---- sim ---- */
const sockets = new Map();

function lockoutLine(l) {
  const el = $("#lockout");
  el.className = `lockout ${l.state}`;
  el.textContent = l.state === "locked" ? `e-stop engaged · ${l.by || "dashboard"} · ${l.reason}` : l.state === "unknown" ? `lockout unknown · ${l.reason}` : `clear · ${l.reason}`;
}

async function loadSim() {
  const sel = $("#sim-robot");
  if (!sel.options.length) {
    const f = await api("/api/fleet?mode=sim");
    for (const r of f.robots.filter(r => r.model_local)) {
      const o = document.createElement("option"); o.value = r.name; o.textContent = `${r.name} · ${r.joints} dof`; sel.appendChild(o);
    }
    sel.value = "so101";
  }
  lockoutLine((await api("/api/safety")).lockout);
  const { sessions } = await api("/api/sim");
  const box = $("#sessions");
  for (const el of [...box.children]) if (!sessions.some(s => s.id === el.dataset.id)) { sockets.get(el.dataset.id)?.close(); sockets.delete(el.dataset.id); el.remove(); }
  for (const s of sessions) if (!box.querySelector(`[data-id="${s.id}"]`)) mountSession(s);
  if (!sessions.length) box.innerHTML = '<p class="muted">No session yet. Pick a robot and press Start - it steps in this process and streams here.</p>';
  else box.querySelector("p.muted")?.remove();
}

function mountSession(s) {
  const el = document.createElement("article"); el.className = "session"; el.dataset.id = s.id;
  el.innerHTML = `<div class="head"><span class="name">${s.robot}</span><span class="pill mono">${s.id}</span><span class="pill state">${s.state}</span></div>
    <div class="view"><img alt="${s.robot} camera" src="/api/sim/${s.id}/stream.mjpg"></div>
    <div class="joints">${s.joint_names.map(n => `<div class="joint"><span class="label">${n}</span><span class="val">0.000</span><span class="bar"><i></i></span></div>`).join("")}</div>
    <div class="foot"><span class="t">t=0.00s</span><span class="fps"></span><button data-act="reset">Reset</button><button data-act="stop">Stop</button></div>`;
  $("#sessions").appendChild(el);
  el.querySelector('[data-act="stop"]').onclick = async () => { await api(`/api/sim/${s.id}`, { method: "DELETE" }); loadSim(); };
  el.querySelector('[data-act="reset"]').onclick = async () => { try { await api(`/api/sim/${s.id}/reset`, { method: "POST" }); } catch (e) { lockoutLine({ state: "locked", reason: e.message }); } };
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/telemetry/${s.id}`);
  sockets.set(s.id, ws);
  const vals = el.querySelectorAll(".joint .val"), bars = el.querySelectorAll(".joint .bar i");
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    el.classList.toggle("frozen", m.state === "frozen");
    el.querySelector(".state").textContent = m.state;
    el.querySelector(".t").textContent = `t=${m.sim_time.toFixed(2)}s`;
    el.querySelector(".fps").textContent = m.fps ? `${m.fps} fps` : "";
    m.qpos.forEach((q, i) => { if (vals[i]) { vals[i].textContent = q.toFixed(3); bars[i].style.transform = `translateX(${Math.max(-1, Math.min(1, q / Math.PI)) * 40}px)`; } });
    lockoutLine(m.lockout);
    if (m.state === "stopped" || m.state === "error") ws.close();
  };
}

$("#sim-new").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try { await api("/api/sim", { method: "POST", body: JSON.stringify({ robot: $("#sim-robot").value }) }); await loadSim(); }
  catch (e) { lockoutLine({ state: "locked", reason: e.message }); }
});
$("#estop").addEventListener("click", async () => {
  const l = $("#lockout").className.includes("locked");
  const r = await api(l ? "/api/safety/resume" : "/api/safety/estop", { method: "POST" });
  lockoutLine(r.lockout); $("#estop").textContent = r.lockout.state === "locked" ? "RESUME" : "E-STOP";
});

/* ---- ceremonies ---- */
$("#enrol").addEventListener("submit", async (ev) => {
  ev.preventDefault(); $("#login-error").textContent = "";
  const fd = new FormData(ev.target);
  try {
    const begin = await api("/api/auth/register/begin", { method: "POST", body: JSON.stringify({ bootstrap: fd.get("bootstrap"), label: fd.get("label") }) });
    const cred = await navigator.credentials.create({ publicKey: toCreationOptions(begin.options) });
    await api("/api/auth/register/finish", { method: "POST", body: JSON.stringify({ challenge_id: begin.challenge_id, credential: serializeCredential(cred) }) });
    await boot();
  } catch (e) { $("#login-error").textContent = e.message; }
});
$("#login").addEventListener("click", async () => {
  $("#login-error").textContent = "";
  try {
    const begin = await api("/api/auth/login/begin", { method: "POST" });
    const cred = await navigator.credentials.get({ publicKey: toRequestOptions(begin.options) });
    await api("/api/auth/login/finish", { method: "POST", body: JSON.stringify({ challenge_id: begin.challenge_id, credential: serializeCredential(cred) }) });
    await boot();
  } catch (e) { $("#login-error").textContent = e.message; }
});

$("#tabs").addEventListener("click", (ev) => {
  const v = ev.target.dataset.view; if (!v) return;
  if (!authenticated) return showLogin();
  show(v);
  if (v === "fleet") loadFleet();
  if (v === "sim") loadSim();
  if (v === "settings") loadSettings();
});
$("#settings-form").addEventListener("submit", saveSettings);

async function boot() {
  const h = await fetch("/api/health").then(r => r.json()).catch(() => ({}));
  $("#version").textContent = h.version ? `strands-robots ${h.version}` : "";
  await loadStatus();
  if (!authenticated) return showLogin();
  await loadWho();
  const v = views.includes(location.hash.slice(1)) ? location.hash.slice(1) : "fleet";
  show(v);
  if (v === "fleet") loadFleet();
  if (v === "sim") loadSim();
  if (v === "settings") loadSettings();
}
boot();
