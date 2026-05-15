# PLAN.md — Zenoh Mesh E2E Implementation for `strands-labs/robots`

**Branch:** `autonomous/mesh-session` (fork: `cagataycali/robots`)
**Target:** `strands-labs/robots:main`
**Reference implementation:** `strands-labs/robots:dev` (~741 LoC `zenoh_mesh.py`, ~257 LoC `robot_mesh.py` tool)
**Open PR:** https://github.com/strands-labs/robots/pull/101 — *"feat(mesh): add session singleton and peer registry"*

---

## 0. Why this exists

`dev` branch already proves out a working monolithic `zenoh_mesh.py` where every
`Robot()` and `Simulation()` is automatically a Zenoh peer. We're landing the
same capability on `main` but **decomposed into 6 small, reviewable PRs** so
each piece can be reviewed, tested, and reverted independently.

**Design contract (carried over from dev):**
- One `zenoh.Session` per process, ref-counted across `N` robots
- Each `Robot()` / `Simulation()` owns a `Mesh` component (composition, not inheritance)
- Auto-mesh: first process listens on `tcp/127.0.0.1:7447`, others fall back to client mode
- LAN auto-discovery via Zenoh multicast scouting (zero config)
- Cross-compatible with Reachy Mini's Zenoh schema
- Lazy zenoh import — `import strands_robots` never pays the zenoh cost
- Global kill switch: `STRANDS_MESH=false`
- Per-robot opt-out: `Robot("so100", mesh=False)`

**Topic schema (committed):**
```
strands/{peer_id}/presence       — 2 Hz heartbeat
strands/{peer_id}/state          — 10 Hz joint/sim state
strands/{peer_id}/cmd            — incoming RPC
strands/{peer_id}/response/{turn}— RPC reply
strands/{peer_id}/stream         — VLA execution steps
strands/broadcast                — fan-out RPC
```

---

## 1. Status of branch `autonomous/mesh-session` (now)

Rebased cleanly onto `strands-labs/robots:main` (commit `c077b7a`).
3 commits on top:
- `feat(mesh): add session singleton and peer registry` — `mesh_session.py` (349 LoC)
- `fix(ci): correct generator return types in mesh session test fixtures`
- `fix(ci): sort imports per ruff I001 — move Iterator import after bare imports`

**Files landed (PR1):**
| File | LoC | Purpose |
|------|-----|---------|
| `strands_robots/mesh_session.py` | 349 | Session singleton + `PeerInfo` + peer registry + `put()` |
| `tests/mesh/test_mesh_session.py` | 447 | 29 tests, 90% coverage, fully mocked |
| `tests/mesh/__init__.py` | 1 | Package marker |
| `pyproject.toml` | +5 | `[mesh]` extra → `eclipse-zenoh>=1.0.0,<2.0.0` |

**Public surface of `mesh_session.py` (use these in PR2+):**
```python
PeerInfo                                 # dataclass
update_peer / prune_peers / get_peers / get_peer / peer_count / clear_peers
get_session / release_session / session_alive
put(key, data)                           # safe no-op when session is None
```

**What `dev` has but `main` does NOT yet have:**
- The `Mesh` class (lifecycle, presence loop, state loop, RPC, subscribe, publish_step, on_stream, send/broadcast/tell/emergency_stop)
- The `init_mesh()` factory used inside `Robot.__init__` / `Simulation.__init__`
- `_LOCAL_ROBOTS` registry of in-process meshes
- Wiring inside `simulation.py` and `robot.py` / `factory.py`
- The `robot_mesh` agent-facing tool
- Integration tests with a real zenoh router

---

## 2. The 6-PR roadmap (this is the whole project)

| # | PR title | Branch (off `main`) | LoC | Status |
|---|----------|---------------------|-----|--------|
| 1 | feat(mesh): session singleton + peer registry | `autonomous/mesh-session` | ~800 | ✅ open as #101, rebased |
| 2 | feat(mesh): Mesh class — presence + peer discovery | `mesh/02-presence` | ~500 | ⏳ |
| 3 | feat(mesh): tell/send/broadcast + response correlation | `mesh/03-rpc` | ~400 | ⏳ |
| 4 | feat(mesh): publish_step + subscribe + on_stream | `mesh/04-streams` | ~350 | ⏳ |
| 5 | feat(mesh): emergency_stop + safety audit log | `mesh/05-safety` | ~250 | ⏳ |
| 6 | feat(mesh): wire mesh into Simulation/Robot/factory + `robot_mesh` tool | `mesh/06-wiring` | ~400 | ⏳ |

Each PR is **mergeable on its own**. PR2 lands the `Mesh` class API but nothing calls it. PR6 is what users actually see.

---

## 3. PR-by-PR breakdown

### PR1 — Session singleton + peer registry — ✅ #101

**Already shipped on this branch.** Outstanding work:
- [ ] Address review comments when they arrive on #101
- [ ] Confirm `pyproject.toml` rebase produced the right merge for `[mesh]` extra and mypy `zenoh.*` override (done — verify with `git diff main -- pyproject.toml`)
- [ ] Run `hatch run lint && hatch run test` clean before requesting review
- [ ] Squash commits on merge

**Acceptance:**
- 29 tests pass without `eclipse-zenoh` installed (all mocked)
- `python -c "from strands_robots.mesh_session import get_session; print(get_session())"` returns `None` when zenoh missing
- `pip install -e ".[mesh]"` then same command opens a real session, `release_session()` closes it

---

### PR2 — `Mesh` class: presence + peer discovery

**New file:** `strands_robots/mesh.py` (~400 LoC)
**Builds on:** `mesh_session.py` (PR1)

**API (mirrors dev `zenoh_mesh.Mesh` minus RPC/stream which land in PR3/4):**
```python
class Mesh:
    def __init__(self, robot, peer_id: str, peer_type: str = "robot")
    def start(self) -> None
    def stop(self) -> None
    @property
    def alive(self) -> bool
    @property
    def peers(self) -> list[dict]   # delegates to mesh_session.get_peers()

    # Presence — 2 Hz heartbeat thread
    def _build_presence(self) -> dict
    def _heartbeat_loop(self) -> None
    def _on_presence(self, sample) -> None

    # State — 10 Hz publish thread
    def _state_loop(self) -> None
    def _read_state(self) -> dict | None

def init_mesh(robot, peer_id=None, peer_type="robot", mesh=True) -> Mesh | None
```

**Module-level state:**
```python
_LOCAL_ROBOTS: dict[str, Mesh] = {}     # in-process registry
HEARTBEAT_HZ = 2.0
STATE_HZ = 10.0
```

**Implementation notes:**
- Threads must be `daemon=True` so they don't block process exit
- `start()` must be idempotent (`if self._running: return`)
- `stop()` must release the session ref *only if* `start()` actually acquired one (track `_has_session_ref`)
- Presence payload reflects whatever attributes the `robot` exposes (duck-typed; wrap in `try/except Exception: pass`)
- State publishing skips images/tensors with `>1` dim — only numeric

**Tests (`tests/mesh/test_mesh.py`):**
- Lifecycle: start → alive → stop → not alive
- Idempotency: start twice = no error, one heartbeat thread
- Two `Mesh` instances in same process share one session (refcount = 2)
- Mocked subscriber receives presence → `update_peer` called
- Stale peer pruning runs on heartbeat tick

**Acceptance:**
- Two python processes with `from strands_robots.mesh import init_mesh; m = init_mesh(MockRobot(), peer_id='a')` discover each other within 1 second
- `m.peers` returns the other peer
- `kill -9` one process → other process prunes within `PEER_TIMEOUT` (10 s)

---

### PR3 — `tell` / `send` / `broadcast` + response correlation

**Adds to:** `strands_robots/mesh.py`

**API:**
```python
class Mesh:
    # RPC — outgoing
    def send(self, target: str, cmd: dict, timeout: float = 30.0) -> dict
    def broadcast(self, cmd: dict, timeout: float = 5.0) -> list[dict]
    def tell(self, target: str, instruction: str, **kw) -> dict

    # RPC — incoming dispatch (private)
    def _on_cmd(self, sample) -> None
    def _exec_cmd(self, data: dict) -> None
    def _dispatch(self, cmd: dict) -> dict
    def _on_response(self, sample) -> None
```

**Correlation protocol:**
- Sender generates `turn_id = uuid.uuid4().hex[:8]`
- Sender holds `threading.Event` in `self._pending[turn_id]`
- Responder publishes to `strands/{sender_id}/response/{turn_id}`
- Sender waits up to `timeout`, collects `self._responses[turn_id]`, then pops both dicts

**Dispatched actions (must match dev behaviour):**
- `status` → `robot.get_task_status()` if available
- `stop` → `robot.stop_task()`
- `features` → `robot.get_features()`
- `state` → `self._read_state()`
- `execute` / `start` → `robot._execute_task_sync(...)` / `robot.start_task(...)`
- `step` → `robot.step(steps)`
- `reset` → `robot.reset()`

**Tests:**
- `send` with mocked subscriber returning a response → returns the response dict
- `send` timeout → returns `{"status": "timeout"}`
- `broadcast` collects 3 responses from 3 mocked peers
- `_dispatch` routes correctly for each known action; unknown action returns `{"error": ...}`
- Self-loop guard: `_on_cmd` ignores messages where `sender_id == self.peer_id`

**Acceptance:** end-to-end demo in `tests_integ/test_mesh_rpc.py`:
```
two procs ↔ one zenoh router →
  proc A: m.send("peer-b", {"action": "status"}) → {"status": ..., "peer_id": "peer-b"}
  proc A: m.broadcast({"action": "status"}) returns >= 1 response
```

---

### PR4 — `publish_step` + `subscribe` + `on_stream`

**Adds to:** `strands_robots/mesh.py`

**API:**
```python
class Mesh:
    def subscribe(self, topic: str, callback=None, name: str = None) -> str | None
    def unsubscribe(self, name: str) -> None
    def publish_step(self, step: int, observation: dict, action: dict,
                     instruction: str = "", policy: str = "") -> None
    def on_stream(self, peer_id: str, callback=None) -> str | None

    # Buffered messages for non-callback subscribers
    inbox: dict[str, list[tuple[str, dict]]]
```

**Implementation notes:**
- `subscribe` accepts wildcards: `reachy_mini/*`, `*/joint_positions`, `strands/*/state`
- When `callback` is `None`, append `(key, data)` tuples to `self.inbox[name]`
- Cap each inbox list at 1000 entries (slice to last 500 when overflowing)
- `publish_step` must filter out tensors/arrays with `>1` dim (camera frames)
- All numpy arrays converted via `.tolist()`

**Tests:**
- `subscribe` with callback → handler fires on mocked sample
- `subscribe` without callback → messages buffered in `inbox[name]`
- `inbox` overflow caps at 1000, slices to 500
- `publish_step` filters camera frames (3D arrays) but keeps joint_positions (1D)
- `on_stream("peer-b")` subscribes to `strands/peer-b/stream`

**Acceptance:** integ test reproduces dev branch behaviour:
```
proc A starts a Robot with policy execution → publish_step at every loop iter
proc B: m.on_stream("peer-a") → inbox fills with step dicts
```

---

### PR5 — `emergency_stop` + safety audit log

**Adds to:** `strands_robots/mesh.py` + new `strands_robots/mesh_audit.py`

**API:**
```python
class Mesh:
    def emergency_stop(self) -> list[dict]   # broadcast({"action": "stop"}, timeout=3.0)

# mesh_audit.py — append-only JSONL log
def log_safety_event(event_type: str, peer_id: str, payload: dict) -> None
def read_audit_log(since: float | None = None) -> list[dict]
```

**Behaviour:**
- `emergency_stop` issues `broadcast({"action": "stop"}, timeout=3.0)`
- Every E-STOP gets logged to `~/.strands_robots/mesh_audit.jsonl`
- Log entry: `{"ts", "event", "peer_id", "sender_id", "responses_received"}`
- Audit log directory created with mode `0o700`
- Reading the log respects an optional `since` epoch filter

**Why a separate file:** safety actions need a tamper-evident trail independent of stdout/loguru. Tests must verify the file is `chmod 0600`.

**Tests:**
- `emergency_stop` calls `broadcast` with `{"action": "stop"}` and timeout 3.0
- Audit log file is created with `0o600` permissions
- Each E-STOP appends one line with required keys
- `read_audit_log(since=t)` filters correctly

---

### PR6 — Wire mesh into `Simulation`, `Robot`, `factory`, plus `robot_mesh` tool

**Files touched:**
- `strands_robots/simulation/simulation.py` — `__init__` accepts `mesh: bool, peer_id: str`, calls `init_mesh(self, peer_type="sim", mesh=mesh)`, `cleanup()` calls `self.mesh.stop()`
- `strands_robots/robot.py` (factory) and `strands_robots/hardware_robot.py` — same pattern, `peer_type="robot"`
- `strands_robots/factory.py` (or whatever the dev branch calls the merged Robot/Simulation factory) — pass `mesh` and `peer_id` through to backend
- **New file:** `strands_robots/tools/robot_mesh.py` (~257 LoC, port of dev branch)
- `strands_robots/tools/__init__.py` — export `robot_mesh`
- `strands_robots/__init__.py` — lazy import of `robot_mesh` in `__getattr__`

**The `robot_mesh` tool actions (from dev):**
`peers`, `tell`, `send`, `broadcast`, `stop`, `emergency_stop`, `status`,
`subscribe`, `watch`, `inbox`.

**Tests:**
- `Robot("so100", mesh=False)` → `robot.mesh is None`, no zenoh import attempted
- `Robot("so100")` → `robot.mesh.alive is True`
- Two `Robot()` instances in same process share one session (refcount=2)
- `Robot()` then `del robot` → session refcount goes back to 0, session closed
- `robot_mesh(action="peers")` returns local + remote
- `STRANDS_MESH=false` env var → `robot.mesh is None` even with `mesh=True` arg

**Integration tests (`tests_integ/test_mesh_e2e.py`, requires `[mesh]` extra):**
- Spawn 2 subprocesses, each creates a `Simulation`, verify mutual discovery
- Subprocess A `tell`s subprocess B to step the sim → B steps and replies
- Subprocess A `emergency_stop` → both sims stop
- Audit log on both processes has the E-STOP event

**Documentation:**
- README.md: new "Mesh networking" section
- AGENTS.md: append a "Mesh conventions" subsection (peer_id format, topic schema, env vars)
- One example notebook: `examples/mesh_two_robots.ipynb`

**Acceptance for the whole feature:**
- `pip install strands-robots[mesh]` is the only step needed
- Demo in README:
  ```python
  from strands_robots import Robot
  sim_a = Robot("so100")
  sim_b = Robot("so100")  # second process
  print(sim_a.mesh.peers)            # sees sim_b
  sim_a.mesh.tell(sim_b.mesh.peer_id, "pick up the cube")
  ```

---

## 4. Cross-cutting requirements

### Type hints & mypy
- Use `from __future__ import annotations` everywhere
- Prefer modern syntax: `list[X]`, `dict[K, V]`, `X | None`
- `zenoh.*` is in mypy `ignore_missing_imports`
- Every public function gets a complete signature
- Every `try/except Exception: pass` needs a comment explaining why

### Error handling (per AGENTS.md review learnings)
- Tool dispatch returns `{"status": "error", "content": [...]}`, never raises
- No silent catch-all `except Exception` outside of best-effort callbacks (mesh handlers); when used, comment why
- On partial init failure (`init_mesh` raises mid-way), call `release_session()` to avoid leaks
- All thread `target` functions wrap their bodies so a thrown exception doesn't kill the loop silently — log + continue

### Concurrency
- All access to shared dicts (`_PEERS`, `_LOCAL_ROBOTS`) goes through the existing locks in `mesh_session.py`
- `Mesh._pending` and `Mesh._responses` are accessed from both the response subscriber thread and the calling thread — both reads/writes need locking; use a single `self._rpc_lock`
- `Mesh.start()` must not race with `Mesh.stop()` — guard with `self._lifecycle_lock`

### Performance
- Don't create thread pools per call — reuse `self._executor` (already a pattern in `simulation.py`)
- Cache JSON-encoded presence payload between heartbeats if the robot state hasn't changed (optimisation, not blocker)
- `prune_peers` runs once per heartbeat tick (every 0.5 s), not on every `get_peers()` call

### No host paths, no emojis
- Strings returned from tools and log messages: ASCII only (per project rule)
- The dev branch uses emojis (`🔗 🤖 🛑 📡`) — strip these when porting; replace with ASCII tokens (`[mesh]`, `[peer]`, `[stop]`, `[sub]`)
- Test files: never commit `/Users/cagatay/...`; use `tmp_path` fixture
- `tests/test_no_host_paths.py` enforces this — keep it green

### Testing
- Unit tests: 100% mocked, no real `zenoh` import. Use `monkeypatch` to inject a fake session
- Integration tests: under `tests_integ/`, gated by `pytest.importorskip("zenoh")`. Run with `hatch run test-integ`
- Every reviewed fix from PRs 2–6 gets a regression test
- Use `monkeypatch.setenv` for env vars, never `os.environ[k] = v`

### Linting
- `hatch run format` before every push
- `ruff check`, `ruff format --check`, `mypy strands_robots/` all clean
- `hatch run lint` is the gate

---

## 5. Open questions / decisions to confirm

1. **File name**: dev calls it `zenoh_mesh.py`; PR1 split it into `mesh_session.py`. Should the `Mesh` class live in `strands_robots/mesh.py` or `strands_robots/mesh/__init__.py`? — Decision: **`strands_robots/mesh.py`** (one module, ~1000 LoC after PRs 2–5 is fine; matches `simulation.py` pattern).

2. **Tool location**: dev has `tools/robot_mesh.py`. Keep that exact path. ✅

3. **Backwards compat**: `from strands_robots.zenoh_mesh import init_mesh` is used by some downstream code. Add a thin shim `strands_robots/zenoh_mesh.py` that re-exports from `mesh_session` + `mesh` for one minor version, then remove. **Document the deprecation in PR6.**

4. **Default `mesh=True`** in `Robot()`/`Simulation()` — opinion: **yes**, matches dev. Disable via env or kwarg. CI should run with `STRANDS_MESH=false` to keep unit-test isolation.

5. **`peer_id` format**: dev uses `f"{tool_name}-{uuid.uuid4().hex[:4]}"`. Keep. Document in AGENTS.md.

6. **Audit log location**: `~/.strands_robots/mesh_audit.jsonl`. Add `STRANDS_MESH_AUDIT_DIR` env override.

---

## 6. Definition of Done (whole feature)

- [ ] All 6 PRs merged into `main`
- [ ] `strands-robots[mesh]` installable from PyPI
- [ ] README has a working two-process mesh example
- [ ] `tests_integ/test_mesh_e2e.py` runs in CI on `ubuntu-latest` with `eclipse-zenoh` installed
- [ ] `robot_mesh` tool is registered and visible in `strands_robots.tools.__all__`
- [ ] `STRANDS_MESH`, `STRANDS_MESH_PORT`, `ZENOH_CONNECT`, `ZENOH_LISTEN`, `STRANDS_MESH_AUDIT_DIR` documented in README
- [ ] AGENTS.md has a "Mesh conventions" section with the topic schema
- [ ] One project-board issue per PR exists on https://github.com/orgs/strands-labs/projects/2 with Status + Priority

---

## 7. What I'm doing right now (autonomous execution log)

I'm running this PLAN.md autonomously via DevDuck ambient mode. Each cycle picks
the next un-checked item, makes progress, runs lint+tests, commits with a
`feat(mesh): ...` or `test(mesh): ...` prefix, and updates the checklist below.

### Working checklist (live)

**PR1 cleanup:**
- [x] Rebase `autonomous/mesh-session` onto `strands-labs/robots:main` (resolved pyproject.toml conflicts: kept both `benchmark-libero` and `mesh` extras; merged mypy module list to include both `libero.*` and `zenoh.*`)
- [ ] `hatch run format && hatch run lint && hatch run test` clean
- [ ] Force-push the rebased branch (will require PR re-review approval)
- [ ] Verify `pyproject.toml` diff vs `main` only adds the `[mesh]` extra and the `zenoh.*` mypy override

**PR2 prep:**
- [ ] Sketch `strands_robots/mesh.py` skeleton — `Mesh` class + `init_mesh()`
- [ ] Port presence loop from `/tmp/dev_zenoh_mesh.py` (lines 220–340)
- [ ] Port state loop (lines 340–400)
- [ ] Add tests in `tests/mesh/test_mesh.py`

**PR3+ prep:** see breakdown above.

