# Lane mesh - HANDOFF

Branch `dash/lane-mesh`, worktree `/Users/cagatay/robots-dash/mesh`, 3 commits on top of `1a0123e`.

## Files added (all under `strands_robots/dashboard/`)

| file | LOC | from |
|------|-----|------|
| `routes_mesh.py` | ~640 | port of branch `server.py` lines 452-1949 (mesh half) onto main's rails |
| `peer_tools.py` | 549 | `fork/dashboard-revamp` verbatim, archaeology stripped; `SIM_CALL_BLOCKED` is now a local literal (see gaps) |
| `teleop_health.py` | 210 | branch, verbatim |
| `joint_silence.py` | 277 | branch, verbatim + module docstring |
| `lan_hint.py` | 99 | branch, verbatim |
| `refusals.py` | 101 | branch, verbatim + docstrings |
| `ws_observability.py` | 105 | branch, verbatim |
| `health_ingest.py` | 95 | branch, verbatim, archaeology stripped |

Import clean (`python -c "import strands_robots.dashboard.routes_mesh"`), `ruff check` + `ruff format` clean,
mypy clean on `routes_mesh.py` + `peer_tools.py` (uvx mypy, `--ignore-missing-imports`).

## What routes_mesh exposes

```python
from strands_robots.dashboard import routes_mesh
app.include_router(routes_mesh.router)   # APIRouter(prefix="/api", tags=["mesh"]), 18 HTTP routes
routes_mesh.attach(app)                  # state + startup/shutdown hooks + mounts routes_mesh.ws_router
```

`attach(app)`:
- `app.state.bridge = MeshBridge()` if absent (an existing one is kept), `app.state.mesh_online = False`,
  `app.state.refusals = RefusalTally()`, `app.state.camera_close_log = CloseLogThrottle()`,
  `app.state.mesh_ingest_prev = None`
- appends to `app.state.startup_hooks` / `shutdown_hooks` (lists, created if absent) two async callables:
  `mesh_online = await to_thread(bridge.start, loop)` and `await to_thread(bridge.stop)`
- includes `ws_router` (the two websockets have no `/api` prefix so they cannot live on `router`);
  guarded by `app.state.mesh_ws_mounted` so a second `attach` is a no-op.

Routes (every HTTP route `Depends(access.require_session)`; websockets run `access.caller(ws)` and
`access.refuse_socket(ws, 4401)` exactly like `/ws/agent`):

| route | notes |
|-------|-------|
| `GET /api/fleet` | `bridge.snapshot()` (type/peers/mesh/dashboard_peer_id/managed_no_presence/absent_children/t) + `mesh_online`, `peer_count`, `mesh_coalesce`, `mesh_ingest` (health_ingest.mesh_ingest), `joint_streams` when silent_arms says so |
| `GET /api/network/hint` | lan_hint.hint; psutil optional; port from `app.state.port` else 8090 |
| `GET /api/robots/registry` | `registry.list_robots()` in a thread |
| `GET /api/activity?limit=` | bridge.activity_log, 1..300 |
| `GET /api/robots/{pid}/teleop` | teleop_health + published_frames + consent.classify_refusal on the child log (log tail read from `app.state.devices.robots[pid].logs` when the devices lane is mounted) |
| `POST .../teleop/publish`, `/teleop/receive`, `/teleop/stop` | verbatim command shapes; 30/45/10 s budgets |
| `POST /api/robots/{pid}/task` | see "Motion gate" below; WIRE_CMD_KEYS allowlist, route_task_target, task_ack_budget/timeout_verdict, twin mirroring, consent.attach_consent on a refused result |
| `POST /api/robots/{pid}/stop` | never gated; stop_outcome |
| `POST /api/robots/{pid}/twin` | needs `app.state.devices` (devices lane); 503 otherwise |
| `GET /api/robots/{pid}/policy-fit` | imports `checkpoints.declared_features` + `policy_fit.policy_fit` from lane train at call time |
| `POST /api/mesh/safety/estop` | per-peer stop fan-out + `bridge.signed_estop()`; payload: targeted, stale_skipped, counts, all_stopped, stopped, signed_rail, responses_received, peers_not_stopped, lockout_engaged, **lockout** (= `bridge._lockout.as_fields()`, safety_state.Lockout) |
| `POST /api/mesh/safety/resume` | `{override_code}` -> `bridge.signed_resume` + `lockout` |
| `GET/POST /api/mesh/config` | GET = `bridge.mesh_info()`; POST = `settings.update_strict({"mesh": {...}})` for connect/listen/port/backend/camera_hz/policy_type_allow, then `_restart_mesh` unless `restart:false`; 409 while managed robots hold the session unless `force:true`; a loopback-via caller must be a loopback peer (same rule as `/api/settings`) |
| `POST /api/mesh/restart` | `{force}` |
| `GET /api/frame/{pid}/{cam}` | 404 no frame, 415 undecodable |
| `WS /ws/mesh` | snapshot then attach_queue fan-out; detach on close |
| `WS /ws/camera/{pid}/{cam}?max_fps=` | 15 fps pacing, fps_cap, one `camera_error` text frame per distinct problem, CloseLogThrottle close line; ChurnGuard is consulted only when `app.state.camera_churn` exists (devices lane sets it; imports `churn_guard.viewer_identity` lazily) |

POST bodies go through `routes_auth._json_body` (must be `application/json`, like every write on main).

### Motion gate (`task_gate`)

Fail closed. A task on a peer `agent_motion.peer_is_physical` calls metal is sent only when one holds:
1. body `confirmed` is the JSON boolean `true` (the browser's play button); a non-boolean is refused
   via `utils.boolean_flag_error`, so `"confirmed": "false"` cannot select the confirmed posture;
2. `STRANDS_DASH_AGENT_PHYSICAL_MOTION` grants unattended motion AND `STRANDS_DASH_TASK_REQUIRES_CONFIRM`
   is not set.

Otherwise 403 with `{"error": {"error": <reason>, "peer_id", "ok": false, "verdict", "needs_consent": {...}}}`;
`needs_consent.kind == "agent_physical_motion"` (consent.classify_refusal on the agent_motion verdict),
so the frontend's consent card can grant it. Sim peers are never gated; stop is never gated.

## Verified (TestClient smoke, not committed)

`STRANDS_MESH=false`, `FastAPI(); include_router(router); attach(app)`, startup hooks run on the client loop:
- `GET /api/fleet` 200, `mesh_online: false`, snapshot keys present
- `GET /api/activity` 200 `{"activity": []}`; `/api/network/hint` 200; `/api/mesh/config` 200; `/api/robots/registry` 200 (76 rows)
- `GET /api/frame/x/cam` 404; `POST /api/robots/ghost/stop` 404 with `known_peers`
- physical peer, no confirm -> 403 `needs_consent.kind=agent_physical_motion`; `"confirmed":"true"` -> 403; `confirmed:true` -> 200 with `{"error":"mesh offline","ok":false}` (mesh is off in the smoke)
- `POST /api/mesh/safety/estop` 200 with `lockout_engaged false`, `responses_received 0`, `peers_not_stopped []`, `lockout.state unknown`; `/resume` 422 without code, 200 with the rail-disabled error + lockout
- `/ws/mesh` accepts, sends the snapshot, closes; `/ws/camera/arm1/front?max_fps=5` accepts and closes with the throttled close line
- with `STRANDS_DASH_AUTH_ENABLED=1`: `POST /api/mesh/safety/estop` 401, `GET /api/fleet` 401, `/ws/mesh` closed 4401

## Deps

Nothing new: fastapi + what main's dashboard already needs. `psutil` optional (network hint degrades).
`strands` (for `peer_tools._agent_tool_base`) is imported lazily.

## Gaps

- `peer_tools.py` is ported but NOT wired: main replaced `agent_bridge` with `agent_console`, and the
  KIND_SIM proxies route over a `sim_call` mesh action that does not exist in main's `mesh/security.py`
  (no `SIM_CALL_BLOCKED_ACTIONS` there either; the list is a local literal now). Wiring the per-peer
  proxies into `agent_console` and/or adding the `sim_call` rail is a coordinator decision.
- `joint_silence.merge` is ported; `mesh_bridge.snapshot()` still does the plain overlay
  (`peers[pid] = {**peer, **fields}`) its own comment says to upgrade to `joint_silence.merge(peer, fields)`.
  `mesh_bridge.py` is not mine to edit.
- `RefusalTally` sits on `app.state.refusals` but nothing records into it: the branch fed it from its
  `TokenAuthMiddleware`, which main does not have (access is a dependency). A `/api/health`
  `refused_handshakes` block would need a hook in `access.caller` or the 401 handler (coordinator).
- The twin task mirror keeps its `asyncio.Task` on `app.state.twin_task` so it is not garbage-collected mid-flight.

## Needs coordinator

1. **`/api/fleet` exists twice**: `fleet.py` (main, registry + `mesh.session.get_peers`) and `routes_mesh`
   (bridge-backed, the shape the React frontend reads). Register `routes_mesh.router` BEFORE `fleet.router`
   or drop `fleet.py`'s `/fleet` (its `/robots/{name}` does not collide: `/robots/registry` is a literal
   path and FastAPI ranks it first only if `routes_mesh` is included first - so include routes_mesh first).
2. Call `routes_mesh.attach(app)` in `create_app` and run `app.state.startup_hooks` / `shutdown_hooks` in
   `_lifespan` (before/after the existing `safety.store.shutdown()`).
3. Set `app.state.port` in `cli.py` so `/api/network/hint` names the real LAN URL.
4. When the devices lane lands: set `app.state.devices`, `app.state.camera_churn = ChurnGuard()` and the
   three bridge hooks (`protected_peer_ids`, `peer_annotations`, `managed_children`) the branch wired in
   `create_app` lines 391-397 - `routes_mesh` reads them all via `getattr` and degrades without them.
5. Frontend: safety buttons must call `/api/mesh/safety/estop|resume` (not `/api/safety/*`); the task
   refusal is nested under `error` (main's HTTPException handler shape).
