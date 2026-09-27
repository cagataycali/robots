# Lane devices - HANDOFF

Branch `dash/lane-devices`, worktree `/Users/cagatay/robots-dash/devices`, 2 commits on top of
`feat/dashboard-complete-20260927 @ 1a0123e`:

- `b0baeed6f` dashboard(devices): port the device roster modules from the revamp branch
- `1fb080f84` dashboard(devices): /api/devices router + attach() with autospawn lifespan hooks

## Files added (all under `strands_robots/dashboard/`)

| file                 | LOC   | ported from `fork/dashboard-revamp` | changes vs the branch |
|----------------------|-------|--------------------------------------|-----------------------|
| `routes_devices.py`  | ~330  | server.py lines 1417-1640 + 280-300 (autospawn audit) + 418-446 (startup/shutdown) | rewritten as `APIRouter(prefix="/api", tags=["devices"])`, state via `request.app.state`, auth via `Depends(access.require_session)`, bridge optional |
| `device_manager.py`  | 2,090 | whole                                | `joint_silence` (mesh lane) imported lazily inside `annotations_by_peer`; docstrings for main's D-rules; `_camera_fault` keeps the classifier's reason when stderr is non-empty; em dash -> `-` |
| `cameras.py`         | 187   | whole                                | module docstring; `classify_probe_stderr` reads AVFoundation's `out device of bound` as `absent` (the branch's comment promised this; before, a missing index came back `unreadable`/503) |
| `camera_liveness.py` | 225   | whole                                | none |
| `direct_serial.py`   | 190   | whole                                | none |
| `bus_claim.py`       | 68    | whole                                | dropped a `BUGS.md Q84` cite from the refusal text |
| `churn_guard.py`     | 88    | whole                                | one docstring |
| `arm_roles.py`       | 97    | whole                                | module docstring |
| `argv_exposure.py`   | 39    | whole                                | none |
| `build_info.py`      | 68    | whole                                | em dash -> `-` |
| `disk_headroom.py`   | 91    | whole                                | none |

`DeviceManager` public API kept verbatim for lanes mesh and record: `robots`,
`annotations_by_peer`, `managed_children`, `collect`, `replay`, `start_autospawn`, `shutdown`,
`spawn`, `despawn`, `logs`, `reconfigure_cameras`, `settle`, `devices`, `profiles`,
`profile_for_port`, `preview_frame`, `probe_modes`, `measure_arm_role`. Module-level helpers
`validate_spawn`, `validate_cameras`, `validate_replay`, `respawn_payload`, `remembered_spawn`,
`scan_serial_ports`, `diagnose_camera_indices`, `AUTOSPAWN_POLL_S`, `SPAWN_SETTLE_S` unchanged.

## Routes served (router prefix `/api`)

| route                                   | verified on this Mac (TestClient, bearer token from `DASHBOARD_AUTH_TOKEN`) |
|-----------------------------------------|------------------------------------------------------------------------------|
| `GET /devices?refresh=`                 | 200: `serial_ports: []`, 4 camera indices (FaceTime `ready`, 3 `absent`), `camera_names`, `managed`, `camera_problem` |
| `GET /devices/profiles`                 | 200: `{profiles, path, autospawn}` (one remembered profile on this machine)  |
| `GET /devices/arm-role?port=&model=`    | 200 with `role: unknown` + remedy for a port that does not exist (bus probe ran, no servo answered) |
| `GET /devices/camera/{i}/preview`       | index 47 -> 404 `{error, index, state: absent, reason, remedy}`; index 0 delivers a JPEG when nothing streams it |
| `GET /devices/camera/{i}/modes`         | index 47 -> 404 with the same body                                           |
| `POST /devices/spawn`                   | no auth -> 401; `mode=auto` -> 422; unknown robot -> 422 (SDK's message); `mode=real` without port -> 422; `so101 sim` -> 200 `{peer_id, pid, mode, status: starting, waited_s: 5.0}` (child alive; `starting` not `running` because the venv has no zenoh, so the peer never joins a mesh - honest) |
| `POST /devices/spawn-remembered`        | unknown port -> 404 with the branch's operator text                          |
| `POST /devices/despawn`                 | unknown peer -> 404; spawned peer -> 200 `{stopped: true}`                   |
| `POST /devices/{peer}/cameras`          | `cameras: 5` -> 422; bad per-camera config -> 422 (validation runs before the peer lookup, as on the branch) |
| `GET /devices/logs/{peer}`              | unknown -> 404 `{error, hint, managed_peers}`; spawned peer -> 200 with the ring buffer |

Audit: with a bridge on `app.state`, spawn and despawn each landed one `record_activity("api", ...)`
row (`('spawn','smoke-so101',True,'so101 mode=sim')`, `('despawn','smoke-so101',True,None)`).
Without a bridge the routes still answer; the trail is simply not written (logged as such).

Also verified: `python -c "import strands_robots.dashboard.routes_devices"` clean; `ruff check` +
`ruff format --check` clean on all 11 files; `attach(app)` creates the hook lists, startup hook
without `app.state.bridge` logs "USB auto-spawn skipped" and sets no task; shutdown hook stops
every managed child.

## Gates kept (physical consequence)

- `validate_spawn` refuses an unknown robot / unspawnable mode with 422 before any process exists.
- `bus_claim.bus_conflict` inside `DeviceManager.spawn` refuses a port another process holds
  (`lsof` on `/dev/cu.*` and its `/dev/tty.*` sibling); the route surfaces it as 409 and audits it.
- `settle` watches the spawned pid for `SPAWN_SETTLE_S` (5 s); a child that died gets
  `error` + `consent.attach_consent(result, reason, log_tail)` so the frontend can classify it.
- Every spawn / despawn / camera change -> `bridge.record_activity`, including autospawn polls.

## Deps

- Required by the routes: nothing beyond fastapi (already in the venv).
- Optional, lazy inside functions: `opencv-python-headless` (camera roster, preview, modes; a
  missing cv2 is a 501 naming the extra), `pyserial` (`scan_serial_ports`; absent -> empty
  `serial_ports` list), `lerobot` (arm-role voltage probe runs in a subprocess; absent -> role
  `unknown` with a remedy). All three are present in `/Users/cagatay/robots/.venv`.

## Gaps

- `attach()` creates `app.state.autospawn_task` only when `app.state.bridge` exists at startup-hook
  time: the coordinator must run the mesh lane's attach (or set `app.state.bridge`) BEFORE the
  devices startup hook runs, otherwise autospawn is skipped (logged at INFO).
- `GET /devices/profiles.autospawn` reads `app.state.autospawn_task`, so it is `false` until that
  hook has run.
- No `/api/deploy/snippet` here (train lane owns `deploy.py`; the branch had it in this block).
- `direct_serial.py` ports verbatim; it imports `strands_robots.tools.{pose_tool,serial_tool}`
  lazily and reads `bus_claim` + `scan_serial_ports` from this lane - nothing from other lanes.
- `disk_headroom.py` and `build_info.py` are ported for `/api/health` (mesh lane's route) and the
  record lane; no route in this lane calls them.

## Needs coordinator

1. `server.create_app()`: `from strands_robots.dashboard import routes_devices` ->
   `app.include_router(routes_devices.router)` and `routes_devices.attach(app)`; run
   `app.state.startup_hooks` / `shutdown_hooks` from the lifespan, mesh lane's attach first.
2. `pyproject.toml` `[dashboard]` extra: `opencv-python-headless`, `pyserial` (both optional at
   import time, needed for the roster to show anything).
3. `device_manager.annotations_by_peer` imports `strands_robots.dashboard.joint_silence` (mesh
   lane) lazily; the merged branch must carry that file or the annotations call raises
   ImportError at request time (only reached from the mesh lane's fleet route).
4. `device_manager.spawn` lazily imports `strands_robots.dashboard.calibration
   .robot_calibration_gap` (train lane) inside a try/except; without it the calibration hint on a
   spawned card is just absent.
