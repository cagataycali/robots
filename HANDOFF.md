# HANDOFF - lane record (branch `dash/lane-record`, commit 1cfad36df)

## Files added (strands_robots/dashboard/)

| file | LOC | from | adapted |
|------|-----|------|---------|
| routes_record.py | 213 | branch server.py 753/785/823 | NEW router: /api/collect, /api/replay, /api/datasets/labels; mounts record_api's router; `attach(app)`; path containment |
| record_api.py | 619 | branch (539) | `Depends(access.require_session)` on the router; router late-bound (resolver per request); lerobot refusal 424; devices=None refusal 503; camera_liveness / disk_headroom imported lazily (devices lane files) |
| record_worker.py | 585 | branch | verbatim + docstrings, one em dash -> `-` |
| record_crash.py, record_joints.py, record_motion.py, episode_label_view.py, dataset_check.py, output_dir_check.py, upload_preflight.py | 110/71/122/90/167/121/118 | branch | verbatim + module docstrings (main's ruff D100/D102 rules) |

No archaeology found in the ported text (grep for R<n>, finding #, Defect, BUGS.md, thread, PR numbers: clean).

## Routes served (all `Depends(access.require_session)` -> 401 without a session)

    GET  /api/record/session                     200 idle view (crash crumb + disk notice when present)
    GET  /api/record/upload-preflight            needs train lane's checkpoints.hf_auth_state (lazy import)
    POST /api/record/open                        424 named-extra refusal without lerobot; 503 without app.state.devices
    POST /api/record/episode/start|stop|redo     409 "no recording session is open" when idle
    POST /api/record/episode/discard             422 without index
    POST /api/record/close
    GET  /api/record/thumb/{episode}/{camera}    camera sanitised to [A-Za-z0-9_-]
    POST /api/collect                            dataset_root contained; 503 without devices
    POST /api/replay                             root contained BEFORE validate_replay (which would leak existence)
    GET  /api/datasets/labels?root=|path=        contained; 404 inside home when missing

Verified with a TestClient smoke (not committed; `app = FastAPI(); app.include_router(router); attach(app)`,
`settings.override("security","auth_token",...)`):

- `import strands_robots.dashboard.routes_record` clean under PYTHONPATH=worktree; `ruff check` + `ruff format --check` clean on all 10 files
- openapi lists exactly the 12 paths above
- GET /api/record/session -> 200 phase=idle; without bearer -> 401; POST /api/collect without bearer -> 401
- GET /api/datasets/labels?path=/etc -> 400 `{"error":"path_outside_dataset_home",...}`; `/definitely/not/here` -> the SAME 400 body; `<home>/../..` -> same 400; `<home>/nope` -> 404
- POST /api/collect dataset_root=/tmp/x -> 400 same body; POST /api/replay root=/etc -> 400 same body
- POST /api/record/open with `sys.modules["lerobot"]=None` -> 424 `{"error":"missing_optional","extra":"lerobot","install":"pip install 'strands-robots[lerobot]'"}`; with lerobot but no devices -> 503 clear body
- thumb `/0/../../etc` -> 404

## Design notes

- Dataset home = `strands_robots.dataset_source._lerobot_home()` (lerobot's `HF_LEROBOT_HOME` when
  installed, else `~/.cache/huggingface/lerobot`), `.expanduser().resolve()`. `contained_path()` in
  routes_record does `Path(x).expanduser().resolve()` then `is_relative_to(home)`; one fixed body
  `OUTSIDE_DATASET_HOME` for every refusal. Nothing added to settings.py (no dataset key exists there;
  steer the home with the `HF_LEROBOT_HOME` env var).
- The containment ALSO applies to `/api/collect dataset_root` and `/api/replay root` per the lane
  ruling ("any path the client names"). The branch let collect write anywhere; an operator who wants
  a dataset outside `$HF_LEROBOT_HOME` now has to point that env var there. Owner may relax.
- `record_api.build_router(controller, on_activity, late_bound=True)` takes resolvers
  `(request) -> RecordController` / `(request) -> callable|None`; `routes_record.controller()` reads
  `app.state.record` and builds it on first use if `attach()` never ran. Paths/payloads verbatim.
- `attach(app)`: sets `app.state.record = RecordController(app.state.devices, bridge=app.state.bridge)`
  eagerly when `devices` already exists; otherwise appends a startup hook to `app.state.startup_hooks`
  that rebuilds once every lane has attached (devices lane may attach after this one); if there is no
  hooks list it builds with devices=None (session view works, /open refuses 503).
- Activity: `_activity(request)` returns `app.state.bridge.record_activity` per request (never pins a
  bridge instance). Signature used: `record_activity("record", "session_open"|"session_close",
  target=..., detail=..., ok=...)` - same as the branch; main's mesh_bridge.record_activity at line 1200.
- lerobot: `record_api.lerobot_refusal()` uses `strands_robots.utils.require_optional("lerobot",
  extra="lerobot")` at the top of `RecordController.open()`; the HTTP code is 424 (Failed Dependency).
  The hardware backend (`record_worker.hardware_backend`) still imports Robot/Teleoperator lazily.

## Cross-lane imports (by the branch's public names, lazy, inside functions)

- devices lane: `dashboard.camera_liveness` (dead_cameras/refusal/missing_cameras/missing_refusal/
  identity_drift/drift_refusal) inside `open()`; `dashboard.disk_headroom` (free_space/headroom_verdict)
  inside `_disk_notice()` with ImportError -> no notice; `dashboard.device_manager.validate_replay` in
  /api/replay; `DeviceManager.{robots, spawn, despawn, autospawn, annotations_by_peer, collect, replay,
  _camera_names_cache, _camera_names_cache_t}` duck-typed through `app.state.devices`.
- train lane: `dashboard.checkpoints.hf_auth_state` in /upload-preflight; `dashboard.training.
  remember_dataset_root` in /api/collect (ImportError tolerated with a debug log - the collection
  still runs; once the train lane is merged the picker remembers the root).
- package: `strands_robots.dataset_recorder.{DatasetRecorder, resolve_dataset_dir}`,
  `strands_robots.episode_labels`, `strands_robots.mesh.pacing.Ticker`, `strands_robots.robot.Robot`,
  `strands_robots.teleoperator.Teleoperator` - all present on main.

## Deps

Nothing new for import; lerobot (extra `lerobot`) to actually open a session; opencv or Pillow for
thumbnails (best-effort, skipped when absent); huggingface_hub for upload preflight facts.

## Gaps

- `output_dir_check.py` is ported but nothing in this lane calls it (the branch's training routes do:
  train lane). Left in place because ownership put it here.
- `/api/record/upload-preflight` will 500 with ImportError until the train lane's `checkpoints.py`
  lands in the merged branch (branch behaviour, unchanged).
- Not exercised end to end against real arms in this sitting (no leader/follower pair on this Mac).

## Needs coordinator

1. `server.create_app()`: `app.include_router(routes_record.router)` and `routes_record.attach(app)`;
   run `app.state.startup_hooks` at startup (this lane appends one coroutine when devices are not yet
   attached). Attach the devices lane BEFORE record if you want the eager path.
2. Merge order: devices lane files (`camera_liveness.py`, `disk_headroom.py`, `device_manager.py`) and
   train lane files (`checkpoints.py`, `training.py`) are imported lazily here; the merged branch needs
   them for /open, /upload-preflight and the picker memory to be complete.
3. Decide whether `/api/collect dataset_root` containment (this lane's reading of the ruling) is the
   final posture, or whether collect may write outside `$HF_LEROBOT_HOME`.
4. Changelog fragment + docs for the four route groups (coordinator-owned paths).
