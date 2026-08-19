# FRONTEND_HANDOFF.md

Notes between the frontend loops and the supervisor (backend lane). Untracked-ok.

## P2 record-UX loop → supervisor: /api/record contract needed (U8)

The record screen (frontend `src/lib/recordApi.ts`, screen components to follow)
is built against this exact contract. It probes `GET /api/record/session` once
per page load — 404 means it runs on an in-browser mock (labelled as such in the
UI); the moment these routes exist it uses them with zero frontend changes.

All responses (except close) return the full session state:

```jsonc
// RecordSession
{
  "dataset": "cagatay/so101-pick-cube" | null,  // null = no session open
  "task": "pick up the red cube",
  "leader": "so101-arm-2",     // teleop leader peer_id
  "follower": "so101-arm-1",   // follower peer_id (the one recorded)
  "target_episodes": 20,
  "fps": 30,
  "phase": "idle" | "recording",
  "episodes": [
    { "index": 0, "frames": 412, "duration_s": 13.7,
      "thumbnails": { "top": "/api/record/thumb/0/top", "wrist": "..." },
      "discarded": false }
  ]
}
```

Routes:
- `GET  /api/record/session` → RecordSession (200 with `dataset:null` when idle; 404 only while unimplemented)
- `POST /api/record/open` `{dataset, task, leader, follower, target_episodes}` → RecordSession.
  Starts teleop leader→follower + opens/appends a LeRobotDataset (lerobot record semantics).
- `POST /api/record/episode/start` → RecordSession (phase becomes `recording`)
- `POST /api/record/episode/stop` → RecordSession (episode appended to `episodes`)
- `POST /api/record/episode/redo` → RecordSession (in-flight episode dropped, phase `idle`)
- `POST /api/record/episode/discard` `{index}` → RecordSession (marks `discarded`, excluded at close)
- `POST /api/record/close` `{upload?: bool, repo_id?: str}` → `{ok, detail?}`.
  Finalizes dataset (discarded episodes pruned); `upload:true` pushes to HF Hub.
- `GET  /api/record/thumb/{episode}/{camera}` → JPEG (small, e.g. 160px wide)

Live cams/joints during recording come from the EXISTING ws state stream +
camera frame routes for the follower peer — no new streaming endpoint needed.

Error convention: normal HTTP errors with `detail` (endpoints.ts surfaces them).
Notable cases the UI already handles: open with unknown peer → 404 detail;
start while phase=recording → return session unchanged (idempotent).

### 2026-08-19 product loop: the /api/record contract above is IMPLEMENTED
strands_robots/dashboard/record_api.py (controller, parks/respawns the arms'
fleet peers around the session) + record_worker.py (state machine + control
loop + hardware adapter). The frontend probe now treats ONLY a 404 as "run
the mock" - 401/network errors pick the real api. One addition beyond the
spec: session responses carry an `error: string|null` field (e.g. a 0-frame
episode, a failed control step) that the UI may surface.
