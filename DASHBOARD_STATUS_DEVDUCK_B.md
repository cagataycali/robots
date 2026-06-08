# Dashboard Status — DevDuck-B handoff (2026-06-08, ~03:00)

Worktree: `/home/cagatay/mujoco-perfectining/robots-B`
Branch:   `feat/dashboard-policies-recording` (based on `feat/dashboard-v2`)
(DevDuck-A continues live-editing the main tree at `.../robots` on `feat/dashboard-v2`.)

## TL;DR for the morning
1. Dashboard shows all robots (sims + real arm) — WORKS (A's work + mine).
2. Leader -> follower teleop — plumbing verified; needs both arms CALIBRATED first.
3. Calibrate devices via dashboard — UNBLOCKED by my hardware fix; do the physical
   begin->home->sweep->finish in the Calibrate panel.
4. Policies panel — ADDED + tested (mock/groot/lerobot_local providers).
5. Record datasets — ADDED + tested (record_start -> run policy -> record_stop -> 12 frames on disk).
6. Rebase to PR #366 (fix/lerobot-052-recording) — ALREADY DONE (feat/dashboard-v2 is built on it).

## What I changed (4 files + 2 new, all additive)
- `strands_robots/mesh/core.py`: new _dispatch actions list_policies, list_robots,
  record_start, record_stop (+ helper methods).
- `strands_robots/mesh/security.py`: allowlisted the 4 actions + validators
  (repo_id traversal-safe, fps bounds, task length).
- `strands_robots/dashboard/observer.py`: list_policies/list_robots/record_start/record_stop.
- `strands_robots/dashboard/server.py`: WS handlers for the 4 actions.
- `strands_robots/dashboard/static/index.html`: Policies panel + Record panel.
- `strands_robots/robot.py`: **eager bus connect in mode='real'** (the key real-arm fix).
- `tests/test_robot_factory.py`: regression test for eager connect.
- NEW: `start_leader_follower.sh` — one-command physical teleop launcher.

## Root-cause bugs found + fixed
1. **Real arm on dashboard had no joints / calibrate failed with "FeetechMotorsBus
   is not connected".**  Cause: `Robot(mode='real')` joined the mesh but never
   connected the lerobot bus (connection was lazy, only inside the first policy
   run). FIX: factory now `inner.connect(calibrate=False)` at construction.
   Verified on real /dev/ttyACM3: `calibrate begin/cancel` now succeed over the
   mesh->dashboard WS path (previously dispatch error).
2. **Calibration dir mismatch (latent):** lerobot 0.5.2 stores SO-101 follower
   calibration under `.../robots/so_follower/<id>.json`, but several existing
   files live under `.../robots/so101_follower/<id>.json` (won't auto-load).
   The dashboard Calibrate flow writes fresh calibration directly to motors and
   persists via `_save_calibration` to the correct `so_follower` path, so
   calibrating THROUGH the dashboard sidesteps this. (If you want old calibrations
   to load, copy them into the `so_follower` dir with the matching `id`.)

## Hardware map (probed read-only)
- `/dev/ttyACM1` (USB 5AB0158428): SO-101, all joints ~2047 (centered) => likely LEADER.
- `/dev/ttyACM3` (USB 5AB0181806): SO-101, varied positions => likely FOLLOWER.
- Both respond on the bus; both currently UNCALIBRATED (is_calibrated=False).
- Cameras: /dev/video0, /dev/video1.
- NOTE: a stale `/tmp/hw_connect.py` (PID 1060533) holds a deleted /dev/ttyACM0 — kill it.

## Morning runbook
```bash
cd /home/cagatay/mujoco-perfectining/robots-B   # (my worktree, has all fixes)
# 1. (optional) kill stale holder
kill 1060533 2>/dev/null

# 2. Dashboard is already running on :7861 (mine) and :7860 (A's). To run the
#    fixed code: stop A's, or use mine. To start fresh with my fixes:
PYTHONPATH=$PWD MUJOCO_GL=egl STRANDS_MESH_AUTH_MODE=none \
  STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1 STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1 \
  .venv/bin/python -m strands_robots.dashboard --host 0.0.0.0 --port 7860

# 3. Spawn the real follower so it appears on the dashboard:
STRANDS_ROBOT_MODE=real PYTHONPATH=$PWD STRANDS_MESH_AUTH_MODE=none \
  STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1 STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1 \
  .venv/bin/python -c "import time; from strands_robots import Robot; r=Robot('so101',mode='real',port='/dev/ttyACM3',id='follower'); print('peer',r.peer_id); [time.sleep(1) for _ in iter(int,1)]"

# 4. In the dashboard: select the so101 peer -> Calibrate panel ->
#    1.Begin (arm goes limp) -> move to centre -> 2.Set Home ->
#    sweep every joint -> 4.Finish. Repeat for the leader arm.

# 5. Leader -> follower teleop (after BOTH calibrated):
./start_leader_follower.sh /dev/ttyACM1 /dev/ttyACM3 leader follower
#    Move the leader; the follower mirrors. Watch in dashboard.

# 6. Record a dataset: select sim/real peer -> Record panel -> repo_id +
#    task + fps -> Start -> run a policy (Policies panel) -> Stop+save.
```

## Tests
- All 1272 mesh unit tests pass.
- 79 factory tests pass (+ my new eager-connect regression).
- Sim E2E: real ACT policy -> MuJoCo -> LeRobot recording = 12 frames (act_e2e.py).
- Dashboard browser-protocol E2E: record_start -> mock policy -> record_stop = 12 frames.

## Notes / follow-ups
- Policy runs over the wire use the mesh policy_type allowlist (_DEFAULT_POLICY_TYPES:
  act, diffusion, groot, lerobot, lerobot_local, mock, pi0, pi0fast, sac, smolvla,
  tdmpc, vqbet). Raw HF model-ids (e.g. "lerobot/act_aloha_...") are blocked by
  design; to run them from the dashboard set STRANDS_MESH_POLICY_TYPE_ALLOW or run
  them server-side. Consider adding a model-picker that maps provider -> known
  checkpoint with explicit operator opt-in.
- list_policies occasionally times out on first call (heavy import); the refresh
  button retries. Could warm list_providers() at peer startup.
- My commits are in the robots-B worktree only. To merge into feat/dashboard-v2:
  `git -C /home/cagatay/mujoco-perfectining/robots cherry-pick <shas>` AFTER A's
  uncommitted work is committed (mine is based on A's HEAD + their live patch).
