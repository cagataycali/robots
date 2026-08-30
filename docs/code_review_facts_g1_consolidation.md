# Code Review Facts: neon-the-g1 → strands-labs/robots port

**Date:** 2026-08-30 · **Repos reviewed:** strands-labs/robots@main (c3ac656f), cagataycali/neon-the-g1, cagataycali/tiny-the-reachy, cagataycali/scout-the-rover, unitreerobotics/unitree_sdk2_python

---

## 1. The headline problem: tool bloat in `strands_robots/tools/g1/`

| Metric | neon-the-g1/tools (g1 files) | strands_robots/tools/g1 |
|---|---|---|
| Files | 15 g1 modules (+ shared stack) | **68 files** |
| `@tool` functions | ~60 (curated into bundles) | **115** |
| Lines | ~10.4k (all tools incl. telegram/memory) | **19.5k (g1 only)** |
| Execution verbs (do real work) | ~50 | **~14** |
| Metadata/lookup-only verbs | ~6 (joints reference) | **~100** |

The port inverted the ratio: neon is mostly *execution* tools; strands-robots is mostly *lookup* tools.

### 1a. The "envelope/admits pair" anti-pattern
~50 of the 68 files follow one template: `g1_list_X_envelope` + `g1_X_admits` — pure read-only constant tables snapshotting neon's clamp values, with zero execution path. Examples:
- `g1_walk_forward_envelope.py` (418 lines) → answers "is distance=0.5 in [-1,1]?" — no walking
- `g1_slam_map_names.py` (375 lines) → validates a map name string — no SLAM
- `g1_speak_vad_envelope.py` (445 lines) → VAD threshold table — no speech
- `g1_bidi_audio_frame_size_envelope.py`, `g1_turn_envelope.py`, `g1_swing_height_envelope.py`, ... (×~45 more)

Each is 200–520 lines of prose docstring wrapping ~20 lines of constants. **From an LLM-agent perspective these ~100 tool names pollute the tool registry**: an agent asked to "walk forward" sees `g1_list_walk_forward_envelope` and `g1_walk_forward_admits` but no `g1_walk_forward`. Every envelope tool duplicates what a good execution tool does internally in 2 lines (`clamp()` + refusal message).

### 1b. Tool names exposed today (main) — the actual execution surface
Real verbs present: `g1_battery, g1_imu, g1_lidar_state, g1_lidar_summary, g1_mainboard, g1_pressure, g1_get_task_status, g1_run_policy, g1_send_action, g1_start_task, g1_stop_task, g1_set_stand_height, g1_set_swing_height, g1_joint_reference/name/index, g1_decode_error_code`. Everything else is lookup.

### 1c. Missing from the port (present + battle-tested in neon)
- **`use_unitree`** — the universal SDK dispatcher (use_aws pattern). ONE tool covering the entire unitree_sdk2py surface: 6 services (loco, arm, audio, motion_switcher, vui, robot_state), dynamic discovery (inspect + AST fallback for SDK-less dev boxes), mutative-op detection, HIGH_DANGER_OPS set, singleton client cache. **This single tool replaces ~40 of the envelope files** because `describe_operation` already returns signature + danger flags per SDK method.
- **Locomotion execution**: `g1_move_velocity, g1_stop_move, g1_walk_forward, g1_turn, g1_wave_hand_loco, g1_shake_hand_loco, g1_set_task_id`
- **Arm execution**: `g1_arm_action, g1_release_arm, g1_list_arm_actions, g1_get_arm_action_list_from_robot` (release_arm in-flight as PR #3034)
- **Posture**: `g1_set_fsm, g1_balance_stand, g1_safe_squat_to_stand, g1_safe_lie_to_stand, g1_safe_stand_to_squat`
- **Audio**: `g1_speak, g1_play_wav, g1_asr` (+ bidi audio)
- **SLAM execution**: `g1_slam_start/stop/pose/reset/accumulate/save/load/list_maps/stats` (kiss-icp, out-of-SDK)
- **DDS escape hatch**: `g1_dds_list_topics/discover/snapshot/subscribe/read/unsubscribe/stats/publish`
- **Camera/vision**: `use_camera, capture_camera`
- **Curated bundles**: neon exports `G1_STATE_TOOLS / G1_POSTURE_TOOLS / G1_LOCOMOTION_TOOLS / G1_ARM_TOOLS / G1_AUDIO_TOOLS / G1_SENSING_TOOLS / G1_SAFE_TOOLS / G1_ALL_TOOLS` — strands-robots exports no bundles at all (`tools/g1/__init__.py` exports only DDS-layer constants).

---

## 2. Driver state (`strands_robots/drivers/g1.py`, 2147 lines)

Solid and should be preserved as the transport/gate layer:
- Subscribes 6 topics (`rt/lowstate, rt/lf/bmsstate, rt/utlidar/lidar_state, rt/utlidar/cloud_livox_mid360, rt/mainboardstate, rt/pressuresensorstate`) on background DDS thread with cache + `_snapshot()` accessor.
- `_check_motion_gates` (FSM ∈ HANDSHAKE_FSMS/WALK_FSMS + battery floor 15%) gates `send_action` / `run_policy`.
- 500 Hz `_ControlLoop` with separate 10 Hz FSM refresher thread, zero-torque frame on exit, staleness bound = 10 missed reads.
- `tool_spec` surface already exists (line 429) — the seam issue #2928 shape (c) wants.
- `start_task` still refuses (provider registry not plumbed); motion-switcher wiring tracked in #2765/#2891/#2916.

**vs unitree_sdk2_python:** SDK `LocoClient` exposes 24 methods (GetFsmId, SetFsmId, SetVelocity, Move, Damp, ZeroTorque, Squat2StandUp, WaveHand, ShakeHand, HighStand/LowStand, BalanceStand, ...). The driver replicates none of the high-level loco RPCs — it only owns lowcmd frames + gates. neon's tools call SDK clients directly through `_g1_common` singletons. **Conflict:** two DDS init paths (driver's `_dds_engine` vs neon's `_g1_common.ensure_dds`) — both exist in strands-robots tree; `_DDS_INIT_LOCK` in `_g1_common` is the agreed serialization point (per docstrings), so ported verbs must read through the driver cache, not stand up their own subscribers (issue #2928 question 1).

---

## 3. GitHub state (strands-labs/robots)

- **Issue #2928 (OPEN)**: "the tools-bundle port has no seam decided" — the design issue. Proposes shape (c): P0/P1 verbs as **driver methods + tool_spec**, P3 (audio/camera/speak/slam/dds) as free-standing `@tool`s. Notes #358 reference is **dangling** (consumed by unrelated flake fix) yet ~60 merged files still cite "refs #358".
- **Issue #2891 (OPEN)**: FSM read cadence blocks `send_action` wire step.
- **Issue #2916 / #2765**: gate producer / wire-format (referenced, gating P1/P2).
- **PRs in flight**: #3034 ports `g1_release_arm` (execution verb ✓), #3029 ports *another* envelope pair (bloat continues ✗). Recent main history is a stream of envelope-pair ports (#3027, #3032, #3031 …).
- **Process smell (#2940, #2720)**: verb ports shipped dead xref roles; mass issue-closes without PRs.

**Conclusion: main's merge stream contradicts #2928's own proposal** — envelope lookups keep landing as separate tools while the seam decision sits open.

---

## 4. Reachy state

- `strands_robots/tools/reachy/`: only `__init__.py` + `_reachy_common.py` (116 lines, envelope_error helper). **Zero tools.**
- `strands_robots/drivers/reachy.py`: good native daemon driver (REST + WS/Zenoh via device_connect transports, IMU/pose/battery caches, envelope refusals).
- tiny-the-reachy has the tool set to port: `reachy_motion (5 tools), reachy_camera (2), reachy_audio (3), reachy_expression (2), reachy_state (2), vision (1)` — ~15 tools, all execution-shaped, no envelope bloat.

## 5. Earth rover (scout) state

- **No rover driver, no rover tools in strands-robots at all** (grep for rover/frodobot/earth in drivers/: empty).
- scout-the-rover tool surface to port: `rover_motion (3), rover_navigate (1), rover_pose (1), rover_state (2), rover_camera (2), rover_record (7), room_map (1), rover_memory (1), rover_async (1)` — ~19 tools + `_rover_common.py`/`_controller_engine.py`/`_recorder_engine.py` engines.

## 6. Cross-repo shared stack (dedupe opportunity)

neon, tiny, scout each vendor near-identical copies of: `telegram.py, voice_bridge.py, memory.py, dispatch.py, prompts.py, manage_messages.py, manage_tools.py, vision.py, agent_log.py, thinker_loop.py, telegram_listener.py`. None of these are robot-specific → belong in a shared package (robots-harness), not per-robot trees and not strands_robots/tools/<robot>/.

---

## 7. Recommended target shape (for the branch)

### tools/g1/ — from 68 files → ~12 files
```
_g1_common.py        # keep: singletons, ensure_dds, ERR_CODES, FSM sets
_dds_engine.py       # keep: driver's subscriber layer
_motion_switcher.py  # keep
use_unitree.py       # ADD from neon: universal SDK dispatcher (replaces ~40 envelope files)
g1_state.py          # merge: g1_state + g1_battery + g1_imu + g1_mainboard + g1_pressure + g1_lidar_state/summary → reads via driver._snapshot
g1_locomotion.py     # ADD from neon: move/stop/walk_forward/turn (+wave/shake), clamps INLINE, routed via driver gate
g1_posture.py        # ADD from neon: set_fsm/stand_height/swing_height/balance_stand + safe_* transitions
g1_arm.py            # ADD from neon: arm_action/release_arm/list_actions
g1_audio.py          # ADD from neon: speak/play_wav/asr (+ bidi later)
g1_slam.py           # ADD from neon: kiss-icp runner (fix _safe_map_path prefix bug noted in g1_slam_map_names.py — use Path.is_relative_to)
g1_dds.py            # ADD from neon: subscribe/read/publish escape hatch (through driver's engine, single init lock)
g1_joints.py         # keep: joint_reference/name/index (genuinely useful lookup)
__init__.py          # export curated bundles: G1_SAFE_TOOLS / G1_ALL_TOOLS (neon pattern)
```
**Delete: all ~50 `*_envelope.py` / `*_admits` / `*_topics.py` / `*_ids.py` / `*_keys.py` / `*_notes.py` lookup modules.** Their constants fold into the executing tool (clamp + refusal message) or into `use_unitree.describe_operation`. Net: −~15k lines, −~100 tool names.

### tools/reachy/ — port tiny's ~15 execution tools onto ReachyDriver caches.
### tools/rover/ — new: port scout's engines + tools; needs a RoverDriver first (separate track).

### Preserve from strands-robots (don't regress):
- driver-gated writes (`_check_motion_gates`), SDK-load hygiene (no module-level unitree_sdk2py imports), `driver._snapshot()` read seam, duck-typed driver args for testability, NullConversationManager-free single init lock.
- Fix the SLAM path-containment bug when porting (documented sibling-dir prefix bypass: `../maps-evil/pwn` admitted).

### Open decision (flag in the harness issue):
- #2928 shape (c) vs neon's free-standing `@tool`s: propose **hybrid** — P0 reads stay duck-typed on `driver._snapshot` (already the merged pattern in g1_battery/g1_state), P1/P2 execution verbs call driver methods (`send_action`/`run_policy`), P3 (audio/slam/camera/dds/use_unitree) free-standing per #2928's own paragraph 4.

## 8. Plan of record
1. Branch `g1-tools-consolidation` on cagataycali/robots (fork exists ✓, default main).
2. Apply target shape above; keep driver untouched except FSM/motion-switcher refs.
3. Cut issue on cagataycali/robots-harness (exists ✓) linking this file + branch → PR.
