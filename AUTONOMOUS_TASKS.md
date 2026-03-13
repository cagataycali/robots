# AUTONOMOUS_TASKS.md — strands-robots E2E Test Results (FINAL)
# Last updated: 2026-03-12 20:00 UTC by DevDuck autonomous mode
# Cycles completed: 17 (comprehensive E2E coverage)

## Executive Summary
- **89/89 __all__ exports** importable ✅
- **73/73 existing pytest tests** PASS ✅
- **28/32 sim robots** create+destroy OK (4 missing MJCF models, not code bugs)
- **22 simulation actions** tested ✅
- **12/12 Strands tools** have valid tool_spec ✅
- **8 policy providers** registered, all resolve correctly ✅
- **Full E2E integration pipeline** Robot→Objects→Policy→Video→Render→Eval ✅
- **3 real bugs found** (see below)

## 🐛 BUGS FOUND

### BUG 1: DatasetRecorder drops ALL frames — camera feature mismatch (HIGH)
- **File**: `strands_robots/dataset_recorder.py` + `strands_robots/simulation.py`
- **Repro**: `sim.start_recording(...)` → `sim.run_policy(...)` → all frames dropped
- **Error**: `Feature mismatch: Extra features: {'observation.images.default'}`
- **Cause**: `DatasetRecorder.create()` with `camera_keys=[]` (so100 has no scene cameras declared), but `_get_sim_observation()` renders a "default" free camera and includes it. LeRobot rejects the frame.
- **Impact**: Training data recording from sim is broken for camera-equipped robots.
- **Fix**: Include rendered camera keys in `camera_keys` when calling `DatasetRecorder.create()`, or strip camera images from observation dict before passing to recorder.
- **Also**: Warning logging is unbounded — 4000+ identical lines. Needs rate limiting.

### BUG 2: `StrandsSimEnv(sim_object)` confusing API (MEDIUM)
- **File**: `strands_robots/envs.py:174`
- **Error**: `AttributeError: 'Simulation' object has no attribute 'lower'`
- **Cause**: First arg is `robot_name: str`, not a Simulation object. Natural pattern `StrandsSimEnv(Robot("so100"))` crashes.
- **Fix**: Add isinstance check or document clearly that it takes a string.

### BUG 3: 4 robots have missing MJCF model files (LOW)
- `asimov_v0`, `google_robot`, `so101`, `unitree_a1` — their asset dirs/model files don't exist or aren't auto-downloadable yet.
- Not a code bug — just incomplete asset coverage.

## ✅ Detailed Test Results

### Cycle 1: Package Imports & Pytest
| Test | Result |
|------|--------|
| 89 __all__ exports importable | ✅ |
| 73 pytest tests pass | ✅ |
| Factory Robot(), list_robots() | ✅ |
| Registry: 38 robots, 82 aliases | ✅ |
| Policy ABC, MockPolicy, create_policy | ✅ |
| Policy resolver: 7 shorthands, HF, server, WS, gRPC, ZMQ | ✅ |

### Cycle 2: MuJoCo Simulation (22 actions)
| Action | Result |
|--------|--------|
| create_world | ✅ |
| add_robot(so100) | ✅ |
| get_features (6j, 6a, 0cam) | ✅ |
| get_robot_state | ✅ |
| step(100) | ✅ |
| render(320x240) | ✅ |
| render_depth | ✅ |
| get_contacts | ✅ |
| add_object (cube injection) | ✅ |
| move_object | ✅ |
| remove_object (XML round-trip) | ✅ |
| run_policy(mock, fast) | ✅ |
| randomize(colors+lighting) | ✅ |
| reset | ✅ |
| set_gravity | ✅ |
| set_timestep | ✅ |
| add_camera (injection) | ✅ |
| get_observation | ✅ |
| send_action | ✅ |
| start_policy(async) | ✅ |
| destroy | ✅ |
| get_state | ✅ |

### Cycle 3: Multi-Robot Factory
| Robot | Joints | Mock Policy | Result |
|-------|--------|-------------|--------|
| so100 | 6 | ✅ | ✅ |
| panda | 9 | ✅ | ✅ |
| unitree_g1 | 30 | ✅ | ✅ |
| unitree_go2 | 13 | ✅ | ✅ |
| aloha | 16 | ✅ | ✅ |
| unitree_h1 | 20 | ✅ | ✅ |

### Cycle 4: Video + Eval
- record_video(mock, 2s) ✅
- eval_policy(mock, 3ep, 50steps) ✅
- start_policy(async) + stop ✅

### Cycle 5: Kinematics
- MuJoCoKinematics(body=Moving_Jaw) ✅
- forward_kinematics(qpos) → 4x4 ✅
- inverse_kinematics(current, target) ✅
- create_kinematics() factory ✅

### Cycle 6: Video + Processor + Gym
- encode_frames (306KB) ✅
- get_video_info ✅
- VideoEncoder class ✅
- create_processor_bridge(passthrough) ✅
- StrandsSimEnv(robot_name='so100') — reset/step ✅

### Cycle 7: Telemetry + Zenoh + Recording
- TelemetryStream lifecycle ✅
- BatchConfig ✅
- start_recording (LeRobot) ✅ (but frames dropped — BUG 1)
- stop_recording ✅
- get_peers() ✅

### Cycle 8: All 32 Sim Robots
- 28/32 pass ✅
- 4 fail: missing MJCF models (asimov_v0, google_robot, so101, unitree_a1)

### Cycle 9-10: Training + Advanced
- TrainConfig ✅ (field: max_steps, not num_epochs)
- create_trainer('lerobot') → LerobotTrainer ✅
- evaluate() signature ✅
- RLConfig ✅ (field: algorithm, not algo)
- DreamGenConfig, NeuralTrajectory ✅
- LEISAAC_TASKS: 4 tasks ✅
- NewtonConfig ✅
- CosmosTransferConfig ✅
- StereoConfig ✅
- list_isaac_tasks(): 11 tasks ✅

### Cycle 11: Policy Providers
- 8 providers registered: cosmos_predict, dreamgen, dreamzero, gear_sonic, groot, lerobot_async, lerobot_local, mock ✅
- All shorthands resolve correctly ✅
- HF model → provider resolution ✅ (SmolVLA→lerobot_local, nvidia/gr00t→groot)
- Server address → provider resolution ✅

### Cycle 12: Recording Pipeline
- RecordMode: TELEOP, POLICY, IDLE ✅
- EpisodeStats ✅
- RecordSession ✅
- RecordingVisualizer + RecordingStats ✅
- DatasetRecorder.create() ✅

### Cycle 13: All 12 Strands Tools
| Tool | Status |
|------|--------|
| gr00t_inference | ✅ |
| lerobot_camera | ✅ |
| pose_tool | ✅ |
| serial_tool | ✅ |
| teleoperator | ✅ |
| lerobot_dataset | ✅ |
| newton_sim | ✅ |
| stream | ✅ |
| stereo_depth | ✅ |
| robot_mesh | ✅ |
| use_lerobot | ✅ |
| inference | ✅ |

### Cycle 14: Zenoh Mesh + Async
- Mesh class (12 methods) ✅
- PeerInfo (5 fields) ✅
- _resolve_coroutine (sync + async) ✅

### Cycle 15: Edge Cases & Stress
- Create → destroy → recreate cycle ✅
- Sequential 4-robot creation ✅
- MockPolicy DOF: 1, 3, 6, 12, 30 ✅
- Empty observation → default 6DOF ✅
- Add 5 objects → remove all ✅
- Multi-camera rendering ✅
- Unitree G1: 1000 physics steps ✅

### Cycle 16: Telemetry Transports + Video
- LocalWALTransport ✅
- StdoutTransport ✅
- TelemetryEvent + BatchConfig ✅
- encode_frames at 3 resolutions ✅

### Cycle 17: Full E2E Integration
Robot(panda) → add_objects → get_observation → run_policy → record_video → render(70KB PNG) → get_contacts → randomize → eval_policy → destroy **ALL PASS** ✅

## Test Environment
- **Platform**: Linux aarch64 (NVIDIA Jetson)
- **Python**: 3.13
- **strands-robots**: 0.3.9.dev24+gc20e23799
- **MuJoCo**: EGL headless GPU rendering
- **lerobot**: installed (dataset recording available)
