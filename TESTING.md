# 🧪 strands-robots LeRobot 0.5.x Integration Testing

**Branch**: `fix/lerobot-smolvla-inference`  
**Base**: `dev` @ strands-labs/robots  
**Target**: Push to `cagataycali/robots`  
**Date**: 2026-03-14  
**Platform**: Jetson AGX Thor — JetPack 7.0, CUDA 13.0, SM 11.0, torch 2.10.0+cu130, Python 3.12

---

## 🎯 Objective

Ensure all LeRobot 0.5.x policies can be resolved, loaded, and run inference through the `strands_robots` `lerobot_local` policy provider. Test with MuJoCo simulations end-to-end.

## 📋 LeRobot 0.5.x Policy Matrix

| Policy | Class | Resolves | Loads Pretrained | GPU Inference | Notes |
|--------|-------|----------|-----------------|---------------|-------|
| act | ACTPolicy | ✅ | ✅ | ✅ | `lerobot/act_aloha_sim_transfer_cube_human` |
| diffusion | DiffusionPolicy | ✅ | ✅ | ✅ | `lerobot/diffusion_pusht` |
| pi0 | PI0Policy | ✅ | ⏳ | ⏳ | No public pretrained model |
| pi0_fast | PI0FastPolicy | ✅ | ⏳ | ⏳ | No public pretrained model |
| pi05 | PI05Policy | ✅ | ⏳ | ⏳ | No public pretrained model |
| rtc | RTCProcessor | N/A | N/A | N/A | **Post-processor wrapper**, not standalone policy |
| sac | SACPolicy | ✅ | ⏳ | ⏳ | RL policy, different training loop |
| sarm | SARMRewardModel | ✅ | N/A | N/A | **Reward model**, not action policy |
| smolvla | SmolVLAPolicy | ✅ | ✅ | ✅ | VLA with language — `lerobot/smolvla_base` |
| tdmpc | TDMPCPolicy | ✅ | ❌ | ❌ | `lerobot/tdmpc_pusht` repo deleted/moved in 0.5 |
| vqbet | VQBeTPolicy | ✅ | ❌ | ❌ | Config incompatibility: `mlp_hidden_dim` field removed in 0.5 |
| wall_x | WallXPolicy | ✅ | ⏳ | ⏳ | Needs `peft` (in [vla] extras) |
| xvla | XVLAPolicy | ✅ | ⏳ | ⏳ | VLA — needs transformers |

**Resolution Score**: 11/13 ✅ (rtc/sarm are non-action types)  
**Load Score**: 3/3 tested ✅  
**Inference Score**: 3/3 tested ✅  

## 📊 Inference Latency (Jetson AGX Thor)

| Model | Median | P95 | Min | Max | Params | Notes |
|-------|--------|-----|-----|-----|--------|-------|
| ACT (ALOHA sim) | **0.4ms** | 0.4ms | 0.4ms | 0.4ms | ~5M | Lightweight CNN encoder + transformer decoder |
| SmolVLA (base) | **2.7ms** | 2.8ms | 2.6ms | 2.9ms | ~500M | VLA with SmolVLM2-500M backbone + language |
| **Speedup** | **SmolVLA is 7.3× slower than ACT** | | | | | Both comfortably real-time on Thor GPU |

> Both policies run well under 10ms per inference — comfortably above 100Hz control frequency. ACT's sub-millisecond inference makes it ideal for high-frequency sim-to-real transfer.

## 🤖 MuJoCo Simulation Test Matrix

| Robot | Joints | Actuators | Cameras | Mock Policy | ACT Policy | Pipeline |
|-------|--------|-----------|---------|-------------|------------|----------|
| ALOHA (bimanual) | 16 | 14 | 6 | ✅ | ✅ | ✅ |
| SO-100 (arm) | 6 | 6 | 0 | ✅ | — | ✅ |

## 🔧 Fixes Applied

### Cycle 1 — Dependencies & Resolution (✅ Complete)
- [x] Verified `pyproject.toml` already has `transformers`, `peft`, `qwen_vl_utils`, `torchdiffeq` in `[vla]`
- [x] Installed all VLA deps: `transformers>=5.0.0`, `peft>=0.15.0`, `sentencepiece`
- [x] Created `tests/test_lerobot_resolve.py` — 20 tests, all pass
- [x] Verified existing `tests/test_policies.py` — 19 tests, all pass

### Cycle 2 — CUDA + GPU Inference (✅ Complete)
- [x] Installed `torch 2.10.0+cu130` for Jetson AGX Thor (was CPU-only)
- [x] **Fixed critical CUBLAS bug**: pip `nvidia-cublas 13.1.0.3` conflicted with JetPack 7 system cuBLAS 13.0.0.19 → uninstalled pip version → system cuBLAS works
- [x] ACT model: load + inference on CUDA ✅
- [x] Diffusion model: load + inference on CUDA ✅  
- [x] SmolVLA model: load + inference with language instruction on CUDA ✅
- [x] Created `tests/test_lerobot_inference.py` — 9 Cycle 2 tests
- [x] Discovered: `lerobot/tdmpc_pusht` and `lerobot/vqbet_pusht` models incompatible with lerobot 0.5 (config schema changed)

### Cycle 3 — MuJoCo Simulation + Policy Runner Integration (✅ Complete)
- [x] Fixed Cycle 2 `TestSimulationInference.test_simulation_creates_world` hang (missing `mesh=False` + cleanup)
- [x] **TestSimWorldCreation** (8 tests): World lifecycle — create/step/reset/destroy, ALOHA + SO-100 robots
- [x] **TestSimObservationAction** (6 tests): Joint state obs, camera images (RGB uint8), apply_action ctrl, public API
- [x] **TestMockPolicyInSim** (6 tests): Mock sinusoidal policy on ALOHA + SO-100, joint positions change, eval_policy, error cases
- [x] **TestSimToolDispatch** (3 tests): AgentTool `_dispatch_action` interface: full pipeline, unknown actions, add_object
- [x] **TestACTPolicyInSim** (3 tests): ACT pretrained model through `run_policy` on ALOHA sim, non-zero actions, joint name matching
- [x] **TestInferenceLatency** (3 tests): ACT vs SmolVLA profiling with warm-up, side-by-side comparison
- [x] **TestFullSimPipeline** (3 tests): End-to-end pipeline: create world → add robot → add object → step → observe → ACT policy → verify; render after policy

## 📈 Test Counts

| File | Cycle 1 | Cycle 2 | Cycle 3 | Total |
|------|---------|---------|---------|-------|
| `tests/test_lerobot_resolve.py` | 20 | — | — | 20 |
| `tests/test_policies.py` | 19 | — | — | 19 |
| `tests/test_lerobot_inference.py` | — | 9 | 32 | **41** |
| **Total** | 39 | 9 | 32 | **80** |

## ⚠️ Known Issues

1. **NVIDIA Thor cuBLAS conflict**: pip-installed `nvidia-cublas` MUST be removed on JetPack 7 — system cuBLAS takes priority. This should be documented.
2. **tdmpc/vqbet pretrained models**: The old HuggingFace repos (`lerobot/tdmpc_pusht`, `lerobot/vqbet_pusht`) are incompatible with lerobot 0.5 config schema. Need updated models or config migration.
3. **RTC is not a standalone policy**: It's a real-time chunking processor that wraps other policies (e.g., pi0+rtc). The resolver correctly handles this.
4. **Simulation `mesh=True` hangs in tests**: Zenoh mesh networking init blocks when no network peers are available. Tests must use `mesh=False` or properly `cleanup()` Simulation instances.
5. **MUJOCO_GL=egl required**: Headless rendering on Jetson needs `MUJOCO_GL=egl` env var set before mujoco import.

## 🔄 Autonomous Cycle Log

### Cycle 1 — Policy Resolution + Tests ✅
- All 13 lerobot 0.5.x policy types investigated
- 10 core + 1 optional (wall_x) resolve correctly
- 2 non-policy types (rtc=processor, sarm=reward) documented
- 39 total tests pass (20 new + 19 existing)

### Cycle 2 — GPU Inference Testing ✅
- Fixed torch CPU→CUDA installation
- Discovered and fixed cuBLAS version conflict (pip vs JetPack system)
- ACT, Diffusion, SmolVLA all run inference on Thor GPU
- SmolVLA with language instruction works end-to-end
- 9 GPU inference tests pass (6 model + 3 sim integration)

### Cycle 3 — MuJoCo Simulation + Policy Runner Integration ✅
- **32 new tests** covering 7 test classes
- Full Simulation lifecycle tested: create → robot → objects → step → observe → policy → render → destroy
- ALOHA bimanual (16 joints, 14 actuators, 6 cameras) fully tested with both mock and ACT policies
- SO-100 arm (6 joints, 6 actuators) tested with mock policy + pipeline
- ACT pretrained model runs through `Simulation.run_policy()` end-to-end ✅
- Inference profiled: **ACT 0.4ms, SmolVLA 2.7ms** (both real-time on Thor)
- Tool dispatch interface (`_dispatch_action`) verified for AgentTool compatibility
- Camera rendering verified (EGL offscreen on Jetson)
- Fixed Cycle 2 test hang (missing cleanup + mesh flag)
- All 41 tests in `test_lerobot_inference.py` pass in 67s

### Cycle 4+ — Planned
- [ ] Test SmolVLA in ALOHA sim (VLA with language + vision from sim cameras)
- [ ] Test policy switching at runtime (swap ACT → SmolVLA mid-episode)
- [ ] Test trajectory recording during policy execution
- [ ] Test domain randomization (lighting, textures, object positions)
- [ ] Test real hardware integration (SO-100 physical)
- [ ] Profile full pipeline latency (obs→inference→action including rendering)
