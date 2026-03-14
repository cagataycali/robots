# 🧪 strands-robots LeRobot 0.5.x Integration Testing

**Branch**: `fix/lerobot-smolvla-inference`  
**Base**: `dev` @ strands-labs/robots  
**Target**: Push to `cagataycali/robots`  
**Date**: 2026-03-14  
**Platform**: Jetson AGX Thor — CUDA 13.0, torch 2.10.0+cu130, Python 3.12

---

## 🎯 Objective

Ensure all LeRobot 0.5.x policies can be resolved, loaded, and run inference through the `strands_robots` `lerobot_local` policy provider. Test with MuJoCo simulations.

## 📋 LeRobot 0.5.x Policy Matrix

| Policy | Class | Resolves | Loads Pretrained | Sim Inference | Notes |
|--------|-------|----------|-----------------|---------------|-------|
| act | ACTPolicy | ✅ | ⏳ | ⏳ | Core policy |
| diffusion | DiffusionPolicy | ✅ | ⏳ | ⏳ | |
| pi0 | PI0Policy | ✅ | ⏳ | ⏳ | |
| pi0_fast | PI0FastPolicy | ✅ | ⏳ | ⏳ | |
| pi05 | PI05Policy | ✅ | ⏳ | ⏳ | |
| rtc | RTCProcessor | ❌ | ⏳ | ⏳ | No Policy class in modeling_rtc — processor only |
| sac | SACPolicy | ✅ | ⏳ | ⏳ | RL policy |
| sarm | SARMRewardModel | ✅ | ⏳ | ⏳ | Reward model, not action policy |
| smolvla | SmolVLAPolicy | ✅ | ⏳ | ⏳ | VLA — needs transformers, sentencepiece |
| tdmpc | TDMPCPolicy | ✅ | ⏳ | ⏳ | |
| vqbet | VQBeTPolicy | ✅ | ⏳ | ⏳ | |
| wall_x | WallXPolicy | ❌ | ⏳ | ⏳ | Needs `peft` (installed) — resolution bug |
| xvla | XVLAPolicy | ✅ | ⏳ | ⏳ | VLA — needs transformers |

**Resolution Score**: 12/12 ✅ standalone policies + 1 non-standalone correctly rejected

## 🔧 Fixes Applied

### Cycle 0 — Initial Setup
- [x] Installed `torch 2.10.0+cu130` (was CPU-only)
- [x] Installed `transformers`, `sentencepiece`, `peft`
- [x] Created branch `fix/lerobot-smolvla-inference` from `dev`

### Cycle 1 — Policy Resolution & Dependencies (2026-03-14)
- [x] Fix `pyproject.toml` — add `peft>=0.15.0`, `qwen_vl_utils>=0.0.8`, `torchdiffeq>=0.2.0` to `[vla]` extra
- [x] Fix `_resolve_policy_class_by_name` — broadened class detection:
  - Added `_is_policy_class()` helper: matches `*Policy`, `*RewardModel`, and any `PreTrainedPolicy` subclass
  - Changed `except ImportError` → `except Exception` to handle transitive dep failures (qwen_vl_utils, torchdiffeq, peft)
  - Added `_NON_STANDALONE_POLICY_TYPES` registry with actionable error messages
  - `rtc` → clear `ValueError` explaining it's a PI0 wrapper, not a standalone policy
  - `wall_x` → now resolves correctly (was failing due to missing `qwen_vl_utils` + `torchdiffeq`)
  - `sarm` → resolves via factory (SARMRewardModel doesn't end in "Policy" but is in draccus registry)
- [x] Added `ValueError` catch to Strategy 3 (factory) — lerobot factory raises `ValueError` for unknown types
- [x] Existing tests: **19/19 passed** (test_policies.py)
- [x] New test file: `tests/test_lerobot_resolve.py` — **53 tests**, all passed
- [x] Full suite: **87/87 passed** across test_policies + test_policy_resolver + test_lerobot_resolve

#### Test Matrix After Cycle 1

| Policy | Class | Resolves | Strategy | Notes |
|--------|-------|----------|----------|-------|
| act | ACTPolicy | ✅ | modeling_act | Core policy |
| diffusion | DiffusionPolicy | ✅ | modeling_diffusion | |
| pi0 | PI0Policy | ✅ | modeling_pi0 | |
| pi0_fast | PI0FastPolicy | ✅ | modeling_pi0_fast | |
| pi05 | PI05Policy | ✅ | modeling_pi05 | |
| rtc | *(RTCProcessor)* | ✅ ValueError | Non-standalone | Wrapper around PI0 — not a policy |
| sac | SACPolicy | ✅ | modeling_sac | RL policy |
| sarm | SARMRewardModel | ✅ | factory | Reward model (PreTrainedPolicy subclass) |
| smolvla | SmolVLAPolicy | ✅ | modeling_smolvla | VLA — needs transformers, sentencepiece |
| tdmpc | TDMPCPolicy | ✅ | modeling_tdmpc | |
| vqbet | VQBeTPolicy | ✅ | modeling_vqbet | |
| wall_x | WallXPolicy | ✅ | modeling_wall_x | VLA — needs peft, qwen_vl_utils, torchdiffeq |
| xvla | XVLAPolicy | ✅ | modeling_xvla | VLA — needs transformers |

## 🔄 Autonomous Cycle Log

### Cycle 1 — Complete ✅
**Files changed:**
- `pyproject.toml` — added `peft`, `qwen_vl_utils`, `torchdiffeq` to `[vla]` extras
- `strands_robots/policies/lerobot_local/__init__.py` — rewrote resolver with `_is_policy_class()`, `_NON_STANDALONE_POLICY_TYPES`, broader exception handling
- `tests/test_lerobot_resolve.py` — NEW: 53 tests covering all 13 LeRobot policy types
- `TESTING.md` — updated with cycle results
