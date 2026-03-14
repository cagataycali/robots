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

**Resolution Score**: 11/13 ✅ (2 need fixes)

## 🔧 Fixes Applied

### Cycle 0 — Initial Setup
- [x] Installed `torch 2.10.0+cu130` (was CPU-only)
- [x] Installed `transformers`, `sentencepiece`, `peft`
- [x] Created branch `fix/lerobot-smolvla-inference` from `dev`
- [ ] Fix `pyproject.toml` — add `transformers`, `peft` to `[vla]` extra
- [ ] Fix `_resolve_policy_class_by_name` for `rtc` (processor, not policy) 
- [ ] Fix `_resolve_policy_class_by_name` for `wall_x` (import after peft)

## 🔄 Autonomous Cycle Log

### Cycle 1 — TBD
*(will be updated by autonomous agent)*
