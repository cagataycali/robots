# Cross-Dependency AST Validation Research

> **Date**: 2026-03-11  
> **Goal**: Validate that strands-robots' 45K-line codebase correctly interfaces with all 3rd-party dependencies at the AST level — before code reaches real hardware.

## The Problem

strands-robots imports from **15+ critical libraries** (lerobot, mujoco, torch, transformers, gr00t, unitree_sdk2py, etc.). When any upstream library changes a module path, renames a function, or alters a signature, our code silently breaks. On a laptop this means a traceback. On a real robot, this means an uncontrolled actuator.

Traditional `pytest` catches runtime failures *after* they happen. We need **static cross-boundary validation** that catches breakage *before* a single motor moves.

## Approach: Three-Phase AST Validation

### Phase 1: Import Resolution
Scan every `.py` file in `strands_robots/`, extract all `import` and `from X import Y` statements, resolve the top-level package, and verify the full import path exists in the installed dependency.

**Result**: 73 cross-boundary imports checked across 15 libraries.

### Phase 2: Signature Validation
For every call site where we invoke a 3rd-party function/method, extract the kwargs we pass and validate them against the actual function signature via `inspect.signature()`.

**Result**: 50 lerobot call sites validated. **18 failures found.**

### Phase 3: Version-Pinned Compatibility Matrix
Map each failure to the dependency version that introduced the break, and determine if it's:
- A **hard break** (renamed/removed API)
- A **soft break** (deprecated but still works)
- A **guard failure** (our try/except didn't catch it)

## Key Findings

### Critical Breaks (lerobot 0.5.0)

| Import Path | Status | Impact | File |
|---|---|---|---|
| `lerobot.policies.factory.get_policy_class` | REMOVED | High — lerobot_local and lerobot_async broken | policies/lerobot_local, lerobot_async |
| `lerobot.configs.default.DatasetConfig` | REMOVED | Medium — training config builder broken | training/lerobot.py |
| `lerobot.configs.default.WandBConfig` | REMOVED | Medium — training config builder broken | training/lerobot.py |
| `lerobot.configs.train.TrainPipelineConfig` | REMOVED | Medium — training pipeline broken | training/lerobot.py |
| `lerobot.scripts.lerobot_train.train` | REMOVED | High — in-process training broken | training/lerobot.py |
| `lerobot.transport.services_pb2_grpc` | REMOVED | High — gRPC async inference broken | policies/lerobot_async |
| `lerobot.transport.services_pb2.Empty` | REMOVED | High — gRPC protocol broken | policies/lerobot_async |
| `lerobot.transport.utils.send_bytes_in_chunks` | REMOVED | Medium — chunk transport broken | policies/lerobot_async |
| `lerobot.async_inference.helpers.RemotePolicyConfig` | REMOVED | Medium — async config broken | policies/lerobot_async |
| `lerobot.async_inference.helpers.TimedObservation` | REMOVED | Medium — async observation broken | policies/lerobot_async |
| `lerobot.cameras.realsense...serial_number` | RENAMED | Low — kwarg renamed to `serial_number_or_name` | tools/lerobot_camera.py |

### Safe Boundaries (0 failures)

| Library | Imports | Status |
|---|---|---|
| mujoco | 2 | ✅ Stable C API |
| numpy | 1 | ✅ Stable |
| torch | 3 | ✅ Stable |
| gymnasium | 2 | ✅ Stable |
| PIL | 1 | ✅ Stable |
| cv2 | 1 | ✅ Stable |
| huggingface_hub | 3 | ✅ Stable |

### Expected Missing (not installed in this env)

| Library | Status | Why |
|---|---|---|
| gr00t | 6 failures | GPU-only, not installed on macOS |
| stable_baselines3 | 5 failures | RL optional, not installed |
| unitree_sdk2py | 2 failures | Hardware SDK, not on macOS |
| warp | 1 failure | NVIDIA GPU-only |
| openpi_client | 1 failure | Optional Pi0 client |

## Architecture Insight

The **hot path** for breakage is:

```
strands_robots/policies/lerobot_local/__init__.py
    → lerobot.policies.factory.get_policy_class()     ← REMOVED in 0.5
    → constructs policy → runs inference

strands_robots/policies/lerobot_async/__init__.py
    → lerobot.transport.services_pb2_grpc              ← REMOVED in 0.5
    → gRPC channel → remote inference

strands_robots/training/lerobot.py
    → lerobot.configs.train.TrainPipelineConfig         ← REMOVED in 0.5
    → lerobot.scripts.lerobot_train.train()             ← REMOVED in 0.5
```

These are not edge cases — they're the primary code paths for local policy inference, async policy serving, and training.

## Recommendation

1. **Create `tests/test_cross_deps.py`** — automated AST validation that runs in CI
2. **Pin lerobot version** in pyproject.toml with upper bound: `lerobot>=0.5.0,<0.6.0`
3. **Fix the 18 lerobot breaks** — the 0.5 API moved to `draccus` config system
4. **Add signature snapshot tests** — capture 3rd-party signatures, fail CI when they change
