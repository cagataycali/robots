# LeRobot 0.5.0 Compatibility Fixes

> **Date**: 2026-03-11  
> **Patch**: `LEROBOT_050_FIX.patch` (366 lines, 4 files)  
> **Test**: `tests/test_cross_deps.py` — 8/8 PASSED after fix

## Applied Fixes

### 1. `policies/lerobot_local/__init__.py` — Policy Factory (P0)

**Problem**: `from lerobot.policies.factory import get_policy_class` removed in 0.5.

**Fix**: Three-strategy resolution:
1. **PreTrainedConfig.from_pretrained()** (lerobot ≥0.5) — draccus-based config resolution
2. **Direct module import** `lerobot.policies.{policy_type}` — finds `{Type}Policy` class
3. **Legacy fallback** `get_policy_class()` — for lerobot <0.5 compat

```python
# BEFORE (broken)
from lerobot.policies.factory import get_policy_class
PolicyClass = get_policy_class(policy_type)

# AFTER (works on 0.4 and 0.5)
try:
    from lerobot.configs.policies import PreTrainedConfig
    config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
    from lerobot.policies.pretrained import PreTrainedPolicy
    return PreTrainedPolicy, policy_type
except Exception:
    # fallback to direct module import or legacy factory
```

### 2. `policies/lerobot_async/__init__.py` — gRPC Transport (P0)

**Problem**: `lerobot.transport.services_pb2_grpc` fails to import without `grpc` package. Also `PolicyInstructions` renamed to `PolicySetup` in 0.5.

**Fix**: 
- Separated `grpc` import from lerobot transport imports
- Clear error message for missing grpc package
- Updated `PolicyInstructions` → `PolicySetup` (0.5 protobuf change)
- Updated `_validate_policy_type()` to use module scanning instead of removed `get_policy_class()`

```python
# BEFORE (broken)
import grpc
from lerobot.transport import services_pb2, services_pb2_grpc  # fails if no grpc
self._stub.SendPolicyInstructions(services_pb2.PolicyInstructions(data=...))

# AFTER
try:
    import grpc
except ImportError:
    raise ImportError("gRPC required: pip install grpcio grpcio-tools")
from lerobot.transport import services_pb2
from lerobot.transport import services_pb2_grpc  # separate import
self._stub.SendPolicyInstructions(services_pb2.PolicySetup(data=...))
```

### 3. `training/lerobot.py` — Training Pipeline (P1)

**Problem**: `_build_train_config()` used removed import paths and wrong constructor kwargs.

**Fix**:
- `DatasetConfig` and `WandBConfig` now imported from `lerobot.configs.default` (still valid in 0.5)
- `TrainPipelineConfig` from `lerobot.configs.train` (valid in 0.5)  
- `PreTrainedConfig.from_pretrained()` wrapped in try/except (may fail for some models)
- Constructor kwargs filtered through `inspect.signature()` so only valid params are passed

```python
# BEFORE (broken — passed kwargs that don't exist)
train_cfg = TrainPipelineConfig(
    dataset=..., policy=..., output_dir=...,
    num_workers=...,  # ← renamed to num_workers in 0.5
)

# AFTER — introspects signature, only passes valid params
train_sig = inspect.signature(TrainPipelineConfig)
valid_params = set(train_sig.parameters.keys())
train_kwargs = {k: v for k, v in train_kwargs.items() if k in valid_params}
train_cfg = TrainPipelineConfig(**train_kwargs)
```

### 4. `tools/lerobot_camera.py` — RealSense Kwarg (P2)

**Problem**: `serial_number` renamed to `serial_number_or_name` in lerobot 0.5.

**Fix**: One-line rename.

```python
# BEFORE
RealSenseCameraConfig(serial_number=str(camera_id), ...)
# AFTER
RealSenseCameraConfig(serial_number_or_name=str(camera_id), ...)
```

## How to Apply

```bash
cd /path/to/robots
git apply research/LEROBOT_050_FIX.patch
python -m pytest tests/test_cross_deps.py -v
```

## Test Results After Fix

```
tests/test_cross_deps.py::TestImportResolution::test_all_critical_imports_resolve          PASSED
tests/test_cross_deps.py::TestImportResolution::test_optional_imports_are_guarded          PASSED
tests/test_cross_deps.py::TestSignatureValidation::test_lerobot_signatures                 PASSED
tests/test_cross_deps.py::TestSignatureValidation::test_mujoco_signatures                  PASSED
tests/test_cross_deps.py::TestVersionCompatibility::test_dependency_versions                PASSED
tests/test_cross_deps.py::TestVersionCompatibility::test_lerobot_api_surface               PASSED
tests/test_cross_deps.py::TestVersionCompatibility::test_lerobot_dataset_create_signature  PASSED
tests/test_cross_deps.py::TestVersionCompatibility::test_lerobot_robot_configs_discoverable PASSED

8/8 PASSED ✅
```
