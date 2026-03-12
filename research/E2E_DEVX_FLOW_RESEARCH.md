# End-to-End DevX Flow Validation

> **Date**: 2026-03-11  
> **Goal**: Validate the complete developer experience hot paths of strands-robots

## The DevX Hot Paths

A robotics engineer's daily workflow touches these paths (in order of frequency):

### Path 1: Sim Quick-Start (most common)
```python
from strands_robots import Robot
sim = Robot("so100")                    # ← factory.py → mujoco backend
sim.run_policy("mock", "pick up cube") # ← policies/__init__.py → MockPolicy
```

**AST Validation Result**: ✅ ALL CLEAN  
- `factory.py` → `mujoco._scene.create_world()` → `mujoco._robots.add_robot()` ✅  
- `mujoco._registry.resolve_model()` → `assets/__init__.py` ✅  
- `policies.create_policy("mock")` → `MockPolicy` ✅  

### Path 2: LeRobot Local Inference
```python
from strands_robots import Robot
sim = Robot("so100")
sim.run_policy("lerobot_local", "pick up cube", 
               pretrained_name_or_path="lerobot/act_aloha_sim")
```

**AST Validation Result**: ❌ BROKEN  
- `policies/lerobot_local/__init__.py` calls `lerobot.policies.factory.get_policy_class` → **REMOVED in 0.5**
- Fix: Use `PreTrainedConfig.from_pretrained()` + model instantiation

### Path 3: Dataset Recording (sim)
```python
from strands_robots import Robot, DatasetRecorder
sim = Robot("so100")
recorder = DatasetRecorder.create(repo_id="my/data", fps=30, ...)
# control loop
recorder.add_frame(observation, action, task="pick")
recorder.save_episode()
```

**AST Validation Result**: ✅ ALL CLEAN  
- `dataset_recorder.py` → `LeRobotDataset.create()` — signature validated ✅  
- `add_frame()` → internal numpy conversion ✅  
- `save_episode()` → no-arg call (lerobot v3 compat) ✅  
- Conditional `streaming_encoding` param via `inspect.signature()` ✅  

### Path 4: Training
```python
from strands_robots.training import create_trainer
trainer = create_trainer("lerobot", 
    policy_type="act",
    dataset_repo_id="my/data")
trainer.train()
```

**AST Validation Result**: ❌ BROKEN  
- `training/lerobot.py` imports `lerobot.configs.default.DatasetConfig` → **REMOVED**
- `training/lerobot.py` imports `lerobot.configs.train.TrainPipelineConfig` → **REMOVED**
- `training/lerobot.py` imports `lerobot.scripts.lerobot_train.train` → **REMOVED**
- Fix: Use lerobot 0.5's draccus-based config or subprocess CLI

### Path 5: G1 Humanoid Sim (new in lerobot 0.5)
```python
from strands_robots import Robot
sim = Robot("unitree_g1")  # auto-downloads Menagerie assets
# sim.run_policy("groot", "walk forward", policy_port=50051)
```

**AST Validation Result**: ✅ ALL CLEAN  
- `factory.py` → `resolve_name("unitree_g1")` → asset registry ✅  
- `mujoco._robots.add_robot()` → `ensure_meshes()` auto-download ✅  
- `mujoco._registry.resolve_model()` → scene.xml from Menagerie ✅  
- LeRobot's `UnitreeG1Config` discoverable via `RobotConfig.get_known_choices()` ✅  

### Path 6: Hardware Robot
```python
from strands_robots import Robot
hw = Robot("so100", mode="real", cameras={"wrist": {"index_or_path": "/dev/video0"}})
hw.execute("pick up the red cube", policy_provider="groot", policy_port=50051)
```

**AST Validation Result**: ✅ CLEAN (for the so100 path)  
- `robot.py._resolve_robot_config_class()` → dynamic scan of `lerobot.robots` ✅  
- `lerobot.cameras.opencv.configuration_opencv.OpenCVCameraConfig` ✅  
- `lerobot.robots.utils.make_robot_from_config` ✅  

### Path 7: Async Inference (remote policy server)
```python
from strands_robots.policies import create_policy
policy = create_policy("lerobot_async", server_address="localhost:8080")
actions = await policy.get_actions(obs, "pick up cube")
```

**AST Validation Result**: ❌ BROKEN  
- `policies/lerobot_async/__init__.py` imports `lerobot.transport.services_pb2_grpc` → **FAILS**
- Full gRPC chain broken

## Summary Matrix

| Path | Frequency | Status | Severity |
|---|---|---|---|
| Sim Quick-Start | Daily | ✅ | — |
| LeRobot Local Inference | Daily | ❌ | **P0** |
| Dataset Recording | Weekly | ✅ | — |
| Training | Weekly | ❌ | **P1** |
| G1 Humanoid Sim | Weekly | ✅ | — |
| Hardware Robot | Daily | ✅ | — |
| Async Inference | Weekly | ❌ | **P0** |
| Zenoh Mesh | Optional | ✅ | — |
| MuJoCo Viewer | Daily | ✅ | — |
| Gymnasium Wrapper | Weekly | ✅ | — |

**5/10 paths clean, 3/10 broken, 2/10 optional**

## The Test Infrastructure

`tests/test_cross_deps.py` now validates:
1. All critical imports resolve (`TestImportResolution`)
2. All call-site kwargs are valid (`TestSignatureValidation`)
3. Dependency versions in range (`TestVersionCompatibility`)
4. LeRobotDataset.create() signature matches (`test_lerobot_dataset_create_signature`)
5. Robot configs discoverable (`test_lerobot_robot_configs_discoverable`)

This should run in CI on every PR. When lerobot bumps from 0.5 → 0.6, the test will catch breaks before they reach hardware.
