---
description: protomotions tracks a reference motion under physics with NVIDIA GEAR's Generalist Tracking Policy for the Unitree G1.
---

# protomotions

!!! warning "Deprecated"
    Removed in 0.7. Use [wbc](wbc.md) for G1 whole-body control.

By the end of this page you can play a reference motion clip on a simulated Unitree G1 that balances and reacts to contact, and chain it after `kimodo` so a text prompt becomes a physically tracked motion.

`onnx_path` and `yaml_path` take a local file each; a Hub id is not fetched for you.

```bash
pip install 'strands-robots[protomotions,sim-mujoco]'    # onnxruntime + pyyaml + huggingface_hub; the fences below also need mujoco
```

## What it is

`ProtoMotionsPolicy` wraps the ONNX Generalist Tracking Policy (GTP) from NVIDIA GEAR's ProtoMotions framework, BeyondMimic-trained, published at `cagataydev/protomotions-gtp-unitree-g1` (`unified_pipeline.onnx` plus its `unified_pipeline.yaml` sidecar). Each tick it reads root and anchor rotation plus joint position and velocity, looks ahead into a reference window from a `MotionPlayer`, and emits PD joint targets for the G1's 29 actuators. `requires_images` is `False`. Output is smoothed by the config's `action_ema_alpha` (`1.0` is passthrough).

```python title="sketch"
from strands_robots.policies import create_policy

policy = create_policy("protomotions", onnx_path="unified_pipeline.onnx", yaml_path="unified_pipeline.yaml", motion="walk.npz")
policy = create_policy("gtp_g1", onnx_path="unified_pipeline.onnx", motion="walk.npz")   # same provider
```

## Constructor keywords

{{providers:kwargs:protomotions}}

All keyword-only and no `**kwargs`, so a typo is a `TypeError` at build time. `yaml_path` omitted falls back to `ProtoMotionsConfig` defaults, which match the shipped weights. `motion` may be a `MotionPlayer`, a cache dict, or a `.npz` / `.pt` path, and may be `None` at build time. `providers` defaults to `["CUDAExecutionProvider", "CPUExecutionProvider"]`. `history_length=1` matches the upstream checkpoint.

## Per-call keywords

| keyword | meaning |
|---|---|
| `motion` | swap in a new reference without rebuilding the policy (also `policy.load_motion(...)`) |
| `anchor_rot_xyzw`, `root_ang_vel_local` | supply these from an IMU on hardware instead of deriving them from the observation |

The observation may carry the flat `observation.state` or the joint names in `GTP_G1_JOINT_NAMES` directly; either shape works.

## From a prompt to a tracked motion

`kimodo` samples a `qpos` trajectory. `strands_robots.policies.protomotions.bridge.qpos_to_motion_data(qpos, fps, proto_mjcf_path, control_dt=0.02)` converts it into a `MotionPlayer` cache (body positions and rotations, finite-difference velocities). This policy tracks that cache.

```python title="sketch"
from strands_robots.policies import create_policy
from strands_robots.policies.kimodo import KimodoConfig
from strands_robots.policies.kimodo._diffusers_agent import DiffusersKimodoAgent
from strands_robots.policies.protomotions.bridge import qpos_to_motion_data

config = KimodoConfig()
agent = DiffusersKimodoAgent(config)                                # the sampler KimodoPolicy builds internally
qpos = agent.sample("a person waving with the right hand", num_frames=120, diffusion_steps=100, guidance_scale=7.5, seed=0)
cache = qpos_to_motion_data(qpos, fps=config.native_fps, proto_mjcf_path="g1_bm_no_mesh_box_feet.xml")   # (frames, 7 + 29) -> MotionPlayer cache

tracker = create_policy("protomotions", onnx_path="unified_pipeline.onnx", motion=cache)
```

The MJCF is `g1_bm_no_mesh_box_feet.xml` from `NVlabs/ProtoMotions` (`curl -LO https://raw.githubusercontent.com/NVlabs/ProtoMotions/main/protomotions/data/assets/mjcf/g1_bm_no_mesh_box_feet.xml`); its three meshed siblings need Git LFS, otherwise the parse fails with `decoder failed for mesh file`.

## Run it

Needs the extra and the two artifact files.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco")
sim.create_world()
sim.add_robot("g1")
result = sim.run_policy(
    robot_name="g1",
    policy_provider="protomotions",
    policy_config={"onnx_path": "unified_pipeline.onnx", "yaml_path": "unified_pipeline.yaml", "motion": "walk.npz"},
    duration=5.0,
    control_frequency=50.0,
)
print(result["status"])
```

## Limits

- Unitree G1 only, 29 actuators, the ordering in `GTP_G1_JOINT_NAMES`.
- The tracker needs a reference. With `motion=None` and no per-call `motion`, `get_actions` refuses.
- The ONNX runs on CPU when no CUDA provider is available, at a lower rate; the default provider list falls through to CPU.
