---
description: wbc and wbc_gait run NVIDIA's GR00T Whole-Body-Control ONNX locomotion controllers for the Unitree G1 in process.
---

# wbc

By the end of this page you can make a simulated Unitree G1 walk from a `[vx, vy, omega]` command with the GR00T Whole-Body-Control (SONIC) controllers, know which checkpoint files it expects, and know why the MuJoCo backend installs a torque shim for it.

```bash
pip install 'strands-robots[wbc]'    # onnxruntime + pyyaml + huggingface_hub; no weights bundled
```

## What it is

`WBCPolicy` ports the non-gait reference runner from NVlabs/GR00T-WholeBodyControl (`run_mujoco_gear_wbc.py`): an 86-wide observation, a 7-wide command block, and two ONNX policies, a balance `policy.onnx` and a `walk_policy.onnx` selected by commanded velocity. It drives the 15 leg and waist joints of the G1 and holds the arm joints at their nominal defaults. `requires_images` is `False`; the controller reads joint state and base IMU only. Layer an arm policy on top with [`CompositePolicy`](../../reference/api/policies.md#built-in-policies) (legs and waist from `wbc`, arms from a manipulation policy).

`WBCGaitPolicy` (`wbc_gait`) ports the gait-clock variant (`run_mujoco_gear_wbc_gait.py`): a 95-wide observation with a step-frequency command slot and a two-element left/right foot phase clock, and a single ONNX policy whose input is `[batch, 570]`. Everything else (SONIC PD gains, name-resolved joint map, checkpoint resolution) is inherited.

```python title="sketch"
from strands_robots.policies import create_policy

policy = create_policy("wbc", checkpoint="./GR00T-WholeBodyControl", target_velocity=[0.5, 0.0, 0.0])
policy = create_policy("sonic", checkpoint="./GR00T-WholeBodyControl")             # same provider
gait = create_policy("wbc_gait", checkpoint="./gait-ckpt", gait_frequency=1.5)
```

## Constructor keywords

`wbc`:

{{providers:kwargs:wbc}}

`wbc_gait`:

{{providers:kwargs:wbc_gait}}

`checkpoint` is a directory holding the ONNX files and an optional `config.json`, a direct path to the main `.onnx`, or a HuggingFace model id. The loader accepts the official artifact names `GR00T-WholeBodyControl-Balance.onnx` and `-Walk.onnx` verbatim, so you do not rename the download. When a G1 checkpoint ships ONNX only, the SONIC gains and default angles for 15 actuators are applied. `walk` is a strict boolean; `"false"` is refused, not read as truthy. `target_velocity` in the constructor is the default command for paths that forward constructor kwargs only, such as the mesh `tell()`; the per-call keyword overrides it.

The repo `nvidia/GEAR-SONIC` ships the SONIC VLA inference stack (`model_encoder.onnx`, `planner_sonic.onnx`, ...), not these controllers; pointing `checkpoint` at it is refused with the reason. Its decoder is what [`wbc_latent`](wbc-latent.md) runs, for a VLA that predicts SONIC motion tokens.

## Goals

| keyword | shape | meaning |
|---|---|---|
| `target_velocity` | `[vx, vy, omega]` | m/s, m/s, rad/s in the base frame; three components required here |
| `target_orientation` | `[roll, pitch, yaw]` | optional torso orientation command |

## Run it

Needs the extra and a downloaded checkpoint. `run_policy` on MuJoCo detects a `WBCPolicy` anywhere in the policy tree and installs `WBCTorqueController`, which applies SONIC's per-joint PD law to the compiled model; without it the scene's position servos override the controller and the robot falls within a fraction of a second while the rollout reports success. The Isaac and Newton backends cannot install the shim and refuse to start the rollout unless you pass `wbc_install_torque_control=False` against a torque-actuated scene.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco", mesh=False)
sim.create_world()
sim.add_robot("g1")
result = sim.run_policy(
    robot_name="g1",
    policy_provider="wbc",
    policy_config={"checkpoint": "./GR00T-WholeBodyControl"},
    policy_kwargs={"target_velocity": [0.5, 0.0, 0.0]},
    duration=10.0,
    control_frequency=50.0,
)
print(result["status"])
```

`Robot("g1")` on hardware takes the same `policy_provider` and `policy_config`.

## Limits

- Unitree G1 only. The joint map resolves the G1's 29 names by name inside the caller's state keys; a different humanoid fails that membership check.
- `wbc` has no reference-pose input. It cannot track a [kimodo](kimodo.md) motion; that pairing is [protomotions](protomotions.md).
- `wbc_gait` needs a gait-clock checkpoint (`[batch, 570]` input, `[batch, 15]` output). The shipped Balance and Walk weights are the non-gait family (516-wide) and are rejected by shape.
- Weights are fetched under the NVIDIA Open Model License; nothing is bundled.
