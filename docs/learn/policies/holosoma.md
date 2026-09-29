---
description: holosoma runs Amazon FAR's Holosoma ONNX locomotion controllers for the Unitree G1 in process, next to the GR00T-WBC family.
---

# holosoma

By the end of this page you can make a simulated Unitree G1 walk from a `[vx, vy, omega]` command with Amazon FAR's Holosoma controllers, know where the weights come from and under which licence, and know what this family shares with [wbc](wbc.md).

```bash
pip install 'strands-robots[holosoma]'    # onnxruntime + huggingface_hub; weights fetched on first use
```

## What it is

`HolosomaPolicy` ports the deployment loop of `holosoma_inference` from [amazon-far/holosoma](https://github.com/amazon-far/holosoma): one ONNX actor (`fastsac_g1_29dof.onnx` or `ppo_g1_29dof.onnx`) with input `actor_obs [1, 100]` and output `action [1, 29]`. The checkpoint describes itself: joint names, per-joint PD gains and the command ranges ride in the ONNX metadata, and the provider reads them the way upstream does (a config override wins, then the metadata, otherwise a refusal). The controller drives all 29 joints; `requires_images` is `False`.

Code and weights are Apache-2.0 in the same git tree (`src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/`). A bare file name is fetched from the Hub mirror `nepyope/holosoma_locomotion`, the repository lerobot's own `HolosomaLocomotionController` downloads; nothing is bundled.

```python title="sketch"
from strands_robots.policies import create_policy

walk = create_policy("holosoma", target_velocity=[0.5, 0.0, 0.0])            # fastsac, fetched
ppo = create_policy("holosoma", algorithm="ppo")
local = create_policy("holosoma", checkpoint="./holosoma/src/holosoma_inference/holosoma_inference/models/loco/g1_29dof")
```

## Constructor keywords

{{providers:kwargs:holosoma}}

`checkpoint` is a `.onnx` file, a directory holding the upstream file name for `algorithm`, or a bare Hub file name; a path with directories that does not exist is refused rather than downloaded. `driven_joints="legs_waist"` emits the first 15 targets only and `arm_observation="default"` feeds the network the nominal arm pose instead of the measured one; together they reproduce lerobot's convention for arm teleoperation on top of the gait. The defaults (`"all"`, `"live"`) are upstream's.

## Observation

`build_actor_obs` in `strands_robots/policies/holosoma/observation.py` lays the 100 floats out in the order upstream does: the term names sorted alphabetically, so `actions(29)`, `base_ang_vel(3)` scaled by 0.25, `command_ang_vel(1)`, `command_lin_vel(2)`, `cos_phase(2)`, `dof_pos(29)` as the offset from the default stance, `dof_vel(29)` scaled by 0.05, `projected_gravity(3)`, `sin_phase(2)`. The two-foot gait clock advances `2 pi / 50` per tick with a one second period, pins both feet to `pi` when the commanded velocity is below 0.01, and restarts on the first moving tick. Joint state is read by name from the unified sim observation (`<name>`, `<name>.vel`, `base_quat`, `base_ang_vel`) or from the `Robot("g1")` snapshot (`joints[name]["q"]`, `imu["gyroscope"]`), so one object serves both.

## Goals

| keyword | shape | meaning |
|---|---|---|
| `target_velocity` | `[vx, vy, omega]` | m/s, m/s, rad/s in the base frame; clipped to the checkpoint's ranges (1 m/s, 1 rad/s for the shipped files) with one warning |

The raw action is clipped to `[-100, 100]`, scaled by 0.25 and added to the default stance; the clipped raw value feeds the next tick's `actions` block.

## Run it

`run_policy` on MuJoCo installs the same `WBCTorqueController` the wbc family uses, because a Holosoma checkpoint also emits joint-position targets that the stock position servos would override: the shim flips the driven actuators to torque, steps physics at 0.005 s four times per control tick, and applies the checkpoint's own `kp` and `kd`. The Isaac and Newton backends cannot install it and refuse the rollout unless you pass `wbc_install_torque_control=False` against a torque-actuated scene.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco")
sim.create_world()
sim.add_robot("unitree_g1")
result = sim.run_policy(
    robot_name="unitree_g1",
    policy_provider="holosoma",
    policy_kwargs={"target_velocity": [0.5, 0.0, 0.0]},
    duration=5.0,
    control_frequency=50.0,
)
print(result["status"])
```

Measured on the Menagerie G1 at 50 Hz on a laptop CPU, real time: fastsac walks 1.9 m in 5 s at a 0.5 m/s command with under 4 cm of lateral drift and the pelvis at 0.79 m; ppo walks 1.7 m; a zero command stands with 2 cm of creep. The wbc family on the same scene walks 1.9 m.

`Robot("g1")` on hardware takes the same `policy_provider` and `policy_config`; the driver's 500 Hz loop re-gates every step.

## Next to wbc

Both families drive the 29 joints of `WBC_G1_ALL_JOINTS` in the same order, emit absolute joint targets, run at 50 Hz over a 200 Hz PD loop, and share the torque shim through the `PDTorquePolicy` protocol. They differ in what the network sees and does: wbc observes 86 or 95 floats with a height and orientation command and drives 15 joints while the arms hold; holosoma observes 100 floats with a phase clock and drives all 29. Weights differ in licence: NVIDIA Open Model License for wbc, Apache-2.0 for holosoma.

## Limits

- Unitree G1 only. `set_robot_state_keys` requires all 29 G1 joint names; the metadata `dof_names` of a checkpoint must be that table.
- Locomotion only. The `*_dancing.onnx` whole-body-tracking files in the same upstream folder take a 58-wide motion command and are refused by input width.
- Upstream trains and deploys at 50 Hz with a one second gait period; lerobot's port steps the same network at 200 Hz with a half second period. This provider follows upstream.
