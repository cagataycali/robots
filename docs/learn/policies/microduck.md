---
description: microduck runs Pollen Robotics' ONNX locomotion skills for the 14-DOF Microduck biped, configured from the graph's own metadata.
---

# microduck

By the end of this page you can make the Microduck walk, stand, or run any other shipped skill from a single ONNX file, hot-swap skills mid-rollout, and know what the observation must carry.

```bash
pip install 'strands-robots[microduck]'    # onnxruntime + huggingface_hub; CPU is enough
```

## What it is

The Microduck is Pollen Robotics' open 14-DOF biped. Its skills ship as ONNX actors (`alpha_walking`, `alpha_stand`, `alpha_sitstand`, `roulade`, `ball_kick_*`, `roller*`, `alpha_ground_pick`) with the input normaliser fused into the graph. `MicroduckPolicy` adapts one export to the `Policy` contract: the ONNX metadata carries `joint_names`, `default_joint_pos`, `action_scale` and `command_names`, so pointing the policy at a different file reconfigures it. The observation is fed raw, never re-normalised, and the decode is `motor_target = DEFAULT_POSE + action * action_scale`. The raw action feeds the next tick's `last_action` block, matching Pollen's reference deployment. `requires_images` is `False`.

A bare filename such as `alpha_walking.onnx` is fetched from `pollen-robotics/microduck-policies` on first use when it is not in the working directory.

```python title="sketch"
from strands_robots.policies import create_policy

walk = create_policy("microduck", onnx_path="alpha_walking.onnx", command=[0.15, 0.0, 0.0])
stand = create_policy("microduck_stand", onnx_path="alpha_stand.onnx")
```

## Constructor keywords

{{providers:kwargs:microduck}}

All keyword-only, no `**kwargs`. `providers` defaults to `["CPUExecutionProvider"]`; the actor is a small MLP. `command` width comes from the ONNX `command_names`; the default is all zeros, stand in place. `action_scale` must be a positive finite number: `0` would hold the default pose and discard the network.

## Observation

`build_observation` in `strands_robots/policies/microduck/observation.py` assembles `[command, base_ang_vel(3), projected_gravity(3), joint_pos, joint_vel, last_action]`, 48 non-command floats for 14 joints. The observation dict must carry the base blocks the sim backends and the Microduck driver both emit; a missing block or a non-finite value is refused with the key named.

## Per-call keywords

| keyword | shape | meaning |
|---|---|---|
| `command` | width of `command_names` | replace the whole command vector |
| `target_velocity` | `[vx, vy, omega]` or `[vx, vy]` | write the twist block; the two-component form leaves `omega` as it was, because this policy's command persists across ticks |

## Skills as a bundle

`MicroduckPolicyBundle` in `strands_robots.policies.microduck.composite` holds several `MicroduckPolicy` instances warm and exposes one as active. `bundle.switch("alpha_stand")` swaps mid-rollout. `switch_on_velocity=<threshold>` with `move_key` and `idle_key` auto-selects between two skills by the magnitude of the commanded twist each tick; both keys must name held skills when the gate is on.

```python title="sketch"
from strands_robots.policies.microduck import MicroduckPolicy
from strands_robots.policies.microduck.composite import MicroduckPolicyBundle

bundle = MicroduckPolicyBundle(
    {"walk": MicroduckPolicy(onnx_path="alpha_walking.onnx"), "stand": MicroduckPolicy(onnx_path="alpha_stand.onnx")},
    active="stand",
    switch_on_velocity=0.05,
    move_key="walk",
    idle_key="stand",
)
sim.run_policy(robot_name="microduck", policy_object=bundle, policy_kwargs={"target_velocity": [0.15, 0.0]}, duration=10.0)
```

## Skill scenes

A weight and the scene it was trained in are one pair. `Robot("microduck")` resolves the entry's declared asset, flat ground with no props; four skills need more, shipped in the same asset directory. A skill on the wrong scene is not an error: a roller policy without wheels stands, a ball kick swings at nothing, both report success.

| skill | scene | the scene adds |
|---|---|---|
| `alpha_walking`, `alpha_stand`, `alpha_sitstand`, `roulade`, `alpha_ground_pick` | `scene.xml` (the declared asset) | nothing |
| `roller`, `roller_crouch` | `scene_rollers.xml` | four passive ankle wheels |
| `ball_kick_left`, `ball_kick_right` | `scene_ball.xml` | a 70 mm ball in front of the duck |

Reach a variant by path: find `scene_rollers.xml` under `microduck/` on `get_search_paths()` and pass `Robot("microduck", urdf_path=str(scene))`. `scene_rollers.xml` inserts two wheel joints after each ankle, so a flat `qpos[7:21]` read gets wheels where `neck_pitch` and `head_pitch` sit on the default scene; the actuator order is the same on all three and `MicroduckPolicy` reads by joint name.

### The ball scene places the ball, not the kick geometry

`scene_ball.xml` declares the ball 0.3 m straight ahead; training placed it 0.09 m ahead and 0.042 m to the side of the kicking foot, so from the shipped position `ball_kick_left` reports success and misses. Teleport the ball before the rollout: write the free joint's `qpos` to that offset in the trunk's yaw frame and zero its `qvel`. The file names the joint `ball_free`, but `add_robot(name=...)` prefixes every joint, so resolve the name.

### The stance every weight was trained in

Actions decode as `default_pose + raw_action * action_scale`, so the stance is the origin of the network's output. It ships as `MICRODUCK_DEFAULT_POSE` and as the `STAND` keyframe of `scene.xml` and `scene_rollers.xml`: `Robot("microduck", urdf_path=str(scene), keyframe="STAND")` seats it and every `reset()` restores it. Without it the robot starts at the zero configuration, 0.458 rad off at the widest joint. `scene_ball.xml` declares no keyframe, so seat the stance there yourself.

## Run it

Needs the extra; the weights download on first use.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco")
sim.create_world()
sim.add_robot("microduck")
result = sim.run_policy(
    robot_name="microduck",
    policy_provider="microduck",
    policy_config={"onnx_path": "alpha_walking.onnx"},
    policy_kwargs={"target_velocity": [0.15, 0.0, 0.0]},
    duration=10.0,
    control_frequency=50.0,
)
print(result["status"])
```

## Limits

- Microduck only. The joint names come from the graph's metadata and must match the robot's.
- One skill per policy; the bundle is how several coexist.
- No re-normalisation means an export without the fused normaliser produces wrong actions with no error. Use Pollen's shipped exports.
