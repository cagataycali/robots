---
description: curobo plans collision-free joint trajectories in process on a CUDA GPU from a Cartesian or joint goal.
---

# curobo

!!! warning "Deprecated, removed in 0.7"
    Use `simulation.motion_primitives` with mink IK in sim, or Isaac cuMotion.

By the end of this page you can hand a motion planner a `target_pose` or `target_joints` goal and stream the resulting collision-free trajectory to an arm in `action_horizon` sized chunks, with no server in between.

```bash
pip install 'strands-robots[curobo]'    # the extra is EMPTY: install nvidia-curobo from source yourself
```

## What it is

`CuroboPolicy` is a thin wrapper around NVIDIA cuRobo's `MotionPlanner`. Unlike [moveit2](moveit2.md), which talks to a ROS 2 sidecar, cuRobo is a CUDA library running in this process: no network round trip, but a CUDA GPU is required. The module targets cuRobo's restructured `main` API (`curobo.motion_planner.MotionPlanner`, `MotionPlannerCfg`, `curobo.types.DeviceCfg`, `GoalToolPose`), not the `0.7.x` series.

Like the rest of the non-VLA family, `requires_images` is `False` and the goal arrives through the well-known keywords `target_pose`, `target_joints` and `world_update`. The first call plans the whole trajectory and caches it; each following call yields the next `action_horizon` waypoints, so the 50 Hz loop streams targets without re-planning.

```python title="sketch"
from strands_robots.policies import create_policy

policy = create_policy("curobo", robot_config="franka.yml", action_horizon=16)
actions = policy.get_actions_sync(
    {"observation.state": [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]},
    "reach the red block",                                # ignored by a planner
    target_pose=[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],      # [x, y, z, qw, qx, qy, qz] in the base frame
)
```

## Constructor keywords

{{providers:kwargs:curobo}}

`robot_config` is a path to, or a dict of, a cuRobo robot description; cuRobo ships `franka.yml`, `ur5e.yml` and many more under `curobo/content/configs/robot/`. `world_config` is the initial collision scene (`cuboid`, `mesh`, `sphere`, `capsule` keyed by name) and is forwarded as `scene_model=`; `None` plans in free space. `action_horizon` shares the chunk-count domain with every other provider's `actions_per_step`. `tensor_args` and `motion_gen_kwargs` are the legacy `0.7.x` spellings of `device_cfg` and `motion_planner_kwargs` and still resolve. `motion_gen` injects a pre-built planner and is a test seam; production callers pass `robot_config`. `warmup=True` pays the JIT cost at construction instead of on the first call.

## Goals

| keyword | shape | meaning |
|---|---|---|
| `target_pose` | `[x, y, z, qw, qx, qy, qz]` | Cartesian goal for the tool frame, metres and a unit quaternion in the robot base frame |
| `target_joints` | `{joint_name: value}` | joint-space goal, radians or metres |
| `world_update` | `dict` or `None` | per-call collision refresh, forwarded to `MotionPlanner.update_scene`; `None` reuses the init scene |

The start state is read through `policies/_state_keys.py`: the flat `observation.state` when present, otherwise the per-joint scalars in observation order minus their `.vel` siblings.

## Run it

Needs a CUDA GPU and cuRobo installed. The sim's Panda joint names are `joint1..joint7` plus the fingers; cuRobo's `franka.yml` plans `panda_joint1..7`, so a `target_joints` goal is spelled in the robot's names and the planner's own config carries its ordering.

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco")
sim.create_world()
sim.add_robot("panda")
result = sim.run_policy(
    robot_name="panda",
    policy_provider="curobo",
    policy_config={"robot_config": "franka.yml"},
    policy_kwargs={"target_pose": [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]},
    n_steps=100,
    control_frequency=50.0,
)
print(result["status"])
```

## Limits

- No CPU fallback. Without CUDA the import fails, and `create_policy` names the provider and the missing module.
- One `CuroboPolicy` per worker. The planner state lives on one CUDA device and is not shared across processes.
- The trajectory is cached on the first call; a new goal keyword on a later `get_actions` re-plans.
- The `[curobo]` extra is empty on purpose: cuRobo is not on PyPI in a form the lockfile can pin.
