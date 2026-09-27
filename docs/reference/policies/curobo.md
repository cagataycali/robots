---
description: NVIDIA cuRobo collision-aware motion planning - in-process CUDA, no network round-trip, goal via target_pose / target_joints kwargs.
---

# cuRobo

[`CuroboPolicy`](https://github.com/strands-labs/robots/blob/main/strands_robots/policies/curobo/policy.py)
wraps NVIDIA [cuRobo](https://curobo.org/)'s `MotionPlanner`. Unlike the
sidecar VLA providers (GR00T, Cosmos 3), cuRobo runs **in the same process**
as a CUDA library: there is no network round-trip, but a CUDA-capable GPU is
required. It is a non-VLA, collision-aware motion planner - it reads its goal
from `**kwargs` (`target_pose` / `target_joints`), ignores camera frames
(`requires_images = False`), and never parses the instruction string for
control.

## Install

cuRobo is **not** on PyPI (the `nvidia-curobo` PyPI package is an unrelated
v0.1 squatter). Install from the upstream source repository, then this
package:

```bash
uv pip install torch                     # cuRobo's own install declares no torch
git clone https://github.com/NVlabs/curobo.git
uv pip install -e ./curobo
uv pip install 'cuda-core[cu12]'         # its CUDA kernel backend, no compilation
```

Verified against cuRobo `main` at `78fd485` on CUDA 12.8. Neither trailing
install is optional: without torch the constructor raises
`ModuleNotFoundError`, and without a kernel backend the first plan raises
`RuntimeError: No curobo kernel backend available!` (compile one instead with
`CUROBO_USE_PYBIND=1 pip install -e ./curobo --no-build-isolation`).

The `[curobo]` extra exists but is empty - reserved for a future stable cuRobo
PyPI wheel - so `pip install "strands-robots[curobo]"` exits 0 and installs
nothing. Constructing a `CuroboPolicy` without cuRobo raises an `ImportError`
that carries the checkout recipe above rather than that extra.

This policy targets cuRobo's restructured `main` API (`MotionPlanner` /
`MotionPlannerCfg` / `DeviceCfg` / `JointState` / `GoalToolPose`). The
on-device cuRobo APIs are still moving on `main` until upstream cuts a stable
release; if you hit a fresh API shift, pin to a known-good commit or open an
issue with the cuRobo SHA you tested.

## Quickstart

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "curobo",                      # alias: "cumotion"
    robot_config="franka.yml",     # any cuRobo built-in YAML, or a dict
    action_horizon=16,
)

actions = policy.get_actions_sync(
    observation_dict={
        "observation.state": [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854],
    },
    instruction="reach for the red block",   # ignored by planners
    target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],   # [x,y,z, qw,qx,qy,qz]
)
```

## Parameters

```python
CuroboPolicy(
    robot_config="franka.yml",     # cuRobo built-in YAML name, a path, or a dict
    world_config=None,             # initial collision world (dict | None)
    action_horizon=16,             # waypoints streamed per get_actions() call
    device_cfg=None,               # None | "cuda:0" | torch.device | DeviceCfg
    motion_planner_kwargs=None,    # extra kwargs forwarded to MotionPlannerCfg.create
    motion_gen=None,               # pre-built planner (tests / advanced); skips build
    warmup=True,                   # warm the planner at construction
    # Legacy 0.7.x aliases (pass only one of each pair):
    tensor_args=None,              # alias of device_cfg=
    motion_gen_kwargs=None,        # alias of motion_planner_kwargs=
)
```

Supplying both a canonical kwarg and its legacy alias (e.g. `device_cfg=` and
`tensor_args=`) raises `ValueError` rather than silently picking one.

## Goal kwargs

cuRobo shares the non-VLA goal vocabulary with the rest of the planner family,
so a goal can flow across providers without coupling to a backend:

| Key | Type | Meaning |
|-----|------|---------|
| `target_pose` | `list[float]` | Cartesian goal `[x, y, z, qw, qx, qy, qz]` in the base frame |
| `target_joints` | `dict[str, float]` | Joint-space goal keyed by joint name (rad / m) |
| `world_update` | `dict \| None` | Per-call collision-scene refresh |
| `replan` | `bool` | Force a fresh plan even if cached waypoints remain |

Pass exactly one of `target_pose` / `target_joints`; both at once is refused
with a `ValueError`, since they name two different plans. When neither is given,
the policy makes a best-effort parse of a JSON `target_pose` / `target_joints`
payload embedded in the instruction (for LLM-agent flows); if none is found it
raises `ValueError`. Each `{...}` object in the instruction is decoded on its
own, so prose braces before or after the payload are harmless, and only a
**top-level** goal field counts - a goal nested inside a wrapper object
(`{"goal": {"target_pose": [...]}}`) is not read as a goal.

## Trajectory chunking

The full collision-free trajectory is planned and cached on the first call.
Each subsequent call yields up to `action_horizon` waypoints from the cache so
the 50Hz execution loop in `Robot` streams per-step joint targets without
re-planning. Force a fresh plan with `replan=True` (or `policy.reset()`) when
the world changes mid-rollout. `world_update` is forwarded to
`MotionPlanner.update_scene` (with a legacy `update_world` fallback) for
per-call collision-scene refresh.

Every waypoint of a plan must carry the same non-zero number of joint positions.
The joint key names are resolved once per chunk from the width of that chunk's
first waypoint, so a waypoint of a different width would be commanded partially -
a narrower one leaves its trailing joints holding mid-motion, a wider one has its
trailing positions dropped, and one carrying no position at all becomes an action
dict that moves no joint. A planner's degree-of-freedom count does not change
mid-plan, so a plan that is not rectangular is refused with a `RuntimeError`
naming the waypoint and both widths, and nothing is cached. The check runs when
the plan is cached rather than when a chunk of it is served, so the refusal
arrives before any waypoint of a broken plan has moved the robot - and so the
verdict does not depend on `action_horizon`, which only sets the chunk width.
`MoveIt2Policy` refuses a positionless waypoint for the same reason: a plan that
commands nothing is a planning failure, not a successful no-op plan.

## In simulation

```python
from strands_robots import Robot

sim = Robot("panda")              # sim-by-default; needs a CUDA GPU for cuRobo
# cuRobo plans from the robot's current configuration and refuses one outside
# the joint limits - the Panda model rests at zero, which is outside its own
# elbow range, so start from the model's home keyframe.
sim.set_joint_positions(
    dict(
        zip(
            sim.robot_joint_names("panda"),
            [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853, 0.04, 0.04],
            strict=True,
        )
    ),
    robot_name="panda",
)
sim.run_policy(
    robot_name="panda",
    instruction="",               # ignored by the planner
    policy_provider="curobo",
    policy_config={"robot_config": "franka.yml", "action_horizon": 16},
    policy_kwargs={"target_pose": [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]},
    duration=10.0,
    control_frequency=50.0,
)
```

That rollout takes the Panda's tool frame from 231 mm away to 6.2 mm from the
commanded `target_pose`. The start configuration is read from the observation -
the flat `observation.state` vector when a caller passes one, and the per-joint
scalars a simulation publishes otherwise - and is projected onto the joints
cuRobo plans over, which excludes any joint the robot configuration locks (the
two Panda fingers). An observation carrying neither shape is refused rather than
planned from a default pose.

`policy_config` and `policy_kwargs` are two different sinks. `policy_config`
is expanded into the policy **constructor**; the per-call goal belongs in
`policy_kwargs`, which the runner hands to every `get_actions()` call. Passing
`target_pose=` directly to `run_policy` raises `TypeError` - it has no such
parameter and no `**kwargs`.

The mesh path forwards the same goal vocabulary:
`mesh.tell(peer, "...", policy_provider="curobo", target_pose=[...])`.
`Robot.start_task` forwards the same keywords through `**policy_kwargs` to
`create_policy`, so the hardware path accepts the goal vocabulary the mesh
dispatch collects from the wire command.

## See also

- [Policy overview](overview.md)
- [MoveIt2](moveit2.md) - ROS 2 sidecar collision-aware planning (non-VLA).
- [GR00T](groot.md) - ZMQ service VLA.
- [Cosmos 3](cosmos3.md) - WebSocket VLA.
- [LeRobot Local](lerobot-local.md) - in-process HF models.
- [Custom policies](custom-policies.md) - implement the non-VLA goal-kwargs contract.
- [cuRobo project](https://curobo.org/)
