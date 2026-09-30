---
description: moveit2 plans through a ROS 2 MoveIt2 sidecar over ZMQ and msgpack, keeping ROS out of the agent process.
---

# moveit2

!!! warning "Deprecated, removed in 0.7"
    Send the MoveIt goal as a ROS 2 action through `use_ros` or `use_rosbridge` ([ROS 2](../ros2.md)).

This page plans a collision-free trajectory with MoveIt2 from a Python process that has no ROS 2 sourced, and streams it to an arm.

```bash
pip install 'strands-robots[moveit2]'    # pyzmq + msgpack; ROS 2 stays in the sidecar
```

## What it is

`MoveIt2Policy` is a ZMQ and msgpack client. A sidecar ROS 2 node running `moveit_py` receives a goal and returns a joint trajectory; this process unpacks it into per-tick action dicts. The reference sidecar is import-only Python under `strands_robots/policies/moveit2/server/`, with a `docker-compose.yml` beside it as the recommended deployment. One sidecar can serve several agent processes.

```python title="sketch"
from strands_robots.policies import create_policy

policy = create_policy("moveit2", host="127.0.0.1", port=5556, planning_group="panda_arm")   # the compose sidecar's default group, 7 joints
actions = policy.get_actions_sync(
    {"observation.state": [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853]},   # the Panda ready pose; its zero pose is in collision
    "reach for the red block",
    target_pose=[0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
)
```

## Constructor keywords

{{providers:kwargs:moveit2}}

`host` defaults to loopback; you opt into network exposure. `port` is an `int` in `[1, 65535]`. `api_token` falls back to `MOVEIT2_API_TOKEN`. `joint_name_map` renames the planner's joint names onto the robot's action keys when the MoveIt config and the driven robot describe one arm in two vocabularies: MoveIt's panda config plans `panda_joint1`, the MuJoCo Panda drives `joint1`. Values must be distinct, because they key the action dict.

## Wire protocol

```python title="sketch"
request = {
    "joint_state": list[float] | None,
    "target_pose": [x, y, z, qw, qx, qy, qz] | None,
    "target_joints": dict[str, float] | None,
    "planning_group": str,
    "world_update": dict | None,
}
response = {"trajectory": [[t0, q0_0, q0_1, ...], ...], "success": bool, "status": str}
```

Goals use the same well-known keywords as [curobo](curobo.md): `target_pose`, `target_joints`, `world_update`, plus a per-call `planning_group` that overrides the constructor default. The start state is read through `policies/_state_keys.py`.

## Run it

Start the sidecar first (a ROS 2 environment with MoveIt2 and `moveit_py`; the compose file builds one):

```bash
cd strands_robots/policies/moveit2/server && docker compose up
```

```python title="sketch"
from strands_robots.simulation import create_simulation

sim = create_simulation("mujoco", mesh=False)
sim.create_world()
sim.add_robot("panda")
result = sim.run_policy(
    robot_name="panda",
    policy_provider="moveit2",
    policy_config={
        "port": 5556,
        "planning_group": "panda_arm",
        "joint_name_map": {f"panda_joint{i}": f"joint{i}" for i in range(1, 8)},
    },
    policy_kwargs={"target_pose": [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]},
    n_steps=100,
    control_frequency=50.0,
)
print(result["status"])
```

## Limits

- The sidecar owns the planning scene and the robot description. Collision objects reach it only through `world_update`.
- A timeout (`timeout_ms`, default 15 s) is a failed plan, not a retry.
- One action dict per trajectory waypoint. The sidecar's waypoint spacing, not the control rate, decides how fast the arm moves; the client does not resample.
- An empty trajectory with `success=True` is refused rather than unpacked to zero actions.
