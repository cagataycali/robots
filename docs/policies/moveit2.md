---
description: MoveIt2 motion planning via a ROS 2 sidecar - ZMQ + msgpack client, goal via target_pose / target_joints kwargs, no ROS 2 deps in the Python venv.
---

# MoveIt2

[`MoveIt2Policy`](https://github.com/strands-labs/robots/blob/main/strands_robots/policies/moveit2/policy.py)
is a thin ZMQ + msgpack client for a sidecar ROS 2 node running
[`moveit_py`](https://github.com/moveit/moveit2). Like
[cuRobo](curobo.md) it is a **non-VLA, collision-aware motion planner**: it
reads its goal from `**kwargs` (`target_pose` / `target_joints`), ignores
camera frames (`requires_images = False`), and never parses the instruction
string for control.

Unlike cuRobo's in-process CUDA library, MoveIt2 runs **out-of-process**: the
ROS 2 stack and `moveit_py` live entirely in a sidecar, so the Python venv
running `strands_robots` stays free of ROS 2 deps. The only client-side
requirements are `pyzmq` + `msgpack` (the `[moveit2]` extra).

## Install

```bash
pip install 'strands-robots[moveit2]'   # client side: pyzmq + msgpack only
```

The ROS 2 / `moveit_py` deps stay out of `pyproject.toml` - they live in the
sidecar. Bring the sidecar up via the docker-compose recipe or natively:

```bash
# Option 1 - docker compose (pinned, reproducible)
cd strands_robots/policies/moveit2/server
docker compose up

# Option 2 - native ROS 2 (dev loop)
source /opt/ros/jazzy/setup.bash         # or your distro
sudo apt install ros-jazzy-moveit-py ros-jazzy-moveit-configs-utils \
    ros-jazzy-moveit-planners-ompl ros-jazzy-moveit-resources-panda-moveit-config
python -m strands_robots.policies.moveit2.server.zmq_node \
    --port 5556 --planning-group panda_arm
```

The sidecar plans for the robot named by `--moveit-config-package` /
`--robot-name`, which default to the panda config MoveIt 2 itself ships - so the
command above plans out of the box, and its planning group is `panda_arm`.
Point both at your own config package for your robot. `pyzmq` + `msgpack` come
from the `[moveit2]` extra; install them into the interpreter that launches the
sidecar (on a distro with [PEP 668] system packages, that means a venv, or the
distro's own python3-zmq / python3-msgpack).

[PEP 668]: https://peps.python.org/pep-0668/

See [`policies/moveit2/server/README.md`](https://github.com/strands-labs/robots/blob/main/strands_robots/policies/moveit2/server/README.md)
for the sidecar deployment and forking guidance.

## Quickstart

```python
from strands_robots.policies import create_policy

policy = create_policy(
    "moveit2",                     # alias: "moveit"
    host="127.0.0.1",
    port=5556,
    planning_group="panda_arm",     # the group name in your MoveIt config
)

actions = policy.get_actions_sync(
    observation_dict={"observation.state": [0.0] * 6},
    instruction="reach for the red block",   # ignored by planners
    target_pose=[0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],   # [x,y,z, qw,qx,qy,qz]
)
```

The smart-string resolver also works:
`create_policy("zmq://127.0.0.1:5556", planning_group="arm")`.

## Parameters

```python
MoveIt2Policy(
    host="127.0.0.1",      # sidecar hostname (loopback only by default)
    port=5556,             # sidecar port
    planning_group="arm",  # default MoveIt2 planning group; per-call override allowed
    timeout_ms=15000,      # ZMQ send + recv timeout in milliseconds
    api_token=None,        # included in every request; falls back to MOVEIT2_API_TOKEN
)
```

`timeout_ms` is applied as `RCVTIMEO` and `SNDTIMEO` on the REQ socket, so only
a positive whole number of milliseconds up to `2**31 - 1` names a budget. An
integral `float` or NumPy integer is accepted and stored as an `int`. `0` is
ZMQ's "return immediately" spelling and `-1` its "block forever" one; both are
refused at construction, because each makes `ping()` unable to report a
reachable sidecar as reachable.

`api_token` falls back to the `MOVEIT2_API_TOKEN` environment variable when not
passed. The client emits a plaintext-over-TCP warning when `host` is a non-
loopback address (the token travels unencrypted; terminate TLS at a proxy or
keep the sidecar on loopback).

## Goal kwargs

`MoveIt2Policy` shares the non-VLA goal vocabulary with the rest of the planner
family (see [cuRobo](curobo.md)), so a goal can flow across providers without
coupling to a backend:

| Key | Type | Meaning |
|-----|------|---------|
| `target_pose` | `list[float]` | Cartesian goal `[x, y, z, qw, qx, qy, qz]` in the base frame |
| `target_joints` | `dict[str, float]` | Joint-space goal keyed by joint name (rad / m) |
| `world_update` | `dict \| None` | Per-call collision-scene refresh, forwarded to the sidecar |
| `planning_group` | `str` | Override the policy's default planning group for this call |

Provide at least one of `target_pose` / `target_joints`. If **both** are set,
`target_joints` wins (mirrors MoveIt2's `setJointValueTarget`). When neither is
given, `get_actions(...)` raises `ValueError` rather than returning a
zero-action.

### Input validation

Goals are validated up-front before they reach the wire (defence-in-depth for
LLM-agent inputs):

- `target_pose` must be exactly 7 finite floats (NaN / inf rejected).
- `target_joints` keys must match `^[A-Za-z][A-Za-z0-9_-]*$` - the same
  allowlist `mesh.security.validate_command` applies, so a value the mesh
  accepts flows end-to-end without a second mismatch. Values must be finite.
- `planning_group` must be a plain identifier (no shell metacharacters or path
  traversal).

## Wire protocol

```
request  = {"endpoint": "plan",
            "data": {"joint_state": list[float] | None,
                     "planning_group": str,
                     "target_pose": [x, y, z, qw, qx, qy, qz] | None,
                     "target_joints": dict[str, float] | None,
                     "world_update": dict | None}}
response = {"trajectory": list[list[float]],
            "joint_names": list[str],        # on success
            "success": bool,
            "status": str}
```

Trajectory rows are `[time_from_start_seconds, q0, q1, ..., qN]`, and
`joint_names` names the joint each of `q0 .. qN` belongs to, in column order -
the planning group's own vocabulary, read from the trajectory message. The
client drops the leading time column when packing per-step action dicts and
keys the remaining columns, in order of preference, by the
`set_robot_state_keys(...)` roster when the row is that wide, else by
`joint_names` (each passed through the constructor's `joint_name_map`), else by
positional `joint_<i>` labels when the sidecar returned no roster. A plan
covers the planning group, which is narrower than the robot that carries it
(`panda_arm` plans 7 joints; the Panda declares 8 action keys), so the second
path is the one a simulation rollout takes: the planner is the only party that
knows which joint a column commands, and its names are used as given rather
than guessed from the robot's roster. A name the robot does not drive stays
unresolved and the runner refuses it; a roster that is not as wide as the row,
or not a list of distinct names, is refused by the client before it keys a
command. A `success=False` response raises `RuntimeError`. The sidecar also
exposes `ping` (health check) and `reset` (per-episode hook).

A `success=True` response must carry at least one waypoint, and every row must
carry at least one joint column - a plan that commands nothing is a planning
failure, not a successful no-op plan. The sidecar reports such a plan as
`planner_returned_empty`, and the client refuses one that arrives as
`success=True` anyway, so `get_actions` never returns an empty list or an action
dict that moves no joint.

### Failure reporting

REQ/REP is lockstep, so the sidecar answers every request it receives: a
request it cannot serve comes back as a failure response, never by dropping the
reply and exiting. A planning failure is reported in the `plan` response as
`success=False` plus a `status` naming the stage that failed:

| `status` prefix | Meaning |
| --- | --- |
| `unknown_planning_group:` | The requested group does not resolve. |
| `start_state_error:` | The current robot state is not readable (no `/joint_states` yet, state monitor not warmed up). |
| `missing_goal:` | Neither `target_pose` nor `target_joints` was supplied. |
| `invalid_goal:` | The goal was rejected - a joint the group does not have, an unresolvable pose link, or a `target_pose` that is not 7 values. |
| `planner_exception:` / `planner_returned_empty` | Planning ran and failed. `planner_returned_empty` also covers a plan that serialised to no waypoint, or to waypoints with no joint position; those carry a `:detail` suffix naming which. |
| `trajectory_error:` | The planned trajectory did not serialise. |

Client-side validation checks `target_joints` key *syntax*, not whether the
group actually has those joints, so a joint-name typo is reported by the
sidecar as `invalid_goal:` rather than rejected before the call. Anything
outside the `plan` contract - an unknown endpoint, a request that does not
decode or that decodes to something other than a map, an unexpected error in a
forked handler - comes back as `{"error": ...}`, which the client raises as
`RuntimeError`. Both malformed-payload cases share the `malformed_request:`
prefix, so a client has one class to match whether the bytes were undecodable
or decoded to an integer.

## In simulation

```python
from strands_robots import Robot

sim = Robot("panda")              # sim-by-default
# MoveIt plans from the state the client sends and refuses a start state in
# collision - the Panda model rests at zero, which is outside its own elbow
# range, so start from the model's home keyframe.
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
    policy_provider="moveit2",
    policy_config={
        "host": "127.0.0.1",
        "port": 5556,
        "planning_group": "panda_arm",
        # MoveIt 2's panda config and the MuJoCo Panda are two descriptions of
        # one arm with two vocabularies: the planner names panda_joint1..7,
        # the simulated robot drives joint1..7.
        "joint_name_map": {f"panda_joint{i}": f"joint{i}" for i in range(1, 8)},
    },
    policy_kwargs={"target_pose": [0.3, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]},
    duration=10.0,
    control_frequency=50.0,
)
```

That rollout takes the Panda's flange from 283.3 mm away to 4.6 mm from the
commanded `target_pose`. The planned columns are keyed by the joint names the
sidecar returns, so without the `joint_name_map` the actions would name
`panda_joint1..7`, which the MuJoCo Panda does not drive, and the runner would
refuse the rollout as one that moved no joint - loudly, rather than commanding
whichever joints happen to lead the robot's roster. The start configuration is
read from the observation -
the flat `observation.state` vector when a caller passes one, and the per-joint
scalars a simulation publishes otherwise - and is read in order onto the joints
the planning group plans over, so a robot that publishes joints the group does
not plan (the two Panda fingers) is planned for anyway; the sidecar logs which
values it ignored. The Cartesian goal is expressed in the robot model's own
frame, for the planning group's end-effector link.

`policy_config` and `policy_kwargs` are two different sinks. `policy_config`
is expanded into the policy **constructor**; the per-call goal belongs in
`policy_kwargs`, which the runner hands to every `get_actions()` call. Passing
`target_pose=` directly to `run_policy` raises `TypeError` - it has no such
parameter and no `**kwargs`.

The mesh path forwards the same goal vocabulary:
`mesh.tell(peer, "...", policy_provider="moveit2", target_pose=[...])`.
`Robot.start_task` forwards the same keywords through `**policy_kwargs` to
`create_policy`, so the hardware path accepts the goal vocabulary the mesh
dispatch collects from the wire command.

## See also

- [Policy overview](overview.md)
- [cuRobo](curobo.md) - in-process collision-aware planning (non-VLA, GPU).
- [GR00T](groot.md) - ZMQ service VLA.
- [Cosmos 3](cosmos3.md) - WebSocket VLA.
- [Custom policies](custom-policies.md) - implement the non-VLA goal-kwargs contract.
- [MoveIt2 project](https://moveit.ai/)
