# MoveIt2 ZMQ Sidecar — Reference Deployment

This directory ships a reference ROS 2 sidecar that exposes
`moveit_py` motion planning over ZMQ + msgpack so
[`MoveIt2Policy`](../policy.py) can talk to it from a plain Python
venv.

The strands-robots client side only needs `pip install
'strands-robots[moveit2]'` (pulls `pyzmq` + `msgpack`). The ROS 2
deps live entirely in the sidecar.

## Two ways to run the sidecar

### 1. Docker compose (pinned, reproducible)

```bash
cd strands_robots/policies/moveit2/server
docker compose up
```

The `docker-compose.yml` here pulls the upstream
`moveit/moveit2:jazzy-tutorial-source` image, layers `pyzmq` + `msgpack`,
and runs `python -m strands_robots.policies.moveit2.server.zmq_node` on
container start. Port `5556` is published to the host loopback.

### 2. Native ROS 2 (recommended dev loop)

```bash
source /opt/ros/jazzy/setup.bash         # or your distro
sudo apt install ros-jazzy-moveit-py ros-jazzy-moveit-configs-utils \
    ros-jazzy-moveit-planners-ompl ros-jazzy-moveit-resources-panda-moveit-config
python -m strands_robots.policies.moveit2.server.zmq_node \
    --port 5556 --planning-group panda_arm
```

`--moveit-config-package` / `--robot-name` default to the panda config MoveIt 2
ships, so that command plans out of the box; point them at your own config
package and pass its group name. `pyzmq` + `msgpack` (the `[moveit2]` extra)
have to be importable by the interpreter that launches the sidecar.

Use this when you're iterating on the sidecar code or attaching to a
specific MoveIt2 config (custom URDF/SRDF, custom planning pipelines).

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
`joint_names` names the joint each of `q0 .. qN` belongs to, in column order,
read from the trajectory message. The client drops the time column when
packing per-step action dicts and keys the columns by those names when the
robot's declared roster is not the row's width —
`MoveIt2Policy._unpack_trajectory` in `../policy.py`.

The reference implementation also exposes `ping` (health check) and
`reset` (per-episode hook; no-op in the reference impl, override in
forks for stateful planners).

## Forking guidance

`zmq_node.py` is intentionally minimal. Hardening for a production
deployment typically involves:

* Validating `world_update` against a known schema before pushing to
  the planning scene.
* Checking `request.get("api_token")` against a secret.
* Mapping `joint_state` onto the planning group by joint name rather than in
  order (the reference impl reads it in order and logs the trailing values it
  ignored).
* Bounding plan time / `MotionPlanRequest` parameters.

Each of these has a TODO-style comment in `zmq_node.py`.

## Running the integration tests against your sidecar

```bash
MOVEIT2_LIVE_SERVER=1 \
MOVEIT2_SERVER_HOST=127.0.0.1 \
MOVEIT2_SERVER_PORT=5556 \
hatch run test-integ tests_integ/policies/moveit2/ -m moveit2 -v
```

The integ tests live at
[`tests_integ/policies/moveit2/test_moveit2_live.py`](../../../../tests_integ/policies/moveit2/test_moveit2_live.py).
They exercise the full client → sidecar round-trip and expect a
`success=True` response for an in-workspace pose.
