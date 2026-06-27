"""MoveIt2 policy - service-mode client for ROS 2 / MoveIt2 motion planning.

The :class:`MoveIt2Policy` is a thin ZMQ + msgpack client that talks to a
sidecar ROS 2 node running ``moveit_py``. ROS 2 lives entirely out of
process, so users without ROS 2 sourced are unaffected - the only
client-side dependencies are ``pyzmq`` and ``msgpack`` (extra ``[moveit2]``).

Wire protocol (mirrors :class:`~strands_robots.policies.groot.client.Gr00tInferenceClient`)::

    request  = {
        "joint_state": list[float] | None,
        "target_pose": [x, y, z, qw, qx, qy, qz] | None,
        "target_joints": dict[str, float] | None,
        "planning_group": str,
        "world_update": dict | None,
    }
    response = {
        "trajectory": list[list[float]],   # [[t0, q0_0, q0_1, ...], ...]
        "success": bool,
        "status": str,
    }

The reference sidecar implementation lives under
:mod:`strands_robots.policies.moveit2.server` (import-only Python source -
the ROS 2 deps stay out of ``pyproject.toml``).

Subtask 3 of issue #299. The :class:`Policy` ABC contract for non-VLA
providers landed in #300 (well-known ``target_pose`` / ``target_joints`` /
``world_update`` kwargs).
"""

from strands_robots.policies.moveit2.client import MoveIt2InferenceClient, MsgSerializer
from strands_robots.policies.moveit2.policy import MoveIt2Policy

__all__ = [
    "MoveIt2Policy",
    "MoveIt2InferenceClient",
    "MsgSerializer",
]
