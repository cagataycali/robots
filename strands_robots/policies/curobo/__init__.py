"""cuRobo policy - in-process collision-aware motion planning on GPU.

The :class:`CuroboPolicy` is a thin wrapper around
`NVIDIA cuRobo <https://curobo.org/>`_'s ``MotionGen`` planner. Unlike
:class:`~strands_robots.policies.moveit2.MoveIt2Policy` (which talks to a
ROS 2 sidecar over ZMQ), cuRobo runs **in process** as a CUDA library -
there is no network round-trip, but a CUDA-capable GPU + the
``[curobo]`` extra (``nvidia-curobo``) are required.

The shape of the policy is identical to the rest of the non-VLA family:

* ``requires_images = False`` - planners ignore camera frames.
* ``get_actions`` reads the goal from the well-known ``**kwargs`` keys
  (``target_pose`` / ``target_joints`` / ``world_update``).
* The full collision-free trajectory is cached on the first call;
  subsequent calls yield ``action_horizon``-sized chunks so the 50Hz
  execution loop in :class:`~strands_robots.robot.Robot` can stream
  per-step joint targets without re-planning.

Subtask 2 of issue #299. The :class:`Policy` ABC contract for non-VLA
providers landed in #300 (well-known ``target_pose`` / ``target_joints`` /
``world_update`` kwargs).
"""

from strands_robots.policies.curobo.policy import CuroboPolicy

__all__ = [
    "CuroboPolicy",
]
