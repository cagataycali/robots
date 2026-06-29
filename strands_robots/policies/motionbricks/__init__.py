"""MotionBricks policy - NVIDIA generative kinematic motion for the Unitree G1.

:class:`MotionBricksPolicy` wraps NVIDIA's MotionBricks generative motion model
(the ``motionbricks/`` subproject of
`GR00T-WholeBodyControl <https://github.com/NVlabs/GR00T-WholeBodyControl>`_).
It synthesises per-frame full-body ``qpos`` for the G1 from a high-level style
(a clip mode such as ``walk`` / ``stealth_walk`` / ``walk_boxing``) plus a
movement/facing command, and like the rest of the non-VLA family:

* ``requires_images = False`` - driven by a style + direction command, never
  camera frames.
* ``get_actions`` reads the goal from the well-known ``**kwargs`` keys
  (``style`` / ``mode``, ``target_velocity``, ``target_heading``).
* The output is the G1's 29 leg+waist+arm joint targets, keyed by
  :data:`MOTIONBRICKS_G1_JOINTS` (the canonical WBC joint ordering), so it
  composes with :class:`~strands_robots.policies.wbc.WBCPolicy` (MotionBricks
  emits motion targets, WBC tracks them) via
  :class:`~strands_robots.policies.composite.CompositePolicy`.

Requires the ``[motionbricks]`` extra and the upstream checkpoints (git-LFS,
NVIDIA Open Model License); no weights are bundled. See ``docs/policies/motionbricks.md``.
"""

from strands_robots.policies.motionbricks.config import MotionBricksConfig
from strands_robots.policies.motionbricks.observation import (
    PLANNER_STYLE_TO_G1_CLIP,
    allowed_pred_num_tokens,
    build_control_signals,
    resolve_mode,
    resolve_planner_style,
)
from strands_robots.policies.motionbricks.policy import (
    MOTIONBRICKS_G1_JOINTS,
    MotionAgent,
    MotionBricksPolicy,
)

__all__ = [
    "MotionBricksPolicy",
    "MotionBricksConfig",
    "MotionAgent",
    "MOTIONBRICKS_G1_JOINTS",
    "resolve_mode",
    "resolve_planner_style",
    "PLANNER_STYLE_TO_G1_CLIP",
    "allowed_pred_num_tokens",
    "build_control_signals",
]
