"""Locomotion intent layer: steer a robot by velocity/style, not joint targets.

The planning package is the top of the locomotion control stack. A
:class:`~strands_robots.planning.kinematic.KinematicPlanner`, fed by a keyboard,
gamepad, LLM agent, or scripted sequence, emits a stream of
:class:`~strands_robots.planning.base.PlannerCommand` goals. Passing the planner
to :meth:`~strands_robots.simulation.base.SimEngine.run_policy` (``planner=``)
makes the runner sample ``planner.poll().to_policy_kwargs()`` each control tick
and merge it into the locomotion policy's goal kwargs, so the existing WBC (or
any ``target_velocity``-reading) policy is steered live without a separate goal
API.

    from strands_robots import Robot
    from strands_robots.planning import KinematicPlanner
    from strands_robots.planning.inputs import KeyboardInput

    robot = Robot("unitree_g1")
    robot.run_policy(policy_provider="wbc",
                     planner=KinematicPlanner(KeyboardInput()),
                     duration=60.0, control_frequency=50.0)
"""

from strands_robots.planning.base import (
    DEFAULT_HEIGHT,
    DEFAULT_STYLE,
    STYLES,
    Planner,
    PlannerCommand,
    PlannerUpdate,
)
from strands_robots.planning.kinematic import KinematicPlanner

__all__ = [
    "Planner",
    "PlannerCommand",
    "PlannerUpdate",
    "KinematicPlanner",
    "STYLES",
    "DEFAULT_STYLE",
    "DEFAULT_HEIGHT",
]
