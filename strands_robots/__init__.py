"""strands-robots v1: a Simulation is a world, a Robot is a body in a world (or on a bus),
a Policy turns an Observation into an Action.

This is the ``dev`` branch: v1 is built here part by part (strands-labs/robots#4749,
schedule #3818). Part 1 is the data that flows between them and the ``Description`` of a robot;
part 2 is the world they live in:

    from strands_robots import Description, Simulation

    d = Description.load("so101")
    d.action_keys   # ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
    sim = Simulation.default()   # MuJoCo, headless, one per process; mujoco is imported here, not before
"""

from __future__ import annotations

from .description import Description
from .simulation import Backend, Object, Scene, Simulation
from .types import Action, Calibration, Contact, JointState, Observation, Physics, Pose, Requirements, RunResult

__version__ = "1.0.0.dev0"

__all__ = [
    "Simulation",
    "Description",
    "Backend",
    "Scene",
    "Object",
    "Observation",
    "Action",
    "Calibration",
    "Contact",
    "JointState",
    "Physics",
    "Pose",
    "RunResult",
    "Requirements",
    "__version__",
]
