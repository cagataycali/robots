"""strands-robots v1: a Simulation is a world, a Robot is a body in a world (or on a bus),
a Policy turns an Observation into an Action.

This is the ``dev`` branch: v1 is built here part by part (strands-labs/robots#4749,
schedule #3818). Part 1 is the data that flows between them and the ``Description`` of a robot;
part 2 is the world they live in; part 3 is the robot; part 5 puts it on a bus:

    from strands_robots import Robot

    arm = Robot("so101")                    # joins Simulation.default(); mujoco is imported here, not before
    g1 = Robot("g1", position=[1, 0, 0])    # the SAME world: arm.sim is g1.sim
    arm.act({"shoulder_pan": 0.3})          # one control step for the world
    arm.run(lambda obs: {"elbow_flex": 0.4}, hz=50, steps=60)   # -> RunResult
    real = Robot("so101", port="/dev/ttyACM0", confirm=ask)     # a feetech bus, a twin in the default world
"""

from __future__ import annotations

from .description import Description
from .driver import Driver
from .policy import Policy
from .robot import Body, Robot, SimBody, Task
from .simulation import Backend, Object, Scene, Simulation
from .types import Action, Calibration, Contact, JointState, Observation, Physics, Pose, Requirements, RunResult

__version__ = "1.0.0.dev0"

__all__ = [
    "Robot",
    "Simulation",
    "Policy",
    "Description",
    "Body",
    "Driver",
    "SimBody",
    "Task",
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
