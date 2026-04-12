"""MuJoCo simulation backend for strands-robots.

CPU-based physics with offscreen rendering. No GPU required.
Supports URDF/MJCF loading, multi-robot scenes, policy execution,
domain randomization, and LeRobotDataset recording.

Usage::

    from strands_robots.simulation.mujoco import MuJoCoSimulation

    sim = MuJoCoSimulation(tool_name="my_sim")
    sim.create_world()
    sim.add_robot("so100", data_config="so100")
    sim.run_policy("so100", policy_provider="mock", instruction="wave")

Or via the top-level alias::

    from strands_robots.simulation import Simulation  # → MuJoCoSimulation
"""

__all__ = [
    "MuJoCoSimulation",
]


def __getattr__(name: str) -> "type":
    if name == "MuJoCoSimulation":
        from strands_robots.simulation.mujoco.simulation import Simulation as _Sim

        globals()["MuJoCoSimulation"] = _Sim
        return _Sim
    raise AttributeError(f"module 'strands_robots.simulation.mujoco' has no attribute {name!r}")
