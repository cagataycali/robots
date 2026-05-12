"""MuJoCo simulation backend for strands-robots.

CPU-based physics with offscreen rendering. No GPU required.
Supports URDF/MJCF loading, multi-robot scenes, policy execution,
domain randomization, and LeRobotDataset recording.

Usage::

    from strands_robots.simulation.mujoco import MuJoCoSimEngine

    sim = MuJoCoSimEngine(tool_name="mujoco_simulation")
    sim.create_world()
    sim.add_robot("so100", data_config="so100")
    sim.run_policy("so100", policy_provider="mock", instruction="wave")

Or via the top-level alias::

    from strands_robots.simulation import Simulation  # → MuJoCoSimEngine
"""

__all__ = [
    "MuJoCoSimEngine",
    "MuJoCoSimulation",  # backward-compat alias
]


def __getattr__(name: str) -> "type":
    if name in ("MuJoCoSimEngine", "MuJoCoSimulation", "Simulation"):
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine as _Cls

        globals()["MuJoCoSimEngine"] = _Cls
        globals()["MuJoCoSimulation"] = _Cls
        globals()["Simulation"] = _Cls
        return _Cls
    raise AttributeError(f"module 'strands_robots.simulation.mujoco' has no attribute {name!r}")
