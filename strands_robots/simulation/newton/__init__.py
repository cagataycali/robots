"""Newton GPU-native simulation backend for strands-robots.

Newton (newton-physics/newton) runs on NVIDIA Warp + MuJoCo-Warp and supports
GPU-batched parallel environments. It ingests the same MJCF assets as the
MuJoCo backend and renders headlessly via a ray-traced tiled camera, so it
requires no display server. Install via the ``[sim-newton]`` extra.

Usage::

    from strands_robots.simulation import create_simulation

    sim = create_simulation("newton", solver="mujoco")
    sim.create_world()
    sim.add_robot("so100")
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import under TYPE_CHECKING only: keeps the runtime export lazy (no
    # newton/warp import cost) while statically defining the name promised by
    # ``__all__`` for type checkers and static analysis.
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

__all__ = ["NewtonSimEngine"]


def __getattr__(name: str) -> type:
    if name == "NewtonSimEngine":
        from strands_robots.simulation.newton.simulation import NewtonSimEngine as _Cls

        globals()["NewtonSimEngine"] = _Cls
        return _Cls
    raise AttributeError(f"module 'strands_robots.simulation.newton' has no attribute {name!r}")
