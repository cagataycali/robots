"""Newton/Warp GPU-accelerated simulation backend.

Provides ``NewtonSimulation(SimEngine)`` for GPU-native physics with 4096+
parallel environments, differentiable simulation, and 7 solver backends.

Heavy dependencies (``warp-lang``, ``newton-sim``) are lazy-imported — this
module is safe to import without triggering GPU initialization.

Usage::

    from strands_robots.simulation import create_simulation

    sim = create_simulation("newton", num_envs=4096, solver="mujoco")
    sim.create_world()
    sim.add_robot("so100")

    # Or direct import
    from strands_robots.simulation.newton import NewtonSimulation, NewtonConfig
"""

from __future__ import annotations

import importlib as _importlib
from typing import Any

# Light re-exports (no heavy deps)
from strands_robots.simulation.newton.config import NewtonConfig
from strands_robots.simulation.newton.solvers import SOLVER_MAP

# Lazy-loaded heavy exports
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "NewtonSimulation": (
        "strands_robots.simulation.newton.simulation",
        "NewtonSimulation",
    ),
}

__all__ = [
    "NewtonConfig",
    "NewtonSimulation",
    "SOLVER_MAP",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
