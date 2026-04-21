"""Newton GPU-native simulation backend for strands-robots.

Newton is built on NVIDIA Warp and provides GPU-native physics simulation
with 4096+ parallel environments on a single GPU.

Heavy imports (warp, newton) are deferred until first use — importing this
module only loads lightweight config and constants.

Usage::

    from strands_robots.simulation import create_simulation

    sim = create_simulation("newton", num_envs=4096, solver="mujoco")
    sim.create_world()
    sim.add_robot("so100")
    sim.step(100)
    sim.destroy()
"""

from __future__ import annotations

import importlib as _importlib
from typing import Any

# Light imports — no heavy deps
from strands_robots.simulation.newton.config import NewtonConfig
from strands_robots.simulation.newton.solvers import (
    BROAD_PHASE_OPTIONS,
    RENDER_BACKENDS,
    SOLVER_MAP,
)

# Lazy-loaded heavy import
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
    "RENDER_BACKENDS",
    "BROAD_PHASE_OPTIONS",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
