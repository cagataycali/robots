"""Strands Robots Simulation — multi-backend simulation framework.

Architecture::

    simulation/
    ├── __init__.py          ← this file (re-exports, lazy loading)
    ├── base.py              ← SimEngine ABC (alias: SimulationBackend)
    ├── factory.py           ← create_simulation() + backend registration
    ├── models.py            ← shared dataclasses (SimWorld, SimRobot, ...)
    └── model_registry.py    ← URDF/MJCF resolution (shared across backends)

    # MuJoCo backend added in subsequent PRs.

Usage::

    # Default (MuJoCo) via factory
    from strands_robots.simulation import create_simulation
    sim = create_simulation()

    # Direct class access
    from strands_robots.simulation import Simulation
    sim = Simulation()

    # Explicit backend
    from strands_robots.simulation.mujoco import MuJoCoSimulation

    # Shared types (no heavy deps)
    from strands_robots.simulation import SimWorld, SimRobot, SimObject

    # ABC for custom backends
    from strands_robots.simulation.base import SimEngine, SimulationBackend

Future backends::

    from strands_robots.simulation.isaac import IsaacSimulation
    from strands_robots.simulation.newton import NewtonSimulation
"""

import importlib as _importlib
from typing import Any

# --- Light imports (no heavy deps — stdlib + dataclasses only) ---
from strands_robots.simulation.base import SimEngine, SimulationBackend
from strands_robots.simulation.factory import (
    create_simulation,
    list_backends,
    register_backend,
)
from strands_robots.simulation.model_registry import (
    list_available_models,
    list_registered_urdfs,
    register_urdf,
    resolve_model,
    resolve_urdf,
)
from strands_robots.simulation.models import (
    SimCamera,
    SimObject,
    SimRobot,
    SimStatus,
    SimWorld,
    TrajectoryStep,
)

# --- Heavy imports (lazy — loaded when mujoco backend is available) ---
# MuJoCo-specific lazy imports will be added when the mujoco/ subpackage
# is introduced. For now, only the lightweight foundation is available.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {}


__all__ = [
    # ABC
    "SimEngine",
    "SimulationBackend",  # backward compat alias
    # Factory
    "create_simulation",
    "list_backends",
    "register_backend",
    # Default backend alias (available when mujoco backend is installed)
    # "Simulation",
    # "MuJoCoSimulation",
    # Shared dataclasses
    "SimStatus",
    "SimRobot",
    "SimObject",
    "SimCamera",
    "SimWorld",
    "TrajectoryStep",
    # MuJoCo builder (available when mujoco backend is installed)
    # "MJCFBuilder",
    # Model registry
    "register_urdf",
    "resolve_model",
    "resolve_urdf",
    "list_registered_urdfs",
    "list_available_models",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'strands_robots.simulation' has no attribute {name!r}")


# NOTE: MuJoCo GL backend configuration lives in the top-level
# strands_robots/__init__.py to ensure it runs before any `import mujoco`.
# Do NOT duplicate it here — see PR #86 for the canonical location.
