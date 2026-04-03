"""Strands Robots Simulation — multi-backend simulation framework.

Architecture::

    simulation/
    ├── __init__.py          ← this file (re-exports, Simulation alias)
    ├── base.py              ← SimEngine ABC (alias: SimulationBackend)
    ├── factory.py           ← create_simulation() + backend registration
    ├── models.py            ← shared dataclasses (SimWorld, SimRobot, ...)
    ├── model_registry.py    ← URDF/MJCF resolution (shared across backends)
    └── mujoco/              ← MuJoCo CPU backend
        ├── __init__.py
        ├── backend.py       ← lazy mujoco import + GL config
        ├── mjcf_builder.py  ← MJCF XML builder
        ├── physics.py       ← advanced physics (raycasting, jacobians, forces)
        ├── scene_ops.py     ← XML round-trip inject/eject
        ├── rendering.py     ← render RGB/depth, observations
        ├── policy_runner.py ← run_policy, eval_policy, replay
        ├── randomization.py ← domain randomization
        ├── recording.py     ← LeRobotDataset recording
        ├── tool_spec.json   ← AgentTool input schema
        └── simulation.py    ← Simulation (AgentTool orchestrator)

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

# --- Heavy imports (lazy — need strands SDK + mujoco) ---
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Simulation": ("strands_robots.simulation.mujoco.simulation", "Simulation"),
    "MuJoCoSimulation": ("strands_robots.simulation.mujoco.simulation", "Simulation"),
    "MJCFBuilder": ("strands_robots.simulation.mujoco.mjcf_builder", "MJCFBuilder"),
    "_configure_gl_backend": ("strands_robots.simulation.mujoco.backend", "_configure_gl_backend"),
    "_ensure_mujoco": ("strands_robots.simulation.mujoco.backend", "_ensure_mujoco"),
    "_is_headless": ("strands_robots.simulation.mujoco.backend", "_is_headless"),
}


__all__ = [
    # ABC
    "SimEngine",
    "SimulationBackend",  # backward compat alias
    # Factory
    "create_simulation",
    "list_backends",
    "register_backend",
    # Default backend alias
    "Simulation",
    "MuJoCoSimulation",
    # Shared dataclasses
    "SimStatus",
    "SimRobot",
    "SimObject",
    "SimCamera",
    "SimWorld",
    "TrajectoryStep",
    # MuJoCo builder
    "MJCFBuilder",
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
