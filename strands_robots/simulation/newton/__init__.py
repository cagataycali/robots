"""Newton simulation backend — GPU-native physics via NVIDIA Warp + Newton.

Lazy-loading module: importing this package does NOT trigger ``import warp``
or ``import newton``. Heavy dependencies are loaded only when
``NewtonSimulation`` is instantiated or its methods are called.

Usage::

    from strands_robots.simulation.newton import NewtonSimulation, NewtonConfig

    config = NewtonConfig(num_envs=4096, solver="mujoco", device="cuda:0")
    sim = NewtonSimulation(config=config)

Or via the factory::

    from strands_robots.simulation import create_simulation
    sim = create_simulation("newton", num_envs=4096)
"""

from __future__ import annotations

from typing import Any

# Light import — dataclass, no heavy deps
from strands_robots.simulation.newton.config import NewtonConfig

# Lazy-loaded heavy imports
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "NewtonSimulation": (
        "strands_robots.simulation.newton.simulation",
        "NewtonSimulation",
    ),
}

__all__ = ["NewtonConfig", "NewtonSimulation"]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
