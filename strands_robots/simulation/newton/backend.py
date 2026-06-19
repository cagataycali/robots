"""Newton physics lazy import and solver registry.

Newton (newton-physics/newton) is a GPU-accelerated, NVIDIA-Warp-based physics
engine. It is an optional dependency installed via the ``[sim-newton]`` extra.
This module centralises the import so the rest of the backend can assume the
modules are present, and exposes the rigid-body solver registry consumed by
:class:`~strands_robots.simulation.newton.simulation.NewtonSimEngine`.
"""

from __future__ import annotations

import logging
from typing import Any

from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)

_newton: Any = None
_warp: Any = None


def ensure_newton() -> tuple[Any, Any]:
    """Import and return ``(newton, warp)``, raising a clear error if missing.

    Returns:
        Tuple of the imported ``newton`` and ``warp`` modules.

    Raises:
        ImportError: With an install hint pointing at the ``[sim-newton]``
            extra when Newton or Warp are not installed.
    """
    global _newton, _warp
    if _newton is not None and _warp is not None:
        return _newton, _warp
    wp = require_optional(
        "warp",
        pip_install="warp-lang",
        extra="sim-newton",
        purpose="the Newton simulation backend",
    )
    nt = require_optional(
        "newton",
        extra="sim-newton",
        purpose="the Newton simulation backend",
    )
    _warp, _newton = wp, nt
    return nt, wp


def solver_registry() -> dict[str, str]:
    """Map friendly solver names to ``newton.solvers`` class names.

    The rigid-body articulation solvers come first; the soft-body / particle
    solvers (``vbd``, ``style3d``, ``mpm``) are included for completeness but
    are not exercised by articulated robots.

    Returns:
        Ordered mapping of solver alias to the ``newton.solvers`` attribute
        name implementing it.
    """
    return {
        "mujoco": "SolverMuJoCo",
        "featherstone": "SolverFeatherstone",
        "xpbd": "SolverXPBD",
        "semi_implicit": "SolverSemiImplicit",
        "vbd": "SolverVBD",
        "style3d": "SolverStyle3D",
        "mpm": "SolverImplicitMPM",
        "kamino": "SolverKamino",
    }


def resolve_solver_class(solver: str) -> Any:
    """Resolve a friendly solver name to its ``newton.solvers`` class.

    Args:
        solver: Friendly solver name (see :func:`solver_registry`).

    Returns:
        The solver class object from ``newton.solvers``.

    Raises:
        ValueError: If ``solver`` is not a known solver name.
    """
    nt, _ = ensure_newton()
    registry = solver_registry()
    key = solver.lower()
    if key not in registry:
        raise ValueError(f"Unknown Newton solver {solver!r}. Available: {sorted(registry)}")
    return getattr(nt.solvers, registry[key])
