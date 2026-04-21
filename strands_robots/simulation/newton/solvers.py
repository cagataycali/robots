"""Solver map and constants for the Newton simulation backend.

Centralises solver names, rendering backends, and broad-phase options
so they can be validated at config time without importing Warp or Newton.
"""

from __future__ import annotations

# Maps user-facing solver name → Newton class name.
# All 7 Newton solvers are listed. Only a subset may be available
# depending on the Warp/Newton version and GPU architecture.
SOLVER_MAP: dict[str, str] = {
    "mujoco": "SolverMuJoCo",  # Default — fastest for rigid bodies
    "featherstone": "SolverFeatherstone",  # ABI-based — Warp ≥1.12 required
    "semi_implicit": "SolverSemiImplicit",  # Stable for stiff systems
    "xpbd": "SolverXPBD",  # Position-based dynamics
    "vbd": "SolverVBD",  # Soft-body only (no revolute joints)
    "style3d": "SolverStyle3D",  # Cloth only
    "implicit_mpm": "SolverImplicitMPM",  # Fluid/granular materials only
}

# Solvers suitable for rigid-body articulated robots.
# vbd, style3d, and implicit_mpm are for soft bodies/cloth/fluids only.
RIGID_BODY_SOLVERS: frozenset[str] = frozenset(
    {
        "mujoco",
        "featherstone",
        "semi_implicit",
        "xpbd",
    }
)

# Rendering backends supported by Newton.
RENDER_BACKENDS: frozenset[str] = frozenset(
    {
        "opengl",
        "rerun",
        "viser",
        "null",
        "none",
    }
)

# Broad-phase collision detection algorithms.
BROAD_PHASE_OPTIONS: frozenset[str] = frozenset(
    {
        "sap",  # Sweep-and-prune (default)
        "bvh",  # Bounding volume hierarchy
        "none",  # No broad-phase (brute-force)
    }
)
