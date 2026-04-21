"""Newton solver map and backend constants.

No heavy imports — this module loads instantly.
"""

from __future__ import annotations

# Maps user-facing solver name → Newton solver class name.
# Validated during GTC on Jetson AGX Thor (9/14 subtests passed):
#   ✅ mujoco, semi_implicit, xpbd
#   ❌ featherstone (Warp 1.11 ABI)
#   ⚠️ vbd, style3d, implicit_mpm (soft-body/cloth/granular only)
SOLVER_MAP: dict[str, str] = {
    "mujoco": "SolverMuJoCo",
    "featherstone": "SolverFeatherstone",
    "semi_implicit": "SolverSemiImplicit",
    "xpbd": "SolverXPBD",
    "vbd": "SolverVBD",
    "style3d": "SolverStyle3D",
    "implicit_mpm": "SolverImplicitMPM",
}

RENDER_BACKENDS: frozenset[str] = frozenset(
    {
        "opengl",
        "rerun",
        "viser",
        "null",
        "none",
    }
)

BROAD_PHASE_OPTIONS: frozenset[str] = frozenset(
    {
        "sap",
        "bvh",
        "none",
    }
)
