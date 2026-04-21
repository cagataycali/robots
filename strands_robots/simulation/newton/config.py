"""Newton backend configuration.

Dataclass-only module — no heavy dependencies (no warp, no newton).
Safe to import at module level without triggering GPU initialization.
"""

from __future__ import annotations

from dataclasses import dataclass

# Solver name → Newton solver class name.
# Kept here (not solvers.py) so config validation works without importing warp.
SOLVER_MAP: dict[str, str] = {
    "mujoco": "SolverMuJoCo",
    "featherstone": "SolverFeatherstone",
    "semi_implicit": "SolverSemiImplicit",
    "xpbd": "SolverXPBD",
    "vbd": "SolverVBD",
    "style3d": "SolverStyle3D",
    "implicit_mpm": "SolverImplicitMPM",
}

RENDER_BACKENDS: frozenset[str] = frozenset({"opengl", "rerun", "viser", "null", "none"})

BROAD_PHASE_OPTIONS: frozenset[str] = frozenset({"sap", "bvh", "none"})


@dataclass
class NewtonConfig:
    """Configuration for the Newton GPU simulation backend.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments. GPU backends support 4096+
        on a single device.
    device : str
        Warp device string (``"cuda:0"``, ``"cpu"``).
    solver : str
        Solver backend name. Must be a key in ``SOLVER_MAP``.
        ``"mujoco"`` is the default — fastest for rigid-body articulated
        systems. ``"xpbd"`` / ``"semi_implicit"`` are alternatives.
        ``"vbd"`` / ``"style3d"`` are cloth-only. ``"implicit_mpm"`` is
        for granular/fluid simulation.
    physics_dt : float
        Physics timestep in seconds.
    substeps : int
        Number of substeps per ``step()`` call.
    render_backend : str
        Rendering backend (``"opengl"``, ``"rerun"``, ``"viser"``,
        ``"null"``/``"none"``).
    enable_cuda_graph : bool
        Capture the simulation loop in a CUDA graph for minimal
        Python overhead. Requires static graph (no dynamic shapes).
    enable_differentiable : bool
        Enable gradient tracking for differentiable simulation
        via ``wp.Tape``.
    broad_phase : str
        Broad-phase collision algorithm (``"sap"``, ``"bvh"``, ``"none"``).
    soft_contact_margin : float
        Soft-contact margin distance.
    soft_contact_ke : float
        Contact stiffness.
    soft_contact_kd : float
        Contact damping.
    soft_contact_mu : float
        Contact friction coefficient.
    soft_contact_restitution : float
        Contact restitution coefficient.

    Raises
    ------
    ValueError
        If ``solver``, ``render_backend``, or ``broad_phase`` is invalid,
        or if ``physics_dt <= 0`` or ``num_envs < 1``.
    """

    num_envs: int = 1
    device: str = "cuda:0"
    solver: str = "mujoco"
    physics_dt: float = 0.005
    substeps: int = 1
    render_backend: str = "none"
    enable_cuda_graph: bool = False
    enable_differentiable: bool = False
    broad_phase: str = "sap"
    soft_contact_margin: float = 0.5
    soft_contact_ke: float = 10000.0
    soft_contact_kd: float = 10.0
    soft_contact_mu: float = 0.5
    soft_contact_restitution: float = 0.0

    def __post_init__(self) -> None:
        if self.solver not in SOLVER_MAP:
            raise ValueError(f"Unknown solver {self.solver!r}. Valid options: {sorted(SOLVER_MAP.keys())}")
        if self.render_backend not in RENDER_BACKENDS:
            raise ValueError(
                f"Unknown render_backend {self.render_backend!r}. Valid options: {sorted(RENDER_BACKENDS)}"
            )
        if self.broad_phase not in BROAD_PHASE_OPTIONS:
            raise ValueError(f"Unknown broad_phase {self.broad_phase!r}. Valid options: {sorted(BROAD_PHASE_OPTIONS)}")
        if self.physics_dt <= 0:
            raise ValueError(f"physics_dt must be positive, got {self.physics_dt}")
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")
