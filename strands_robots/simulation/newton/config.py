"""Configuration dataclass for the Newton simulation backend.

All validation happens at construction time via ``__post_init__`` so that
misconfiguration is caught early — before Warp or Newton are imported.
"""

from __future__ import annotations

from dataclasses import dataclass

from strands_robots.simulation.newton.solvers import (
    BROAD_PHASE_OPTIONS,
    RENDER_BACKENDS,
    SOLVER_MAP,
)


@dataclass
class NewtonConfig:
    """Configuration for the Newton GPU-native simulation backend.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments. Newton supports 4096+ on a
        single GPU for rigid-body workloads.
    device : str
        Warp device string (``"cuda:0"``, ``"cpu"``).
    solver : str
        Solver backend name. Must be a key in :data:`SOLVER_MAP`.
        Default ``"mujoco"`` (fastest for rigid bodies).
    physics_dt : float
        Physics timestep in seconds. Must be positive.
    substeps : int
        Number of physics substeps per ``step()`` call.
    render_backend : str
        Rendering backend (``"opengl"``, ``"rerun"``, ``"viser"``,
        ``"null"``, ``"none"``).
    enable_cuda_graph : bool
        Capture CUDA graphs to minimise Python overhead. Requires
        the simulation structure to be static after ``replicate()``.
    enable_differentiable : bool
        Enable gradient tracking via ``wp.Tape`` for differentiable
        simulation.
    broad_phase : str
        Broad-phase collision detection algorithm.
    soft_contact_margin : float
        Soft-contact margin distance (metres).
    soft_contact_ke : float
        Contact stiffness coefficient.
    soft_contact_kd : float
        Contact damping coefficient.
    soft_contact_mu : float
        Contact friction coefficient.
    soft_contact_restitution : float
        Contact restitution coefficient.

    Raises
    ------
    ValueError
        If any parameter is outside its valid range or not a
        recognised option.
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
            raise ValueError(f"Unknown solver {self.solver!r}. Available: {sorted(SOLVER_MAP.keys())}")
        if self.render_backend not in RENDER_BACKENDS:
            raise ValueError(f"Unknown render_backend {self.render_backend!r}. Available: {sorted(RENDER_BACKENDS)}")
        if self.broad_phase not in BROAD_PHASE_OPTIONS:
            raise ValueError(f"Unknown broad_phase {self.broad_phase!r}. Available: {sorted(BROAD_PHASE_OPTIONS)}")
        if self.physics_dt <= 0:
            raise ValueError(f"physics_dt must be positive, got {self.physics_dt}")
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")
        if self.substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {self.substeps}")
