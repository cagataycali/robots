"""Configuration for the Newton simulation backend.

Validates all user-supplied configuration at construction time so that
errors surface during setup rather than during inference.
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
    """Configuration for the Newton simulation backend.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments.  Set to 4096+ for GPU training.
    device : str
        Warp device string (``"cuda:0"``, ``"cpu"``).
    solver : str
        Physics solver.  See :data:`SOLVER_MAP` for options.
    physics_dt : float
        Physics timestep in seconds.
    substeps : int
        Physics substeps per ``step()`` call.
    render_backend : str
        Rendering backend (``"opengl"``, ``"rerun"``, ``"viser"``, ``"null"``).
    enable_cuda_graph : bool
        Capture CUDA graph on first ``step()`` for minimal Python overhead.
    enable_differentiable : bool
        Enable gradient tracking for differentiable simulation.
    broad_phase : str
        Broad-phase collision detection algorithm.
    soft_contact_margin : float
        Soft-contact margin distance (metres).
    soft_contact_ke : float
        Contact elastic stiffness.
    soft_contact_kd : float
        Contact damping coefficient.
    soft_contact_mu : float
        Friction coefficient.
    soft_contact_restitution : float
        Coefficient of restitution (bounciness).
    """

    num_envs: int = 1
    device: str = "cuda:0"
    solver: str = "mujoco"
    physics_dt: float = 0.005
    substeps: int = 1
    render_backend: str = "null"
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
            raise ValueError(f"Unknown solver {self.solver!r}. Available: {sorted(SOLVER_MAP)}")
        if self.render_backend not in RENDER_BACKENDS:
            raise ValueError(f"Unknown render_backend {self.render_backend!r}. Available: {sorted(RENDER_BACKENDS)}")
        if self.broad_phase not in BROAD_PHASE_OPTIONS:
            raise ValueError(f"Unknown broad_phase {self.broad_phase!r}. Available: {sorted(BROAD_PHASE_OPTIONS)}")
        if self.physics_dt <= 0:
            raise ValueError(f"physics_dt must be positive, got {self.physics_dt}")
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")
