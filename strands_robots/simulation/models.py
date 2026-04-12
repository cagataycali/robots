"""Dataclasses for simulation state.

These dataclasses provide a backend-independent typed state representation
consumed by simulation engine implementations (e.g. MuJoCo backend in
``strands_robots.simulation.mujoco``).

They enable:
    - Type-safe state tracking across simulation steps.
    - Serialisation for checkpoints and trajectory recording.
    - A backend-independent interface for agent tools.

They are defined alongside the ``SimEngine`` ABC because its method
signatures reference them (e.g. ``create_world() → SimWorld``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SimStatus(Enum):
    """Simulation execution status."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SimRobot:
    """A robot instance within the simulation."""

    name: str
    urdf_path: str
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # wxyz quat
    data_config: str | None = None
    body_id: int = -1
    joint_ids: list[int] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    actuator_ids: list[int] = field(default_factory=list)
    namespace: str = ""
    policy_running: bool = False
    policy_steps: int = 0
    policy_instruction: str = ""


@dataclass
class SimObject:
    """An object in the simulation scene."""

    name: str
    shape: str  # "box", "sphere", "cylinder", "capsule", "mesh"
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    size: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.05])
    color: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])  # RGBA
    mass: float = 0.1
    mesh_path: str | None = None
    body_id: int = -1
    is_static: bool = False
    _original_position: list[float] = field(default_factory=list)
    _original_color: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._original_position = list(self.position)
        self._original_color = list(self.color)


@dataclass
class SimCamera:
    """A camera in the simulation."""

    name: str
    position: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    target: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    fov: float = 60.0
    width: int = 640
    height: int = 480
    camera_id: int = -1


@dataclass
class TrajectoryStep:
    """A single step in a recorded trajectory."""

    timestamp: float
    sim_time: float
    robot_name: str
    observation: dict[str, Any]
    action: dict[str, Any]
    instruction: str = ""


@dataclass
class SimWorld:
    """Complete simulation world state.

    Engine-specific internals (``_model``, ``_data``) are typed as ``Any``
    so that each backend can store its own native handles without leaking
    implementation details into this base module.
    """

    robots: dict[str, SimRobot] = field(default_factory=dict)
    objects: dict[str, SimObject] = field(default_factory=dict)
    cameras: dict[str, SimCamera] = field(default_factory=dict)
    timestep: float = 0.002  # 500Hz physics
    gravity: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    ground_plane: bool = True
    status: SimStatus = SimStatus.IDLE
    sim_time: float = 0.0
    step_count: int = 0
    # Engine-specific internals (set after world is built by the backend).
    # Each backend stores its own native handles here — e.g. MuJoCo stores
    # MjModel/MjData, Isaac Sim would store its Scene/World objects, etc.
    _xml: str = ""
    _model: Any = None  # Engine-specific model handle (set by simulation backend)
    _data: Any = None  # Engine-specific data handle (set by simulation backend)
    _robot_base_xml: str = ""
    # Trajectory recording
    _recording: bool = False
    _trajectory: list[TrajectoryStep] = field(default_factory=list)
    # LeRobotDataset recorder
    _dataset_recorder: Any = None
    # Temp directory for scene composition
    _tmpdir: Any = None
    # Physics state checkpoints (used by PhysicsMixin.save_state/restore_state)
    _checkpoints: dict[str, Any] = field(default_factory=dict)
