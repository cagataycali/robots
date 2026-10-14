"""The data that flows between a Simulation, a Robot and a Policy.

Everything here is a frozen dataclass with numpy inside. No dicts of dicts:
an ``Observation`` is the same object whether it came from MuJoCo, Isaac or a
servo bus, and ``to_lerobot()`` is the only place the LeRobot key names live.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _ro(a: Any, dtype: Any = np.float64) -> np.ndarray:
    """A read-only float array (frozen dataclasses deserve frozen arrays)."""
    out = np.array(a, dtype=dtype)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class JointState:
    """Positions/velocities/efforts for a named set of joints (SI: rad, rad/s, N·m)."""

    names: tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray
    efforts: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.names)
        for f in ("positions", "velocities", "efforts"):
            arr = _ro(getattr(self, f))
            if arr.shape != (n,):
                raise ValueError(f"JointState.{f} has shape {arr.shape}, expected ({n},) for {n} joints")
            object.__setattr__(self, f, arr)

    def __getitem__(self, joint: str) -> float:
        try:
            return float(self.positions[self.names.index(joint)])
        except ValueError:
            raise KeyError(f"unknown joint {joint!r}; joints are {list(self.names)}") from None

    def __len__(self) -> int:
        return len(self.names)

    def as_dict(self) -> dict[str, float]:
        return {n: float(p) for n, p in zip(self.names, self.positions, strict=True)}

    @classmethod
    def zeros(cls, names: tuple[str, ...] | list[str]) -> JointState:
        z = np.zeros(len(names))
        return cls(tuple(names), z, z.copy(), z.copy())


Action = np.ndarray | Mapping[str, float] | JointState
"""Joint targets, either a vector in ``Description.action_keys`` order, a ``{joint: target}`` mapping
(missing joints hold their current position) or the ``JointState`` an observation measured (a hold)."""


@dataclass(frozen=True)
class Pose:
    """A rigid pose in the world frame: position (m) and orientation as a unit quaternion (w, x, y, z).

    A floating body also reports its twist: ``linear_velocity`` (world frame, m/s) and ``angular_velocity``
    (body frame, rad/s — what a gyro mounted on the base reads). Fixed bases and drivers leave both ``None``.
    """

    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray | None = None
    angular_velocity: np.ndarray | None = None

    def __post_init__(self) -> None:
        p, q = _ro(self.position), _ro(self.orientation)
        if p.shape != (3,):
            raise ValueError(f"Pose.position has shape {p.shape}, expected (3,)")
        if q.shape != (4,):
            raise ValueError(f"Pose.orientation has shape {q.shape}, expected (4,) as (w, x, y, z)")
        object.__setattr__(self, "position", p)
        object.__setattr__(self, "orientation", q)
        for f in ("linear_velocity", "angular_velocity"):
            v = getattr(self, f)
            if v is not None:
                arr = _ro(v)
                if arr.shape != (3,):
                    raise ValueError(f"Pose.{f} has shape {arr.shape}, expected (3,)")
                object.__setattr__(self, f, arr)

    @classmethod
    def identity(cls, position: Any = (0.0, 0.0, 0.0)) -> Pose:
        return cls(np.asarray(position, dtype=float), np.array([1.0, 0.0, 0.0, 0.0]))


@dataclass(frozen=True)
class Observation:
    """What a body reports at time ``t``: joints, base pose, camera images and per-body extras."""

    t: float
    joints: JointState
    pose: Pose | None = None
    images: Mapping[str, np.ndarray] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, joint: str) -> float:
        return self.joints[joint]

    def with_joints(self, positions: np.ndarray) -> Observation:
        """The same observation with joint positions replaced (used for unit conversion before a policy)."""
        return Observation(
            t=self.t,
            joints=JointState(
                self.joints.names, np.asarray(positions, dtype=float), self.joints.velocities, self.joints.efforts
            ),
            pose=self.pose,
            images=self.images,
            extras=self.extras,
        )

    def to_lerobot(self) -> dict[str, Any]:
        """``observation.state`` (float32 vector) + ``observation.images.<cam>`` (HxWx3 uint8)."""
        out: dict[str, Any] = {"observation.state": self.joints.positions.astype(np.float32)}
        for name, img in self.images.items():
            out[f"observation.images.{name}"] = img
        return out


@dataclass(frozen=True)
class JointCalibration:
    """One joint's bus calibration — field names and meaning match LeRobot's calibration JSON."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclass(frozen=True)
class Calibration:
    """Per-joint calibration for a hardware body. ``to_json``/``from_json`` are LeRobot's file format,
    so a calibration written by ``lerobot-calibrate`` loads here unchanged."""

    joints: Mapping[str, JointCalibration]

    def to_json(self) -> dict[str, dict[str, int]]:
        return {name: dict(vars(jc)) for name, jc in self.joints.items()}

    @classmethod
    def from_json(cls, data: Mapping[str, Mapping[str, Any]]) -> Calibration:
        return cls({name: JointCalibration(**{k: int(v) for k, v in jc.items()}) for name, jc in data.items()})


@dataclass(frozen=True)
class Contact:
    """One contact between two bodies (world frame)."""

    body_a: str
    body_b: str
    position: np.ndarray
    normal: np.ndarray
    force: float


@dataclass(frozen=True)
class Physics:
    """Which engine steps the world, how finely, and how many physics steps make one control step."""

    name: str
    timestep: float
    substeps: int

    @property
    def control_dt(self) -> float:
        return self.timestep * self.substeps


@dataclass(frozen=True)
class RunResult:
    """What ``Robot.run`` measured: how many policy steps ran, over how long, and why it stopped."""

    steps: int
    seconds: float
    hz_target: float
    hz_achieved: float
    stopped_by: str
    recorded: str | None = None
    final: Observation | None = None


@dataclass(frozen=True)
class Requirements:
    """What a policy needs from the body that feeds it: camera names (and sizes) and the state length.

    ``Robot.run`` checks these BEFORE the first inference so a missing camera is a one-line refusal naming it,
    not a shape error inside a model. ``images`` maps camera name -> (width, height) or None when any size is fine;
    ``state_dim`` None means "whatever the robot has".
    """

    images: Mapping[str, tuple[int, int] | None] = field(default_factory=dict)
    state_dim: int | None = None

    def check(self, obs: Observation, *, who: str = "policy") -> None:
        have = {k: (v.shape[1], v.shape[0]) for k, v in obs.images.items()}
        missing = [c for c in self.images if c not in have]
        if missing:
            raise ValueError(
                f"{who} needs camera(s) {missing} but the observation has {sorted(have) or 'none'}: "
                f"add them with sim.camera(name, size=(w, h), attach=robot) or cameras={{name: index}} on hardware"
            )
        wrong = {c: (have[c], want) for c, want in self.images.items() if want is not None and have[c] != tuple(want)}
        if wrong:
            desc = ", ".join(f"{c}: has {h[0]}x{h[1]}, needs {w[0]}x{w[1]}" for c, (h, w) in wrong.items())
            raise ValueError(f"{who} camera size mismatch — {desc}")
        if self.state_dim is not None and len(obs.joints.positions) != self.state_dim:
            raise ValueError(
                f"{who} expects a {self.state_dim}-dim state, the robot reports {len(obs.joints.positions)} joints "
                f"{list(obs.joints.names)}"
            )
