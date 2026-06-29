"""Locomotion intent layer: the :class:`Planner` ABC and its command types.

A :class:`Planner` sits one layer *above* a locomotion :class:`~strands_robots.policies.base.Policy`.
Where a policy turns a goal into joint targets each control tick, a planner
turns *user/agent intent* (steer left, walk faster, switch to a stealth gait)
into a stream of :class:`PlannerCommand` goals. The command's
:meth:`PlannerCommand.to_policy_kwargs` maps onto the established locomotion
goal channel (the ``target_velocity`` ``policy_kwargs`` contract that WBC and
other non-VLA providers already read), so a planner composes with the existing
:meth:`~strands_robots.simulation.base.SimEngine.run_policy` path without a
second, redundant goal API on the policy:

    sim.run_policy(robot_name="unitree_g1", policy_provider="wbc",
                   planner=KinematicPlanner(ScriptedInput([...])),
                   duration=60.0, control_frequency=50.0)

Each control tick the runner samples ``planner.poll().to_policy_kwargs()`` and
merges it into the per-call ``policy_kwargs`` so the locomotion goal varies over
time. The ABC mirrors :class:`~strands_robots.policies.base.Policy`: it carries
the loop's control rate, supports ``reset``/``start``/``stop`` lifecycle, and is
usable as a context manager.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

#: Movement styles a planner can emit. Names match NVIDIA's GR00T
#: Whole-Body-Control SONIC kinematic-planner demos exactly so a style switch is
#: portable to a style-conditioned locomotion policy. A policy that does not
#: understand a style ignores the ``locomotion_style`` kwarg (per the goal-kwarg
#: contract), so emitting a style is always safe.
STYLES: tuple[str, ...] = (
    "run",
    "happy",
    "stealth",
    "injured",
    "kneeling",
    "hand_crawling",
    "elbow_crawling",
    "boxing",
)

#: Default style for a fresh command (a plain forward gait).
DEFAULT_STYLE: str = "run"

#: Nominal standing base height (m) for the Unitree G1; the neutral height a
#: fresh command targets. Style-conditioned policies read ``target_height``.
DEFAULT_HEIGHT: float = 0.74


def _finite(value: float, name: str) -> float:
    """Coerce ``value`` to a finite ``float`` or raise ``ValueError``."""
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return f


@dataclass(frozen=True)
class PlannerCommand:
    """A single locomotion-intent command: where to go, how tall, what style.

    The command is immutable and JSON-serialisable (via :meth:`to_dict` /
    :meth:`from_dict`) so it can travel over the mesh to a remote locomotion
    node unchanged.

    Args:
        root_vel: Desired base velocity ``(vx, vy, omega)`` in the robot frame -
            forward m/s, lateral m/s, yaw rad/s.
        height: Desired base height in metres.
        style: Movement style; must be one of :data:`STYLES`.

    Raises:
        ValueError: If ``root_vel`` is not a length-3 finite triple, ``height``
            is non-finite, or ``style`` is not a known style.
    """

    root_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    height: float = DEFAULT_HEIGHT
    style: str = DEFAULT_STYLE

    def __post_init__(self) -> None:
        vel = tuple(self.root_vel)
        if len(vel) != 3:
            raise ValueError(f"root_vel must be a (vx, vy, omega) triple, got {self.root_vel!r}")
        norm = (
            _finite(vel[0], "root_vel.vx"),
            _finite(vel[1], "root_vel.vy"),
            _finite(vel[2], "root_vel.omega"),
        )
        if self.style not in STYLES:
            raise ValueError(f"unknown style {self.style!r}; expected one of {', '.join(STYLES)}")
        object.__setattr__(self, "root_vel", norm)
        object.__setattr__(self, "height", _finite(self.height, "height"))

    @property
    def vx(self) -> float:
        """Forward velocity (m/s)."""
        return self.root_vel[0]

    @property
    def vy(self) -> float:
        """Lateral velocity (m/s)."""
        return self.root_vel[1]

    @property
    def omega(self) -> float:
        """Yaw rate (rad/s)."""
        return self.root_vel[2]

    def to_policy_kwargs(self) -> dict[str, Any]:
        """Map this command onto the locomotion goal-kwarg contract.

        Returns the well-known goal keys a locomotion policy reads from its
        per-call ``policy_kwargs``: ``target_velocity`` (the raw ``[vx, vy,
        omega]`` triple WBC and friends already consume), plus the forward-
        looking ``target_height`` and ``locomotion_style`` keys a style-
        conditioned policy can use. A policy ignores keys it does not
        understand, so the mapping is safe for every provider.
        """
        return {
            "target_velocity": list(self.root_vel),
            "target_height": self.height,
            "locomotion_style": self.style,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serialisable dict (mesh transport friendly)."""
        return {"root_vel": list(self.root_vel), "height": self.height, "style": self.style}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlannerCommand:
        """Rebuild a command from :meth:`to_dict` output.

        Raises:
            ValueError: If required keys are missing or values are invalid.
        """
        try:
            vel = data["root_vel"]
        except (KeyError, TypeError) as e:
            raise ValueError(f"PlannerCommand.from_dict requires a 'root_vel' key: {data!r}") from e
        return cls(
            root_vel=(float(vel[0]), float(vel[1]), float(vel[2])),
            height=float(data.get("height", DEFAULT_HEIGHT)),
            style=str(data.get("style", DEFAULT_STYLE)),
        )


@dataclass
class PlannerUpdate:
    """A partial intent update emitted by an :class:`~strands_robots.planning.inputs.base.InputSource`.

    Each field is the *absolute* new setting for the dimension it controls, or
    ``None`` to leave that dimension unchanged. An input source only sets the
    fields it owns (a keyboard sets ``root_vel`` from integrated WASD presses; a
    style key sets ``style``), and the planner folds the update into its current
    command, clamping velocity/height to its configured limits.

    Args:
        root_vel: New base velocity ``(vx, vy, omega)`` or ``None``.
        height: New base height (m) or ``None``.
        style: New movement style or ``None``.
        stop: When ``True``, request a halt - the planner zeroes velocity and
            sets :attr:`~strands_robots.planning.kinematic.KinematicPlanner.stop_requested`.
    """

    root_vel: tuple[float, float, float] | None = None
    height: float | None = None
    style: str | None = None
    stop: bool = False

    def is_empty(self) -> bool:
        """Return ``True`` when the update carries no change at all."""
        return self.root_vel is None and self.height is None and self.style is None and not self.stop


class Planner(ABC):
    """Abstract base for a locomotion-intent planner.

    Mirrors :class:`~strands_robots.policies.base.Policy`: the runtime tells the
    planner the control rate via :meth:`set_control_frequency` before a rollout,
    the planner exposes the current goal via :meth:`poll` (non-blocking, called
    once per control tick), and lifecycle is managed with
    :meth:`start`/:meth:`stop` (also usable as a context manager).
    """

    #: Control rate (Hz) of the loop consuming this planner's commands. Set by
    #: the runtime before the rollout; ``None`` until then.
    control_frequency: float | None = None

    def set_control_frequency(self, hz: float) -> None:
        """Tell the planner the control rate (Hz) of the executing loop.

        Args:
            hz: Positive control frequency in Hz.

        Raises:
            ValueError: If ``hz`` is not strictly positive.
        """
        if hz <= 0:
            raise ValueError(f"control_frequency must be positive, got {hz}")
        self.control_frequency = float(hz)

    @abstractmethod
    def poll(self) -> PlannerCommand:
        """Return the current locomotion command. MUST be non-blocking."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the planner to its initial command and clear any stop request."""

    def start(self) -> None:
        """Begin producing commands (e.g. start the input thread). Idempotent."""

    def stop(self) -> None:
        """Stop producing commands and release resources. Idempotent."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable identifier for this planner implementation."""

    def __enter__(self) -> Planner:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
