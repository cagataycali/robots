"""Backend-agnostic manager framework for declarative sim environments.

This module is the foundation of :mod:`strands_robots.sim_managers`, a
config-driven composition layer for reinforcement-learning environments modelled
on the *manager* pattern used by Isaac Lab / RSL-RL (and Holosoma). It defines
three primitives:

- :class:`EnvState` - the **backend-agnostic contract**. Every term reads the
  physics quantities it needs from an ``EnvState`` and nothing else, so the same
  term works whether the underlying simulator is MuJoCo, MJWarp, Isaac Gym,
  Isaac Sim, or Newton. A backend (or the caller driving a rollout) is
  responsible for populating an ``EnvState`` each control step.
- :class:`Term` - an atomic, composable unit of computation (one observation
  group, one reward component, one termination check, one command source).
- :class:`Manager` - an ordered collection of terms that the manager combines
  into a single result (concatenated observation vector, summed reward, ...).

Terms are registered in a closed :data:`TERM_REGISTRY` keyed by
``(category, func_name)``. Managers are built from a declarative config that
references terms by ``func`` name; an unknown name is rejected rather than
``eval``-ed, so configs are safe to parse from untrusted / LLM-authored YAML
(same safety posture as :mod:`strands_robots.simulation.predicates`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# Closed registry of term classes: TERM_REGISTRY[category][func_name] -> Term subclass.
TERM_REGISTRY: dict[str, dict[str, type[Term]]] = {}

_T = TypeVar("_T", bound="type[Term]")


def register_term(category: str, name: str) -> Callable[[_T], _T]:
    """Register a :class:`Term` subclass under ``(category, name)``.

    Args:
        category: Manager category, e.g. ``"observation"``, ``"reward"``,
            ``"termination"``, ``"command"``.
        name: The ``func`` name configs reference this term by. Must be unique
            within the category.

    Returns:
        Class decorator that records the term in :data:`TERM_REGISTRY`.

    Raises:
        ValueError: If ``name`` is already registered in ``category``.
    """

    def _decorator(cls: _T) -> _T:
        bucket = TERM_REGISTRY.setdefault(category, {})
        if name in bucket:
            raise ValueError(f"term {name!r} already registered in category {category!r}")
        cls.func_name = name
        cls.category = category
        bucket[name] = cls
        return cls

    return _decorator


def get_term_class(category: str, name: str) -> type[Term]:
    """Look up a registered term class, raising a helpful error if absent.

    Args:
        category: Manager category the term lives in.
        name: The registered ``func`` name.

    Returns:
        The :class:`Term` subclass.

    Raises:
        ValueError: If ``category`` or ``name`` is unknown. The message lists
            the available names so a caller (or agent) can self-correct without
            reading the source.
    """
    bucket = TERM_REGISTRY.get(category)
    if bucket is None:
        raise ValueError(f"unknown term category {category!r}; available: {sorted(TERM_REGISTRY)}")
    cls = bucket.get(name)
    if cls is None:
        raise ValueError(f"unknown {category} term {name!r}; available: {sorted(bucket)}")
    return cls


def list_terms(category: str | None = None) -> dict[str, list[str]]:
    """Return registered term names, optionally filtered to one ``category``.

    Args:
        category: If given, return only that category's terms; otherwise all.

    Returns:
        Mapping of category -> sorted list of registered ``func`` names.
    """
    if category is not None:
        return {category: sorted(TERM_REGISTRY.get(category, {}))}
    return {cat: sorted(bucket) for cat, bucket in TERM_REGISTRY.items()}


def _as_float_array(value: Any, name: str) -> FloatArray:
    """Coerce ``value`` to a 1-D float64 array, raising on ragged / bad input."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"EnvState.{name} must be 1-D, got shape {arr.shape}")
    return arr


@dataclass
class EnvState:
    """Backend-agnostic snapshot of one robot at one control step.

    This is the contract every :class:`Term` reads from. A simulator backend (or
    a rollout driver) populates the fields it can; terms that need a field that
    was not populated raise a clear error rather than silently degrading.

    All array fields are 1-D ``float64`` (coerced on construction). Angular and
    linear base velocities are expressed in the robot's **base frame**;
    ``projected_gravity`` is the gravity unit vector rotated into the base frame
    (``[0, 0, -1]`` when perfectly upright).

    Args:
        joint_pos: Per-joint positions, shape ``(n_joints,)``.
        joint_vel: Per-joint velocities, shape ``(n_joints,)``.
        action: Current control action, shape ``(n_actions,)``.
        last_action: Previous control action, shape ``(n_actions,)``.
        base_lin_vel: Base-frame linear velocity ``[vx, vy, vz]``.
        base_ang_vel: Base-frame angular velocity ``[wx, wy, wz]``.
        projected_gravity: Gravity unit vector in the base frame.
        base_height: Height of the base above the ground plane (metres).
        base_quat: Base orientation quaternion ``[w, x, y, z]`` (optional).
        joint_torque: Per-joint applied torque (defaults to zeros).
        joint_acc: Per-joint acceleration (defaults to zeros).
        default_joint_pos: Nominal joint pose used for relative observations and
            soft-limit penalties (defaults to zeros).
        joint_pos_limits: Per-joint ``(lower, upper)`` hard limits, shape
            ``(n_joints, 2)`` (optional).
        joint_vel_limits: Per-joint velocity magnitude limits (optional).
        joint_torque_limits: Per-joint torque magnitude limits (optional).
        soft_joint_pos_limit_factor: Fraction of the hard position range treated
            as the soft limit for penalties (Isaac Lab default ``0.9``).
        feet_contact: Boolean per-foot contact flags (optional).
        feet_air_time: Per-foot time since last contact, seconds (optional).
        commands: Named command vectors, populated by a ``CommandManager``.
        dt: Control timestep in seconds.
        step_count: Number of control steps elapsed this episode.
        max_episode_length: Episode length in control steps (for ``time_out``).
        terminated: ``True`` when the episode ended for a non-timeout reason
            (used by the ``termination_penalty`` reward term).
        extras: Free-form backend-specific data terms may opt into (e.g.
            ``feet_lin_vel`` for the ``feet_slide`` reward).
    """

    joint_pos: FloatArray
    joint_vel: FloatArray
    action: FloatArray
    last_action: FloatArray
    base_lin_vel: FloatArray = field(default_factory=lambda: np.zeros(3))
    base_ang_vel: FloatArray = field(default_factory=lambda: np.zeros(3))
    projected_gravity: FloatArray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    base_height: float = 0.0
    base_quat: FloatArray | None = None
    joint_torque: FloatArray | None = None
    joint_acc: FloatArray | None = None
    default_joint_pos: FloatArray | None = None
    joint_pos_limits: FloatArray | None = None
    joint_vel_limits: FloatArray | None = None
    joint_torque_limits: FloatArray | None = None
    soft_joint_pos_limit_factor: float = 0.9
    feet_contact: npt.NDArray[np.bool_] | None = None
    feet_air_time: FloatArray | None = None
    commands: dict[str, FloatArray] = field(default_factory=dict)
    dt: float = 0.02
    step_count: int = 0
    max_episode_length: int = 1000
    terminated: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joint_pos = _as_float_array(self.joint_pos, "joint_pos")
        self.joint_vel = _as_float_array(self.joint_vel, "joint_vel")
        self.action = _as_float_array(self.action, "action")
        self.last_action = _as_float_array(self.last_action, "last_action")
        self.base_lin_vel = _as_float_array(self.base_lin_vel, "base_lin_vel")
        self.base_ang_vel = _as_float_array(self.base_ang_vel, "base_ang_vel")
        self.projected_gravity = _as_float_array(self.projected_gravity, "projected_gravity")
        n = self.joint_pos.shape[0]
        if self.joint_vel.shape[0] != n:
            raise ValueError(f"joint_vel length {self.joint_vel.shape[0]} != joint_pos length {n}")
        if self.joint_torque is None:
            self.joint_torque = np.zeros(n)
        if self.joint_acc is None:
            self.joint_acc = np.zeros(n)
        if self.default_joint_pos is None:
            self.default_joint_pos = np.zeros(n)

    @property
    def num_joints(self) -> int:
        """Number of joints described by this state."""
        return int(self.joint_pos.shape[0])

    def command(self, name: str) -> FloatArray:
        """Return a named command vector, raising a clear error if absent.

        Args:
            name: Command key, e.g. ``"base_velocity"``.

        Returns:
            The command vector.

        Raises:
            KeyError: If no command named ``name`` has been set. The message
                lists the available command names.
        """
        try:
            return self.commands[name]
        except KeyError:
            raise KeyError(
                f"command {name!r} not set on EnvState; available: {sorted(self.commands)}. "
                "A CommandManager populates commands before reward/observation terms read them."
            ) from None


class Term(ABC):
    """Atomic, composable unit evaluated against an :class:`EnvState`.

    Subclasses implement :meth:`__call__`. Constructor keyword arguments are the
    term's declarative ``params`` and are stored on ``self.params`` (and as
    attributes) so a config can configure a term without bespoke wiring.

    Class attributes ``func_name`` and ``category`` are set by
    :func:`register_term`.
    """

    func_name: str = ""
    category: str = ""

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)

    @abstractmethod
    def __call__(self, state: EnvState) -> Any:
        """Compute this term's value from ``state``."""
        raise NotImplementedError

    def reset(self, state: EnvState | None = None, *, rng: np.random.Generator | None = None) -> None:
        """Reset any internal term state at episode start. No-op by default."""
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.func_name!r}, params={self.params})"


@dataclass
class TermSpec:
    """Declarative description of one term inside a manager config.

    Args:
        name: Instance label (unique within a manager; used as the breakdown /
            observation-slice key). Defaults to ``func`` when omitted.
        func: Registered term ``func`` name to instantiate.
        weight: Reward weight (reward manager only). Sign encodes reward vs
            penalty; ignored by other managers.
        scale: Observation scale applied before clipping (observation manager).
        clip: Optional ``(low, high)`` clip applied after scaling (observation).
        params: Keyword arguments forwarded to the term constructor.
    """

    func: str
    name: str = ""
    weight: float = 1.0
    scale: float = 1.0
    clip: tuple[float, float] | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.func

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TermSpec:
        """Build a :class:`TermSpec` from a config dict, validating keys.

        Args:
            data: A single term entry from a manager config.

        Returns:
            The parsed spec.

        Raises:
            ValueError: If ``func`` is missing or an unknown key is present.
        """
        allowed = {"name", "func", "weight", "scale", "clip", "params"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown term config keys {sorted(unknown)}; allowed: {sorted(allowed)}")
        if "func" not in data:
            raise ValueError(f"term config missing required 'func' key: {data}")
        clip = data.get("clip")
        clip_tuple = (float(clip[0]), float(clip[1])) if clip is not None else None
        return cls(
            func=str(data["func"]),
            name=str(data.get("name", "")),
            weight=float(data.get("weight", 1.0)),
            scale=float(data.get("scale", 1.0)),
            clip=clip_tuple,
            params=dict(data.get("params", {})),
        )


class Manager(ABC):
    """Ordered collection of terms combined into one result.

    Args:
        terms: ``(label, term)`` pairs in evaluation order. Labels must be
            unique within the manager.

    Raises:
        ValueError: On duplicate labels.
    """

    category: str = ""

    def __init__(self, terms: list[tuple[str, Term]]) -> None:
        labels = [label for label, _ in terms]
        dupes = {label for label in labels if labels.count(label) > 1}
        if dupes:
            raise ValueError(f"duplicate term labels in {type(self).__name__}: {sorted(dupes)}")
        self._terms = terms

    @property
    def term_names(self) -> list[str]:
        """Ordered list of term labels."""
        return [label for label, _ in self._terms]

    def __len__(self) -> int:
        return len(self._terms)

    def __iter__(self) -> Iterator[tuple[str, Term]]:
        return iter(self._terms)

    @abstractmethod
    def compute(self, state: EnvState) -> Any:
        """Combine all terms against ``state`` into the manager's result."""
        raise NotImplementedError

    def reset(self, state: EnvState | None = None, *, rng: np.random.Generator | None = None) -> None:
        """Reset every term at episode start."""
        for _, term in self._terms:
            term.reset(state, rng=rng)
