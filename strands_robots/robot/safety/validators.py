"""Validators: pure checks ``(device, request) -> None`` that raise ``RequestRefused`` naming the bad input.

Validators are advisory command checks for agents and dashboards, not functional safety: a real body
keeps its own firmware limits and an operator keeps the e-stop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..robot import Robot

_EPS = 1e-9


class RequestRefused(PermissionError):
    """A command the device will not carry out; the message names the input, its value and the bound."""


@dataclass(frozen=True)
class Request:
    """A control write on its way through the safety chain: full targets in ``action_keys`` order."""

    keys: tuple[str, ...]
    targets: np.ndarray
    confirmed: bool = False
    source: str = "agent"

    def items(self) -> list[tuple[str, float]]:
        return [(k, float(v)) for k, v in zip(self.keys, self.targets, strict=True)]


Validator = Callable[["Robot", Request], Any]


def _outside(v: float, bound: tuple[float, float] | None) -> bool:
    return bound is not None and not (bound[0] - _EPS <= v <= bound[1] + _EPS)


class JointLimitValidator:
    """Refuse any target outside its control's hard ``range`` (the Description's joint limits)."""

    def __call__(self, robot: Robot, request: Request) -> None:
        manifest = robot.manifest
        for key, v in request.items():
            rng = manifest.control(f"{key}.pos").range
            if _outside(v, rng):
                assert rng is not None
                raise RequestRefused(f"{robot.id}: {key}={v:.4f} rad is outside its range [{rng[0]:.4f}, {rng[1]:.4f}]")


class WorkspaceBoundsValidator:
    """Refuse targets whose tool point ``fk(targets) -> xyz`` (metres, world frame) leaves the box ``lo..hi``."""

    def __init__(self, fk: Callable[[np.ndarray], Any], *, lo: Sequence[float], hi: Sequence[float]) -> None:
        self.fk, self.lo, self.hi = fk, np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    def __call__(self, robot: Robot, request: Request) -> None:
        p = np.asarray(self.fk(request.targets), dtype=float)
        for axis, v, a, b in zip("xyz", p, self.lo, self.hi, strict=True):
            if _outside(float(v), (float(a), float(b))):
                raise RequestRefused(f"{robot.id}: tool {axis}={v:.4f} m is outside the workspace [{a:.4f}, {b:.4f}]")


def outside_safe(robot: Robot, request: Request) -> list[str]:
    """Each target inside ``range`` but outside the soft ``safe`` bound, as a readable phrase."""
    out = []
    for key, v in request.items():
        safe = robot.manifest.control(f"{key}.pos").safe
        if _outside(v, safe):
            assert safe is not None
            out.append(f"{key}={v:.4f} rad outside safe [{safe[0]:.4f}, {safe[1]:.4f}]")
    return out


def compose(*validators: Validator | Sequence[Validator]) -> Validator:
    """One validator that runs the given ones in order and stops at the first refusal."""
    flat: list[Validator] = []
    for v in validators:
        flat.extend(v if isinstance(v, Sequence) else [v])

    def run_all(robot: Robot, request: Request) -> None:
        for v in flat:
            v(robot, request)

    return run_all
