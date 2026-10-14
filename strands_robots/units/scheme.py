"""The unit map and the scheme registry: a policy declares what it speaks, the robot converts.

The robot speaks radians; a policy speaks whatever its dataset stored. A ``Units`` is the affine map
between the two, built for a given robot by a named *scheme*. Schemes register themselves
(:func:`register`), so adding one is adding a module, not editing an ``if``-chain.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..description import Description

RAD = "rad"
SCHEMES: dict[str, Callable[[Description], Units]] = {}


def register(name: str) -> Callable[[Callable[[Description], Units]], Callable[[Description], Units]]:
    """Register a scheme builder under ``name`` (used as a decorator in ``rad.py`` / ``lerobot.py``)."""

    def deco(fn: Callable[[Description], Units]) -> Callable[[Description], Units]:
        SCHEMES[name] = fn
        return fn

    return deco


@dataclass(frozen=True)
class Units:
    """An affine map rad -> policy units per action key: ``u = lo_u + (q - lo) / (hi - lo) * (hi_u - lo_u)``."""

    name: str
    keys: tuple[str, ...]
    lo: np.ndarray  # rad
    hi: np.ndarray  # rad
    lo_u: np.ndarray  # policy units
    hi_u: np.ndarray  # policy units

    @classmethod
    def for_robot(cls, description: Description, name: str) -> Units:
        """The converter between this description's radians and the named unit system."""
        try:
            builder = SCHEMES[name]
        except KeyError:
            raise ValueError(f"unknown units {name!r}; known: {', '.join(sorted(SCHEMES))}") from None
        return builder(description)

    @property
    def is_identity(self) -> bool:
        return self.name == RAD

    def from_rad(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        frac = (q - self.lo) / (self.hi - self.lo)
        return self.lo_u + frac * (self.hi_u - self.lo_u)

    def to_rad(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        frac = (u - self.lo_u) / (self.hi_u - self.lo_u)
        return self.lo + frac * (self.hi - self.lo)


def declared_units(policy: Any) -> str | None:
    """What a policy says it speaks: ``policy.units`` -> a registered scheme name, or None when it cannot say."""
    u = getattr(policy, "units", None)
    if u is None:
        return None
    if u not in SCHEMES:
        raise ValueError(f"policy declares units {u!r}; known: {', '.join(sorted(SCHEMES))}")
    return str(u)


def resolve_units(policy: Any, requested: str | None, description: Description) -> Units | None:
    """``requested`` = "auto" (use the policy's declaration, radians if it has none), "rad", a scheme name, or None."""
    if requested is None or requested == RAD:
        return None
    if requested == "auto":
        name = declared_units(policy)
        if name is None or name == RAD:
            return None
        return Units.for_robot(description, name)
    return Units.for_robot(description, requested)
