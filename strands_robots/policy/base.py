"""A Policy turns an Observation into an Action. That is the whole contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Action, Observation
from ..units import SCHEMES
from .requirements import Requirements


@runtime_checkable
class PolicyLike(Protocol):
    def act(self, obs: Observation, *, instruction: str | None = None) -> Action: ...

    def reset(self) -> None: ...


class Policy:
    """Base class: override ``act``; ``units`` and ``requirements`` say what the policy speaks and needs."""

    units: str | None = None  # a scheme name in strands_robots.units (e.g. "rad", "lerobot") or None = cannot say

    def __init_subclass__(cls, **kw: object) -> None:
        super().__init_subclass__(**kw)
        declared = cls.__dict__.get("units")
        if declared is not None and declared not in SCHEMES:
            raise ValueError(f"policy {cls.__name__} declares units {declared!r}; known: {', '.join(sorted(SCHEMES))}")

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def requirements(self) -> Requirements:
        """Cameras/state this policy needs; the default needs nothing specific."""
        return Requirements()
