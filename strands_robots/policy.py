"""A Policy turns an Observation into an Action. That is the whole contract.

Anything with ``act(obs, *, instruction=None)`` and ``reset()`` is a policy; a bare
``(obs) -> action`` callable (optionally taking ``instruction=``) is adapted by
``as_policy``. ``Robot.run`` accepts all three.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from .types import Action, Observation, Requirements


@runtime_checkable
class PolicyLike(Protocol):
    def act(self, obs: Observation, *, instruction: str | None = None) -> Action: ...

    def reset(self) -> None: ...


class Policy:
    """Base class: override ``act``; ``units`` and ``requirements`` say what the policy speaks and needs."""

    units: str | None = None  # "rad" | "lerobot_so10x" | None = cannot say (see strands_robots.units)

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def requirements(self) -> Requirements:
        """Cameras/state this policy needs; the default needs nothing specific."""
        return Requirements()


class CallablePolicy(Policy):
    """Adapts ``f(obs) -> action`` (optionally ``f(obs, instruction=...)``)."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        try:
            self._takes_instruction = "instruction" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            self._takes_instruction = False

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        return self.fn(obs, instruction=instruction) if self._takes_instruction else self.fn(obs)

    def __repr__(self) -> str:
        return f"Policy({getattr(self.fn, '__name__', self.fn)!r})"


class SequencePolicy(Policy):
    """Plays a fixed list of actions, then holds the last one (handy for tests and scripted motions)."""

    def __init__(self, actions: Sequence[Action]) -> None:
        if not actions:
            raise ValueError("SequencePolicy needs at least one action")
        self.actions = list(actions)
        self.i = 0

    def reset(self) -> None:
        self.i = 0

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return a


def as_policy(p: Any) -> PolicyLike:
    """Accept a Policy, anything with act/reset, or a bare callable."""
    if isinstance(p, PolicyLike):
        return p
    if callable(p):
        return CallablePolicy(p)
    raise TypeError(f"not a policy: {type(p).__name__} (need .act(obs) or a callable)")
