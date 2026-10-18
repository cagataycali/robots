"""Who is commanding: the developer's own code, or an agent/dashboard (whose every write passes the gate).

``with from_agent():`` marks the calls made inside it; the run loop then writes each step through the safety
chain and the operator gate instead of acting directly. A background ``start`` carries the mark to its thread.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_AGENT: ContextVar[bool] = ContextVar("strands_robots_agent_source", default=False)
# The operator already said yes once for this run (the dashboard's ``X-Confirm``): the operator step of the chain is
# pre-satisfied, every other step (e-stop, range, safe, interlocks, validators) still runs on every action.
_CONFIRMED: ContextVar[bool] = ContextVar("strands_robots_operator_confirmed", default=False)
# A policy at a network address (``zenoh://host:port/x``) streams the cameras to that host on every step, before the
# operator gate sees an action: a model never chooses where the frames go. The developer's own ``run("zenoh://...")``
# is untouched.
REMOTE_SPEC = "://"


def refuse_remote_spec(spec: object) -> None:
    """Under ``from_agent()``, a policy spec naming a network address is a ``PermissionError``."""
    if is_agent() and isinstance(spec, str) and REMOTE_SPEC in spec:
        raise PermissionError(f"a policy at a network address is the developer's to pass, not an agent's: {spec}")


@contextmanager
def from_agent(*, confirmed: bool = False) -> Iterator[None]:
    """Mark the calls inside as an agent's or a dashboard's. ``confirmed=True`` carries one operator yes into the
    run: the loop still passes every write through the safety chain, with the operator step pre-satisfied."""
    token, ok = _AGENT.set(True), _CONFIRMED.set(bool(confirmed))
    try:
        yield
    finally:
        _CONFIRMED.reset(ok)
        _AGENT.reset(token)


def is_agent() -> bool:
    return _AGENT.get()


def is_confirmed() -> bool:
    """The operator already approved this run (``from_agent(confirmed=True)``)."""
    return _CONFIRMED.get()


def gated_write(robot: Any) -> Callable[..., Any]:
    """The write an agent's or a dashboard's run uses: ``confirm_act`` with the operator step pre-satisfied when the
    operator already said yes for this run."""
    if is_confirmed():
        return functools.partial(robot.confirm_act, confirmed=True)
    return robot.confirm_act
