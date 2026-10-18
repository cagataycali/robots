"""Who is commanding: the developer's own code, or an agent/dashboard (whose every write passes the gate).

``with from_agent():`` marks the calls made inside it; the run loop then writes each step through the safety
chain and the operator gate instead of acting directly. A background ``start`` carries the mark to its thread.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_AGENT: ContextVar[bool] = ContextVar("strands_robots_agent_source", default=False)
# A policy at a network address (``zenoh://host:port/x``) streams the cameras to that host on every step, before the
# operator gate sees an action: a model never chooses where the frames go. The developer's own ``run("zenoh://...")``
# is untouched.
REMOTE_SPEC = "://"


def refuse_remote_spec(spec: object) -> None:
    """Under ``from_agent()``, a policy spec naming a network address is a ``PermissionError``."""
    if is_agent() and isinstance(spec, str) and REMOTE_SPEC in spec:
        raise PermissionError(f"a policy at a network address is the developer's to pass, not an agent's: {spec}")


@contextmanager
def from_agent() -> Iterator[None]:
    token = _AGENT.set(True)
    try:
        yield
    finally:
        _AGENT.reset(token)


def is_agent() -> bool:
    return _AGENT.get()
