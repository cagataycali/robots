"""What ``run`` and ``lockstep`` share around THE tick: the declared rate and the run events."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .bind import Bound
from .result import RunResult

if TYPE_CHECKING:
    from ..robot import Robot


def emit(robot: Robot, event: str, payload: dict[str, Any]) -> None:
    events = getattr(robot, "events", None)
    if events is not None:
        events.emit(event, {"device_id": robot.id, **payload})


def announce(robot: Robot, result: RunResult) -> None:
    """``run.stopped`` for every finished run, and ``fault`` first when a policy fault ended it."""
    if result.fault is not None and result.stopped_by == "fault":
        emit(robot, "fault", {"reason": result.fault})
    emit(robot, "run.stopped", {"stopped_by": result.stopped_by, "steps": result.steps})


def declared(bounds: Iterable[Bound], hz: float | None) -> float | None:
    """The rate the policies declare (chunks are replayed at it), refusing a disagreement or an ``hz`` against it."""
    rates = {b.hz: b.policy for b in bounds if b.hz is not None}
    if len(rates) > 1:
        said = ", ".join(f"{p!r} {r:g} Hz" for r, p in rates.items())
        raise ValueError(f"these policies declare different rates: {said}; one world steps at one rate")
    if rates and hz is not None and abs(hz - next(iter(rates))) > 1e-9:
        rate, policy = next(iter(rates.items()))
        raise ValueError(f"{policy!r} runs at {rate:g} Hz; hz={hz:g} would replay it at another rate (leave hz out)")
    return next(iter(rates), None)


def rate(bound: Bound, hz: float | None) -> float:
    said = declared([bound], hz)  # an hz against the policy's own rate is refused
    r = hz if hz is not None else (said or 30.0)
    if r <= 0:
        raise ValueError(f"hz must be > 0, got {r}")
    return float(r)
