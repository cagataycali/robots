"""Live-handle refusal shared by the ``reachy_*`` agent verbs.

Every verb in this package takes the live
:class:`~strands_robots.drivers.reachy.ReachyDriver` handle the orchestrator
constructed, so every verb owes the same answer when it is handed something else.
Kept in one module so the four invariants that answer keeps - an envelope and
never an exception, naming the verb, naming ``driver``, naming the type received
- are written once for :mod:`~strands_robots.tools.reachy.reachy_actions` and
:mod:`~strands_robots.tools.reachy.reachy_reads` both.

The travel envelope those verbs' refusals quote is
:mod:`strands_robots.drivers.reachy_envelope`, which lives with the two drivers
that enforce it.
"""

from __future__ import annotations

from typing import Any


def _handle_refusal_envelope(text: str) -> dict[str, Any]:
    """Wrap a refusal sentence in the envelope every ``@tool`` owes a caller."""
    return {"status": "error", "content": [{"text": text}]}


def live_handle_refusal(
    verb: str,
    driver: Any,
    *,
    accessor: str,
    reads: str,
    expected: str,
) -> dict[str, Any] | None:
    """Return the refusal envelope for an unusable live-handle ``driver``, or ``None``.

    The Reachy sibling of :func:`strands_robots.drivers.unitree._common.live_handle_refusal`,
    kept as a second copy because the two packages must not import each other
    (each stays out of the other's SDK-load path). Every ``reachy_*`` verb that
    takes a wired :class:`~strands_robots.drivers.reachy.ReachyDriver` reads it
    through one accessor; the handle is a live Python object an agent cannot
    synthesize, typed :class:`~typing.Any` so no type leaks into the tool schema.

    The judgement keeps four invariants for every verb: the answer is an error
    *envelope* and never an exception, it names the verb, it names ``driver``,
    and it names the type it received. ``accessor`` must be *callable* on the
    handle, not merely present - a namespace built from a cache dump carries the
    name as data and would fail on the call.

    Args:
        verb: The tool name, opening the message so a transcript reader can
            tell which verb refused.
        driver: The handle to judge; a wrong handle is refused, not coerced.
        accessor: The attribute the verb reads the handle through.
        reads: Why an agent cannot supply the handle, completing "an agent
            cannot synthesize it, because ...".
        expected: What the handle failed to expose and what to pass instead,
            completing "of type 'str' does not expose ...".

    Returns:
        ``None`` when ``driver`` exposes a callable ``accessor``, otherwise the
        ``{"status": "error", "content": [...]}`` envelope.
    """
    if driver is None:
        return _handle_refusal_envelope(
            f"{verb}: `driver` is required. Pass the live ReachyDriver "
            "handle the orchestrator constructed - an agent cannot "
            f"synthesize it, because {reads}."
        )
    if not callable(getattr(driver, accessor, None)):
        return _handle_refusal_envelope(
            f"{verb}: `driver` of type {type(driver).__name__!r} does not expose {expected}"
        )
    return None
