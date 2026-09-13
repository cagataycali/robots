"""What the dashboard is entitled to say about an e-stop lockout."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Lockout:
    """The fleet-wide lockout as this dashboard understands it."""

    state: str = "unknown"  # locked | clear | unknown
    #: The instant the sender reported, on the SENDER's clock. For display.
    since: float | None = None
    #: The instant this dashboard learned of the event, on ITS OWN clock. Every
    #: ordering question is decided on this, never on :attr:`since` - see
    #: :func:`_learned_at`.
    arrived: float | None = None
    by: str | None = None
    reason: str = "no e-stop or resume seen since this dashboard started"

    def as_fields(self) -> dict[str, Any]:
        """The lockout as flat log fields, omitting the ones this dashboard was never told."""
        out: dict[str, Any] = {"state": self.state, "reason": self.reason}
        if self.since is not None:
            out["since"] = self.since
        if self.arrived is not None:
            out["arrived"] = self.arrived
        if self.by:
            out["by"] = self.by
        return out


def _learned_at(lockout: Lockout) -> float | None:
    """The lockout instant on THIS dashboard's clock, for ordering local observations.

    ``first_seen`` and ``proof_at`` are stamped by this dashboard, so they may
    only be ordered against a stamp from the same clock. :attr:`Lockout.since`
    is not one: it is whatever the sending peer put in the envelope's ``t``,
    which the mesh admits up to ``STRANDS_MESH_RESUME_FRESHNESS_S`` (60 s by
    default) behind the receiver's clock, before any cross-peer skew.

    Falls back to :attr:`Lockout.since` for a lockout assembled by hand rather
    than folded from an event, where no local stamp exists to prefer.
    """
    return lockout.arrived if lockout.arrived is not None else lockout.since


def _source_of(data: dict[str, Any]) -> str | None:
    for key in ("source", "coordinator", "peer_id", "source_peer_id", "by", "sender"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def apply_event(current: Lockout, *, kind: str, data: dict[str, Any], now: float) -> Lockout:
    """Fold one `strands/safety/**` event into the verdict.

    Args:
        current: The verdict so far.
        kind: The event, ``"estop"`` or ``"resume"``; anything else is ignored.
        data: The envelope body. ``t`` is the SENDER's clock and is recorded for
            display only; it is never used to order this dashboard's own
            observations (see :func:`_learned_at`).
        now: This dashboard's clock, recorded as :attr:`Lockout.arrived`.
    """
    t_val = data.get("t")
    when = t_val if isinstance(t_val, (int, float)) else now
    who = _source_of(data)
    if kind == "estop":
        return Lockout(
            state="locked",
            since=float(when),
            arrived=now,
            by=who,
            reason=(f"an e-stop from {who} locked the fleet" if who else "an e-stop locked the fleet"),
        )
    if kind == "resume":
        # NOT clear: every peer re-verifies the override code on its own and may refuse.
        return Lockout(
            state="unknown",
            since=float(when),
            arrived=now,
            by=who,
            reason=(
                "a resume was broadcast, but each peer verifies the override code itself - "
                "not proof that any of them cleared"
            ),
        )
    return current


def note_command_accepted(current: Lockout, *, now: float) -> Lockout:
    """A peer accepted a command a lockout would have refused: that is proof."""
    if current.state == "clear":
        return current
    return Lockout(
        state="clear",
        since=now,
        arrived=now,
        by=None,
        reason="a command this peer accepted proves its lockout is not engaged",
    )


#: Actions a locked-out peer still answers, so accepting one proves nothing.
LOCKOUT_EXEMPT_ACTIONS = frozenset({"status", "resume"})


def proves_clear(action: str) -> bool:
    """Would a locked-out peer have refused this action?"""
    return bool(action) and action not in LOCKOUT_EXEMPT_ACTIONS


def peer_lockout(fleet: Lockout, *, first_seen: float | None) -> Lockout:
    """The verdict for ONE peer, given when the dashboard first saw it.

    A peer that appeared after the e-stop is a process that never received it.
    "After" is decided against the instant this dashboard *learned* of the stop,
    because that is the same clock ``first_seen`` was stamped by.
    """
    learned_at = _learned_at(fleet)
    if fleet.state == "locked" and first_seen is not None and learned_at is not None:
        if first_seen > learned_at:
            return replace(
                fleet,
                state="unknown",
                reason=(
                    "this peer appeared after the fleet e-stop, so it may never have "
                    "received it - drive it only if you know it is safe"
                ),
            )
    return fleet


def resolve_peer(fleet: Lockout, *, first_seen: float | None = None, proof_at: float | None = None) -> Lockout:
    """The verdict shown on one peer's card.

    Args:
        fleet: The fleet-wide verdict.
        first_seen: When this dashboard first saw the peer, on its own clock.
        proof_at: When this dashboard saw the peer accept an action
            :func:`proves_clear` admits, on its own clock. Only proof from after
            the stop was known counts; earlier proof says nothing about now.
    """
    verdict = peer_lockout(fleet, first_seen=first_seen)
    learned_at = _learned_at(fleet)
    if proof_at is not None and (learned_at is None or proof_at > learned_at):
        return note_command_accepted(verdict, now=proof_at)
    return verdict
