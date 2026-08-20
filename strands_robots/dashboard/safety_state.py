"""What the dashboard is entitled to say about an e-stop lockout.

Q43, measured 2026-08-20: both SO-101 arms had been locked out for ten hours and the
dashboard rendered `so101-arm-2` as a healthy green card with six live joints. There was
no lockout field anywhere in `/api/fleet`; the only representation of a lockout in the
whole product was a five-second flash in the header, so a page reload erased it.

The hard part is not remembering the event - it is not overclaiming. Three facts
constrain every verdict here:

1. **The mesh deliberately does not advertise lockout state.** `Mesh._dispatch` rejects a
   command during lockout with a generic error precisely so a remote caller cannot map
   the lockout window, and `status` returns the robot's task status with no safety flag.
   So the dashboard cannot ask a peer whether it is locked; it can only remember what it
   saw, and admit when it saw nothing.
2. **A resume is a broadcast, not a receipt.** Each peer verifies the override code
   independently and may refuse. Painting the fleet green because a resume was published
   would be the exact failure this dashboard keeps fixing elsewhere: a claim about
   hardware that nobody checked. What DOES prove a peer cleared is the peer itself
   accepting a command that a lockout would have refused - evidence, not a handshake
   (the Q40 lesson, in a second setting).
3. **A peer that appeared after the e-stop is a different process.** A freshly spawned
   child starts with its own unset lockout flag, so inheriting the fleet's verdict would
   mark it red on no evidence. It is `unknown`, and the reason says why.

Hence three states and never a fourth: `locked` (seen, and loud), `clear` (proved by an
accepted command), `unknown` (say so). Absent is not a state - a missing annotation would
read as "fine", which is the bug.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Lockout:
    """The fleet-wide lockout as this dashboard understands it."""

    state: str = "unknown"  # locked | clear | unknown
    since: float | None = None
    by: str | None = None
    reason: str = "no e-stop or resume seen since this dashboard started"

    def as_fields(self) -> dict[str, Any]:
        out: dict[str, Any] = {"state": self.state, "reason": self.reason}
        if self.since is not None:
            out["since"] = self.since
        if self.by:
            out["by"] = self.by
        return out


def _source_of(data: dict[str, Any]) -> str | None:
    for key in ("source", "coordinator", "peer_id", "source_peer_id", "by", "sender"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def apply_event(current: Lockout, *, kind: str, data: dict[str, Any], now: float) -> Lockout:
    """Fold one `strands/safety/**` event into the verdict."""
    when = data.get("t") if isinstance(data.get("t"), (int, float)) else now
    who = _source_of(data)
    if kind == "estop":
        return Lockout(
            state="locked",
            since=float(when),
            by=who,
            reason=(
                f"an e-stop from {who} locked the fleet" if who else "an e-stop locked the fleet"
            ),
        )
    if kind == "resume":
        # NOT clear: every peer re-verifies the override code on its own and may refuse.
        return Lockout(
            state="unknown",
            since=float(when),
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
    """
    if fleet.state == "locked" and first_seen is not None and fleet.since is not None:
        if first_seen > fleet.since:
            return replace(
                fleet,
                state="unknown",
                reason=(
                    "this peer appeared after the fleet e-stop, so it may never have "
                    "received it - drive it only if you know it is safe"
                ),
            )
    return fleet


def resolve_peer(
    fleet: Lockout, *, first_seen: float | None = None, proof_at: float | None = None
) -> Lockout:
    """The verdict shown on one peer's card.

    Proof outranks memory, but only proof that is NEWER than the event it contradicts:
    a command accepted an hour before the e-stop says nothing about now.
    """
    verdict = peer_lockout(fleet, first_seen=first_seen)
    if proof_at is not None and (fleet.since is None or proof_at > fleet.since):
        return note_command_accepted(verdict, now=proof_at)
    return verdict
