"""Refuse a recording whose arms cannot report where they are.

``/api/record/open`` has three camera gates — each one exists because a dataset is the expensive
artifact here: an hour of hand-guiding an arm, discovered to be useless at training time. There was no
gate at all on the JOINTS, which are the dataset's whole point: the follower's positions are the
observations and the leader's are the actions. An arm that publishes no joint positions produces
episodes with nothing to learn from.

Measured on cagatay's rig 2026-08-21, with both real arms in exactly this state (one port-contended,
one uncalibrated): every gate above passed, and the failure arrived from the opening code as
``500 could not open the arms: <exception>`` — a raw traceback string for a fault the dashboard had
already diagnosed and could already explain (:mod:`joint_silence`). This turns that into the same
shape as the camera gates: a 409 the operator can read and act on, with the remedy attached.

Deliberately NOT continuable. The camera refusals offer an override because a missing view is a
degraded dataset; positions that cannot be read are not a degraded dataset, they are an empty one.

The rule is evidence-first, and silent whenever the evidence is not there:

* no snapshot for that peer, or an unreadable one -> None. A peer we cannot see is the ageing gate's
  business, not ours, and refusing on absent evidence would block recording every time the mesh
  bridge hiccups.
* joints present -> None, whatever else is wrong.
* a snapshot older than ``max_age_s`` -> None: "no joints" read from a stale snapshot is not evidence
  about now, and this is the exact trap the fleet-view badge fell into before Q80.
* joints missing in a FRESH snapshot -> refuse, naming the role, the peer, how old the reading is, and
  the classified reason plus remedy when :mod:`joint_silence` has one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: A snapshot older than this says nothing about the arm's present state. Same reasoning as
#: DeviceManager.ROSTER_MAX_AGE_S, a shorter number because joints stream at ~1Hz: if the bridge has
#: not heard from this peer in half a minute, the silence is about the mesh, not about the motors.
MAX_AGE_S = 30.0


def _joint_count(peer: Any) -> int | None:
    """How many joints this peer's snapshot reports, or None when there is nothing to read."""
    if not isinstance(peer, Mapping):
        return None
    state = peer.get("state")
    if not isinstance(state, Mapping):
        return None
    joints = state.get("joints")
    if joints is None:
        return 0
    if isinstance(joints, Mapping):
        return len(joints)
    if isinstance(joints, (list, tuple)):
        return len(joints)
    return None  # a shape we do not understand is not evidence of absence


def _age_s(peer: Mapping, now: float) -> float | None:
    for key in ("last_seen",):
        try:
            seen = float(peer.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if seen > 0:
            return max(0.0, now - seen)
    return None


def refusal(
    *,
    role: str,
    peer_id: str,
    peer: Any,
    problem: Any = None,
    now: float,
    max_age_s: float = MAX_AGE_S,
) -> str | None:
    """Why this arm cannot be recorded from, or None to proceed."""
    count = _joint_count(peer)
    if count is None or count > 0:
        return None
    age = _age_s(peer, now)  # type: ignore[arg-type]
    if age is None or age > max_age_s:
        # Either we cannot date the reading or it is old. Both mean this is not evidence about now.
        return None
    where = f"{role} '{peer_id}'"
    detail = ""
    if isinstance(problem, Mapping):
        headline = str(problem.get("headline") or "").strip()
        remedy = str(problem.get("remedy") or "").strip()
        detail = " ".join(p for p in (f"{headline}." if headline else "", remedy) if p).strip()
    if not detail:
        detail = (
            "Open this robot's log (devices > logs): the joint probe records the exception it raised "
            "there in full."
        )
    return (
        f"{where} is on the mesh but reports NO joint positions (reading is {age:.0f}s old), so a "
        f"recording would capture an arm whose position cannot be read - the episodes would carry no "
        f"{'actions' if role == 'leader' else 'observations'} to learn from. {detail}"
    )
