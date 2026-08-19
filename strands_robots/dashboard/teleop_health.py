"""Is teleop actually working? — the verdict, from the counters.

Measured on real hardware (2026-08-19): a real SO-101 leader published 176 frames
to a follower that accepted NONE of them. Everything the dashboard could see said
success: /teleop/receive returned "Teleop receive started", the receiver reported
``running: true``, and /api/fleet showed both peers healthy. The only place the
truth existed was the FOLLOWER's child log:

    input frame value for 'shoulder_lift.pos' out of range:
    |-46.417582417582416| > 12.566370614359172

The mesh's per-frame safety envelope is 4*pi -- a RADIAN assumption. An SO-101
reports degrees (wrist_roll sits at 170) and a gripper in percent, so every real
frame from a real arm is out of range. The SDK anticipated this and provides a
knob (STRANDS_MESH_INPUT_VALUE_ABS), which is precisely why the failure must be
diagnosable: the operator is one env var away from working teleop and has no way
to learn it.

So this module turns the raw counters into a sentence, and where the cause is a
widenable envelope it produces a CONSENT request rather than widening a safety
bound behind the operator's back (see dashboard/consent.py).
"""

from __future__ import annotations

import re
from typing import Any, Mapping

__all__ = ["diagnose_receiver", "envelope_refusal", "published_frames", "teleop_health"]

#: The SDK's own refusal, as it appears in the follower's log.
_RANGE_RE = re.compile(
    r"input frame value for '([^']{1,120})' out of range: \|(-?[0-9.eE+]{1,40})\| > ([0-9.eE+]{1,40})"
)
_SLEW_RE = re.compile(r"input frame slew for '([^']{1,120})' out of range: ([0-9.]{1,20}) > ([0-9.]{1,20})")


def envelope_refusal(log_tail: Any) -> dict[str, Any] | None:
    """The most recent value/slew refusal in a child's log, parsed.

    Returns ``{"joint", "value", "bound", "kind"}`` or ``None``. The newest line
    wins: an operator who has already widened the envelope once should see the
    bound they are hitting NOW, not the first one they ever hit.
    """
    if not isinstance(log_tail, (list, tuple)):
        return None
    for line in reversed([str(x) for x in log_tail]):
        m = _RANGE_RE.search(line)
        if m:
            try:
                return {"kind": "value", "joint": m.group(1), "value": abs(float(m.group(2))),
                        "bound": float(m.group(3))}
            except ValueError:
                continue
        m = _SLEW_RE.search(line)
        if m:
            try:
                return {"kind": "slew", "joint": m.group(1), "value": float(m.group(2)),
                        "bound": float(m.group(3))}
            except ValueError:
                continue
    return None


def diagnose_receiver(
    stats: Mapping[str, Any],
    log_tail: Any = None,
    source_frames: int | None = None,
) -> dict[str, Any]:
    """One receiver's counters -> a state and a sentence a human can act on.

    States: ``following`` (frames are landing), ``refusing`` (frames arrive and
    every one is thrown away -- the case that used to look like success),
    ``silent`` (subscribed, nothing arriving), ``unrouted`` (nothing arriving
    even though the leader IS publishing), ``stopped``.

    Args:
        stats: One receiver's counters.
        log_tail: The follower's recent log lines, where the SDK writes the
            reason a frame was refused.
        source_frames: How many frames the NAMED LEADER says it has published,
            when the caller could ask it. Without this, "nothing is arriving"
            can only be blamed on the leader - and blaming the leader while it
            publishes 200 frames a minute sends the operator to the wrong end
            of the problem (measured, 2026-08-19: exactly this happened).
    """
    running = bool(stats.get("running"))
    got = int(stats.get("frames_received") or 0)
    rejected = int(stats.get("rejected") or 0)
    slew = int(stats.get("slew_rejected") or 0)
    dropped = int(stats.get("rate_dropped") or 0)
    source = stats.get("source")

    if not running:
        return {"state": "stopped", "headline": "not following anything", "detail": None}

    if got == 0 and (rejected or slew):
        refusal = envelope_refusal(log_tail)
        detail = (
            f"{rejected + slew} frames arrived from {source} and every one was refused, "
            "so the follower has not moved."
        )
        out: dict[str, Any] = {"state": "refusing", "headline": "every frame is being refused",
                               "detail": detail, "refusal": refusal}
        if refusal:
            out["detail"] = (
                f"{detail} The mesh's teleop envelope is {refusal['bound']:g} units, and this leader "
                f"reported {refusal['joint']} at {refusal['value']:g} — the arm reports DEGREES while "
                f"the envelope assumes radians (4·pi)."
            )
        return out

    if got == 0:
        if source_frames:
            # Both ends are healthy on their own terms and the frames still do
            # not meet. Naming that honestly is the whole point: the operator
            # must not go looking at the leader, which is working.
            return {
                "state": "unrouted",
                "headline": f"{source} is publishing, but nothing reaches this follower",
                "detail": (
                    f"the leader has published {source_frames} frames and this follower has "
                    "received none — they are not meeting on the mesh (a subscription bound to a "
                    "session that has since died does this). Restart the follower, then subscribe "
                    "again."
                ),
                "source_frames": source_frames,
            }
        return {
            "state": "silent",
            "headline": f"subscribed to {source}, but no frames are arriving",
            "detail": "the leader is not publishing — start its stream first "
                      "(POST /api/robots/{leader}/teleop/publish)"
                      + ("" if source_frames is None else " (its publisher reports 0 frames)"),
        }

    hz = float(stats.get("hz_actual") or 0.0)
    note = f"following {source} at {hz:.1f}Hz"
    extra = []
    if rejected or slew:
        extra.append(f"{rejected + slew} frames refused")
    if dropped:
        extra.append(f"{dropped} dropped to the rate cap")
    return {"state": "following", "headline": note,
            "detail": ", ".join(extra) or None}


def published_frames(status: Any, device_name: str) -> int | None:
    """How many frames a peer's ``device_name`` publisher has sent, if it says.

    ``None`` means "could not tell" - which must stay distinct from 0, because 0
    is evidence about the leader and None is the absence of evidence.
    """
    payload = _status_payload(status)
    if payload is None:
        return None
    pubs = payload.get("publishers")
    if not isinstance(pubs, Mapping):
        return None
    stats = pubs.get(device_name)
    if not isinstance(stats, Mapping):
        return None
    try:
        return int(stats.get("frames") or 0)
    except (TypeError, ValueError):
        return None


def _status_payload(status: Any) -> Mapping[str, Any] | None:
    """The counters block inside a tool-envelope status, or the block itself."""
    if not isinstance(status, Mapping):
        return None
    if "receivers" in status or "publishers" in status:
        return status
    for block in status.get("content") or []:
        if isinstance(block, Mapping) and isinstance(block.get("json"), Mapping):
            return block["json"]
    return None


def teleop_health(
    status: Any,
    log_tail: Any = None,
    source_frames: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Verdict for a whole peer's teleop status payload.

    ``status`` is the robot's ``get_teleop_status()`` result as it crosses the
    mesh (a tool envelope whose ``content`` carries a ``json`` block). Anything
    unrecognised yields an empty verdict rather than an exception: this decorates
    a status endpoint and must never be the reason it fails.
    """
    payload = _status_payload(status)
    if payload is None:
        return {"receivers": {}, "publishers": {}, "worst": None}

    receivers = {
        key: diagnose_receiver(stats, log_tail, (source_frames or {}).get(key))
        for key, stats in (payload.get("receivers") or {}).items()
        if isinstance(stats, Mapping)
    }
    publishers = {}
    for name, stats in (payload.get("publishers") or {}).items():
        if not isinstance(stats, Mapping):
            continue
        frames = int(stats.get("frames") or 0)
        hz = float(stats.get("hz_actual") or 0.0)
        target = float(stats.get("hz_target") or 0.0)
        state = "publishing" if stats.get("running") and frames else (
            "starting" if stats.get("running") else "stopped")
        detail = None
        # A rate far under target is not a fault: on a shared bus the state probe
        # and the camera publisher get their turns too. Say so, so nobody reads
        # it as a bug and "fixes" it by asking for more.
        if state == "publishing" and target and hz < target * 0.6:
            detail = (f"{hz:.1f}Hz of the {target:g}Hz requested — the servo bus is shared with "
                      "this arm's state and camera reads")
        publishers[name] = {"state": state, "headline": f"{frames} frames at {hz:.1f}Hz",
                            "detail": detail}

    # The one line worth putting on a card: a refusal outranks everything.
    order = {"refusing": 0, "unrouted": 1, "silent": 2, "stopped": 3, "following": 4}
    worst = None
    for key, verdict in sorted(receivers.items(), key=lambda kv: order.get(kv[1]["state"], 9)):
        worst = {"peer_key": key, **verdict}
        break
    return {"receivers": receivers, "publishers": publishers, "worst": worst}
