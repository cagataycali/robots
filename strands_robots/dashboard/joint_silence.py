"""Why a CONNECTED arm publishes no joints (Q80).

The fleet view can show an arm with ``connected: true``, a fresh heartbeat, cameras listed and
**not one joint position** -- and say nothing at all about why. That is what happened on cagatay's
rig on 2026-08-20: both arms logged ``hardware connected`` and then omitted the whole joints
section of every snapshot for hours. The reason existed, precisely and in words, in each child's
log ring buffer, where the fleet view never looks:

* ``ConnectionError("Failed to sync read 'Present_Position' ... [TxRxResult] Port is in use!")``
  -- another process holds the serial port. On that day 179 ORPHANED robot children (ppid=1, from
  earlier spawn generations) still held both arm ports, so the live child could not read a byte.
  The remedy is to find the other owner, NOT to replug or recalibrate.
* ``RuntimeError(FeetechMotorsBus(...) has no calibration registered.)`` -- nothing is contended;
  the board simply has no calibration, so positions cannot be expressed in degrees. The remedy is
  to calibrate that arm, and no amount of restarting will help.

Those two look identical from outside (an arm with no joints) and have opposite remedies, which is
exactly the conflation :mod:`cameras` exists to kill on the camera side (U14/Q44). This module is
the joints half.

Everything here is pure: it reads log LINES that were already captured and a snapshot dict that
was already received. It opens no port, spawns nothing, and never decides to act -- a diagnosis
whose own gathering can disturb the bus would be part of the problem it describes.

The gate in :func:`merge` matters as much as the classification. ``mesh.core`` logs a degraded
probe ONCE per category (a warning every tick at STATE_HZ would be unreadable), and it never logs
a RECOVERY -- so a log line stays in the buffer long after the fault is gone. A badge driven by the
log alone would therefore become permanent and start lying. Live joints in the snapshot win over
any past complaint.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

#: The line mesh.core writes when a probe degrades (core.py ``_warn_read_state_once``).
_PROBE_LINE = re.compile(r"state probe '?\"?hw_joints'?\"?.*?(failed|still failing)", re.I)

#: The recovery line mesh/core emits when the probe works again. Scanning newest-first, this ENDS the
#: search: a fault that later recovered is not a fault, and reporting it anyway is how a badge that
#: cannot clear itself teaches the operator to ignore badges. Before this line existed there was no
#: way to know, which is exactly what the old wording admitted ("records a failure once and never a
#: recovery"); a child running older code still emits none, so absence of a recovery proves nothing.
_RECOVERED_LINE = re.compile(r"state probe '?\"?hw_joints'?\"?.*?recovered", re.I)

#: The cure's own fingerprint (bus_access logs it when it clears a stranded in-use flag). Its presence
#: proves this child runs code that heals itself, which changes the advice completely: the flag is no
#: longer the suspect, so an arm still silent AFTER this line has a fault the recovery could not fix.
#: Absence proves nothing about the build - the flag may simply never have stranded - so it only ever
#: sharpens the verdict, never weakens it.
_FLAG_CLEARED_LINE = re.compile(r"marked in-use by an exchange that never finished", re.I)

#: Said when that fingerprint IS present: the stale-flag explanation is spent, and the remaining
#: causes are a real owner or a bus that stopped answering.
_PORT_IN_USE_AFTER_SELF_HEAL = (
    "This arm has already un-stranded its own bus at least once this session, so a stale flag is no "
    "longer the explanation - the recovery ran and the port was busy again immediately. That means a "
    "REAL second owner or a bus that has stopped answering: `/usr/sbin/lsof /dev/cu.usbmodem*` names "
    "every holder (two means an orphaned child from an earlier spawn - stop that one), and if this "
    "process is the only holder, check the cable and the hub. A bus that keeps stranding is failing "
    "hardware, not bad luck - the card's \u201cbus healed\u201d count is the evidence. "
    "Respawning masks it for a while."
)

#: Ordered: the FIRST match wins, so the specific readings are tried before the generic one.
_SIGNATURES: tuple[tuple[str, str, str, str], ...] = (
    (
        "not_probed",
        "joint probe did not run",
        "this peer never attempted a joint read",
        "Nothing failed on the bus, so there is nothing in the log to find: the arm was built "
        "without a readable hardware object, or it reports itself disconnected. Respawn it from "
        "devices (check the port and that it is not a sim peer) - recalibrating and replugging "
        "change nothing here.",
    ),
    (
        "port_in_use",
        "Port is in use!",
        "this arm's serial port is held - by another process, or by its own aborted read",
        "Ask the OS who holds it first: `/usr/sbin/lsof /dev/cu.usbmodem*` names every holder. "
        "TWO holders means an orphaned child from an earlier spawn - stop that one. ONE holder, "
        "this arm's own process, means the port is busy INSIDE it: a read that died mid-exchange "
        "left the motor bus marked in-use. Current code CLEARS that by itself on the next read "
        "(Q81), so an arm that stays silent is either running an older build - respawn it from "
        "devices to pick up the self-healing - or its recovery was refused, which points at a real "
        "second owner rather than a stale flag. Replugging and recalibrating change nothing either "
        "way.",
    ),
    (
        "uncalibrated",
        "no calibration registered",
        "this board has no calibration, so its positions cannot be read in degrees",
        "Calibrate this arm (devices > calibrate). Nothing is contended and a restart will not "
        "help.",
    ),
    (
        "no_response",
        "after 3 tries",
        "the motors did not answer a position read",
        "Check the arm's power supply and its data cable - a board on USB logic power alone "
        "answers some reads and not others.",
    ),
)


def classify(log_lines: Any, calibrations: Any = None) -> dict[str, str] | None:
    """Explain the NEWEST degraded ``hw_joints`` probe in ``log_lines``, or return None.

    Silence is the common case and must stay silent: a child whose log never mentions the probe
    contributes nothing, so a healthy arm carries no field at all rather than an empty verdict.
    """
    if not isinstance(log_lines, (list, tuple)):
        return None
    for line in reversed([str(x) for x in log_lines]):
        if _RECOVERED_LINE.search(line):
            # Newest-first, so a recovery seen before any failure means the newest word on this probe
            # is "it works" -- stop, and report nothing.
            return None
        if not _PROBE_LINE.search(line):
            continue
        matched = _match(line)
        if matched is not None:
            if matched["kind"] == "port_in_use" and _self_healed(log_lines):
                matched = {**matched, "remedy": _PORT_IN_USE_AFTER_SELF_HEAL}
            if matched["kind"] == "uncalibrated":
                better = calibration_advice(calibrations)
                if better:
                    matched = {**matched, "remedy": better}
            return {**matched, "detail": _tail(line)}
        return {
            "kind": "probe_failed",
            "headline": "the joint read failed and the snapshot omits joints",
            "remedy": "Open this robot's log (devices > logs) - the exception the probe raised is "
                      "recorded there in full.",
            "detail": _tail(line),
        }
    return None


def calibration_advice(available: Any) -> str | None:
    """A better remedy for ``uncalibrated`` when calibration files ALREADY EXIST on this machine.

    Measured on cagatay's rig 2026-08-21: so101-leader was spawned as a real robot with
    ``robot_id="leader"``, so lerobot looked for ``calibration/robots/so101_follower/leader.json``
    and raised "has no calibration registered". The arm was calibrated -- its file sits at
    ``calibration/teleoperators/so101_leader/leader.json``. The generic remedy ("Calibrate this
    arm") would therefore have sent the operator to re-teach a correctly calibrated arm: physical
    work, on hardware, to fix a filename. That is the worst kind of wrong advice this dashboard can
    give, so when the evidence contradicts it, it must not be given.

    ``available`` is what the caller found on disk: ``{"robots/so101_follower": ["follower",
    "leader_arm"], "teleoperators/so101_leader": ["leader"]}``. Pure -- it reads a listing that was
    already gathered, opens no port and touches no file.

    Returns None when the generic remedy is right (nothing on disk, or nothing we can say better),
    because a hint that fires on no evidence is how a diagnosis starts inventing.
    """
    if not isinstance(available, Mapping) or not available:
        return None
    where = sorted(
        f"{group}/{name}.json"
        for group, names in available.items()
        if isinstance(names, (list, tuple))
        for name in [str(n) for n in names]
    )
    if not where:
        return None
    shown = ", ".join(where[:6]) + (f" (+{len(where) - 6} more)" if len(where) > 6 else "")
    return (
        "Calibration files DO exist on this machine, so this is probably an id/path mismatch rather "
        "than an uncalibrated arm: lerobot looks for calibration/robots/<robot_type>/<id>.json, and "
        f"what exists is {shown}. Spawn this arm with the id whose file exists (or copy that file to "
        "the path lerobot wants) BEFORE recalibrating - re-teaching an arm that is already "
        "calibrated is physical work to fix a filename."
    )


def _self_healed(log_lines: Any) -> bool:
    """Whether this child's log shows it clearing a stranded in-use flag (see Q81).

    Read from the WHOLE log rather than the failing line, because the recovery and the failure are
    different events: the cure logs a warning at the moment it clears the flag, and the failure we are
    explaining may be the one that came after the cure gave up. One occurrence anywhere is enough to
    retire the stale-flag explanation, since a build that heals once heals always.
    """
    return any(_FLAG_CLEARED_LINE.search(str(x)) for x in log_lines)


def _tail(line: str, limit: int = 240) -> str:
    """The exception part of the log line, trimmed - a multi-page motor dump is not a badge."""
    text = line.strip()
    marker = "omitted (further failures logged at debug): "
    if marker in text:
        text = text.split(marker, 1)[1]
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _match(text: Any) -> dict[str, str] | None:
    """The signature this text carries, or None. Shared by the log and snapshot readers.

    Both sources describe the SAME exception -- one as a log line, one as the reason the peer
    published about itself -- so the remedies must come from one table. Two tables would drift, and
    the drift would show up as two different answers to "what do I do about this arm".
    """
    haystack = str(text).lower()
    for kind, needle, headline, remedy in _SIGNATURES:
        if needle.lower() in haystack:
            return {"kind": kind, "headline": headline, "remedy": remedy}
    return None


def classify_state(state: Any) -> dict[str, Any] | None:
    """Explain a joint fault the PEER ITSELF reported in its snapshot, or return None.

    ``mesh.core`` publishes ``state["degraded"]["hw_joints"] = {reason, failures, since,
    for_seconds}`` for as long as the probe is failing (commit dde98e46). That is a better source
    than this module's log parsing in four ways, which is why :func:`merge` prefers it:

    * it exists for EXTERNAL peers, which have no log ring buffer here at all;
    * it CLEARS itself when the probe recovers, so the badge cannot become permanent;
    * it says how long and how often, so "failed once" and "failing for three hours" are different
      sentences instead of the same one;
    * it is reported by the process that owns the bus, not inferred from text by a reader.

    The remedies still come from :data:`_SIGNATURES` via :func:`_match`, so both paths give the
    operator the same instruction for the same fault.
    """
    if not isinstance(state, Mapping):
        return None
    degraded = state.get("degraded")
    if not isinstance(degraded, Mapping):
        return None
    entry = degraded.get("hw_joints")
    if not isinstance(entry, Mapping):
        return None
    reason = str(entry.get("reason") or "").strip()
    if not reason:
        return None
    matched = _match(reason) or {
        "kind": "probe_failed",
        "headline": "the joint read failed and the snapshot omits joints",
        "remedy": "Open this robot's log (devices > logs) - the exception the probe raised is "
                  "recorded there in full.",
    }
    if matched["kind"] == "port_in_use" and _published_recoveries(state) > 0:
        # The snapshot's own fingerprint, exactly parallel to the log's (see _FLAG_CLEARED_LINE):
        # bus_recoveries counts stranded flags this peer has already cleared, so the stale-flag
        # explanation is spent here for the same reason it is spent there. Without this, the two
        # sources would answer "what do I do about this arm" differently - the drift this module's
        # single remedy table exists to prevent - and the SNAPSHOT path is the one external peers
        # have, so they would get the weaker answer precisely where no log exists to correct it.
        matched = {**matched, "remedy": _PORT_IN_USE_AFTER_SELF_HEAL}
    out: dict[str, Any] = {**matched, "detail": _tail(reason), "source": "peer"}
    for key in ("failures", "for_seconds"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            out[key] = value
    return out


def _published_recoveries(state: Any) -> int:
    """How many stranded in-use flags the peer says it has cleared (``bus_recoveries``).

    Absent on a peer running an older build and on a peer that never stranded a flag - the two are
    indistinguishable from here, and both correctly mean "no evidence", so the cautious verdict stands.
    A non-numeric or negative value is treated as no evidence rather than as a fault: an invented count
    would sharpen a diagnosis on nothing.
    """
    if not isinstance(state, Mapping):
        return 0
    value = state.get("bus_recoveries")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value) if value > 0 else 0


def has_joints(state: Any) -> bool:
    """Does this peer's published state carry any joint position at all?

    The positions live NESTED, at ``state["joints"]`` -- that is what ``mesh.core._read_state``
    publishes and what ``JointStrip`` renders. This function used to scan the TOP level of the state
    dict for keys ending in ``.pos``, which no real snapshot has, so it answered False for every
    healthy arm in the fleet. The consequence was invisible but serious: :func:`merge` clears a
    joint complaint only when this returns True, so a badge from one old log line could never clear
    itself -- the exact permanence the module docstring promises to avoid.

    A top-level ``*.pos`` key is still accepted, because a flat shape costs nothing to tolerate and
    a diagnosis module should not be the thing that breaks on a snapshot it did not expect.
    """
    if not isinstance(state, Mapping):
        return False
    joints = state.get("joints")
    if isinstance(joints, Mapping) and len(joints) > 0:
        return True
    return any(str(k).endswith(".pos") for k in state)


def merge(peer: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    """The annotation fields to apply to ``peer``, with a stale joint complaint removed.

    Two things can end a log-derived complaint: the arm publishing joints again, and (since the
    recovery line exists) the log itself saying the probe recovered, which ``classify`` honours. A
    child running older code logs no recovery, so for those the arm's joints remain the only proof --
    and a badge that cannot clear itself teaches the operator to ignore badges.
    """
    out = dict(fields)
    state = peer.get("state")
    if has_joints(state):
        # Live joints beat every complaint, from either source.
        out.pop("joint_problem", None)
        return out
    reported = classify_state(state)
    if reported is not None:
        # The peer's own report wins over this module's reading of its log: same fault, better
        # evidence (see classify_state). A log-derived verdict stays only when the peer publishes
        # nothing -- i.e. it runs code older than dde98e46.
        out["joint_problem"] = reported
    return out
