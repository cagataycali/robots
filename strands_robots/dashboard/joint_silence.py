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

#: Ordered: the FIRST match wins, so the specific readings are tried before the generic one.
_SIGNATURES: tuple[tuple[str, str, str, str], ...] = (
    (
        "port_in_use",
        "Port is in use!",
        "another process is holding this arm's serial port",
        "Find the other owner and stop it - `/usr/sbin/lsof -n | grep usbmodem` names every "
        "holder. Orphaned robot children from an earlier spawn keep a port open; replugging and "
        "recalibrating both change nothing while one is alive.",
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


def classify(log_lines: Any) -> dict[str, str] | None:
    """Explain the NEWEST degraded ``hw_joints`` probe in ``log_lines``, or return None.

    Silence is the common case and must stay silent: a child whose log never mentions the probe
    contributes nothing, so a healthy arm carries no field at all rather than an empty verdict.
    """
    if not isinstance(log_lines, (list, tuple)):
        return None
    for line in reversed([str(x) for x in log_lines]):
        if not _PROBE_LINE.search(line):
            continue
        for kind, needle, headline, remedy in _SIGNATURES:
            if needle.lower() in line.lower():
                return {"kind": kind, "headline": headline, "remedy": remedy,
                        "detail": _tail(line)}
        return {
            "kind": "probe_failed",
            "headline": "the joint read failed and the snapshot omits joints",
            "remedy": "Open this robot's log (devices > logs) - the exception the probe raised is "
                      "recorded there in full.",
            "detail": _tail(line),
        }
    return None


def _tail(line: str, limit: int = 240) -> str:
    """The exception part of the log line, trimmed - a multi-page motor dump is not a badge."""
    text = line.strip()
    marker = "omitted (further failures logged at debug): "
    if marker in text:
        text = text.split(marker, 1)[1]
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def has_joints(state: Any) -> bool:
    """Does this peer's published state carry any joint position at all?"""
    if not isinstance(state, Mapping):
        return False
    return any(str(k).endswith(".pos") for k in state)


def merge(peer: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    """The annotation fields to apply to ``peer``, with a stale joint complaint removed.

    A recovered probe is never logged (see the module docstring), so the ONLY evidence that the
    fault is over is the arm publishing joints again. When it does, the past complaint is dropped
    rather than shown - a badge that cannot clear itself teaches the operator to ignore badges.
    """
    out = dict(fields)
    if "joint_problem" in out and has_joints(peer.get("state")):
        out.pop("joint_problem")
    return out
