"""Q81: make a test run that left a REAL robot child behind visible, in that run.

MEASURED 2026-08-20. ``tests/test_dashboard_datasets_route_recording.py`` built the dashboard app
with ``with TestClient(app)``; the startup hook started USB auto-spawn; auto-spawn scanned the real
serial bus and brought cagatay's two SO-101 arms up as real robot children. Every one of those runs
passed, printed a green summary, and exited -- orphaning its children (ppid=1) with the arm ports
still open. 185 of them accumulated from ~30 runs of that single file, and the consequence was
invisible in the suite and brutal on the rig: the live arm child could not read one byte
(``[TxRxResult] Port is in use!``) and the dashboard showed a connected arm with zero joints.

``device_manager.autospawn_veto`` now refuses that door (a pytest process may not take a serial
port). This module is the tripwire behind the fix, because the next way in will not be auto-spawn:
any test that reaches ``spawn`` with a real port, or a future startup hook, lands here. A green run
that leaked a hardware holder must SAY so -- the whole cost of Q81 was that thirty runs looked
perfect while a rig quietly became unusable.

Deliberately it reports and does not kill. A robot child may hold TORQUE, and killing it can let a
powered arm go limp and fall; that is motion, which no automated rail here is allowed to cause. The
report names the pids and the recipe instead, and leaves the decision to a human standing next to
the hardware.

Pure by design so it can be tested: it takes what was found and returns lines to print, or nothing
at all when the run was clean.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping


#: A surviving child is a ROBOT child if its command line shows either the robot bootstrap or a
#: serial device. Both matter: the bootstrap names the peer, the port names what is held.
_ROBOT = re.compile(r"strands_robots|peer_id", re.I)
_PORT = re.compile(r"(/dev/(?:cu|tty)\.[\w.-]+)")


def robot_children(children: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """The surviving children that look like real robot processes, with the port each holds."""
    out: list[dict[str, object]] = []
    for child in children or []:
        cmd = str(child.get("cmdline") or "")
        if not _ROBOT.search(cmd):
            continue
        port = _PORT.search(cmd)
        out.append({
            "pid": child.get("pid"),
            "port": port.group(1) if port else None,
            "peer_id": _peer_id(cmd),
        })
    return out


def _peer_id(cmd: str) -> str | None:
    m = re.search(r"[\"']peer_id[\"']\s*:\s*[\"']([\w.-]+)[\"']", cmd)
    return m.group(1) if m else None


def hardware_leak_report(children: Iterable[Mapping[str, object]]) -> list[str]:
    """Lines to print when this run is about to orphan real robot processes.

    Silence when clean: a guard that prints something after every run is a guard nobody reads.
    """
    leaked = robot_children(children)
    if not leaked:
        return []
    lines = [
        "",
        "!! THIS TEST RUN IS ABOUT TO ORPHAN %d REAL ROBOT PROCESS(ES) (Q81)." % len(leaked),
    ]
    for item in leaked:
        who = item["peer_id"] or "unknown peer"
        where = item["port"] or "no serial port in its command line"
        lines.append("     pid %s  %s  holding %s" % (item["pid"], who, where))
    lines += [
        "   They will become ppid=1 when pytest exits and keep the port open, so the operator's own",
        "   robot child cannot read its motors: '[TxRxResult] Port is in use!' with a connected arm",
        "   showing zero joints. 185 such orphans accumulated on 2026-08-20 before anyone noticed.",
        "   A test must not take a physical port: device_manager.autospawn_veto refuses auto-spawn,",
        "   so whatever created these is a NEW door - find it rather than cleaning up after it.",
        "   NOT killed here on purpose: a robot child may hold torque, and dropping a powered arm is",
        "   motion. Park the arm first, then: kill %s" % " ".join(str(i["pid"]) for i in leaked),
        "",
    ]
    return lines
