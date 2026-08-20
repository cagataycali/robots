#!/usr/bin/env python3
"""List (and, with --confirm, kill) ORPHANED robot children still holding an arm's serial port.

Q80/Q81. On 2026-08-20 a single test file spawned real SO-101 children ~30 times; each run exited and
left them orphaned (ppid=1) with the arm ports open. 185 accumulated, both arm ports were held by
~93 processes each, and every freshly spawned arm streamed cameras with ZERO joints — a respawn
cannot fix it, because the newcomer loses the same race for the port. autospawn_veto (a333287b) shut
the door; this clears what came through it.

WHY A SCRIPT AND NOT A ONE-LINER. The recipe in BUGS.md was `kill <pids from lsof>`, which asks a
tired human to re-derive the safety reasoning every time: that pid list contains the LIVE arm
children too, and killing those mid-motion is exactly what nobody wants at 1am. The judgement lives
here instead, tested, and refuses by default:

  * only a process whose parent is init (ppid=1) is a candidate — a live child of the dashboard is
    still someone's robot;
  * only a process whose command line looks like the robot bootstrap — never an editor, a shell, or
    anything else that happens to have the port open;
  * nothing at all without --confirm; the default is a report.

TORQUE IS THE REAL RISK, AND NO SCRIPT CAN CHECK IT. A robot child may hold its motors energised;
killing it can let a powered arm go limp and fall. That is physical motion, so this file will not
decide it — it prints the warning and waits for a human who can see the hardware. Park the arms
first, then pass --confirm.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

PORT_RE = re.compile(r"/dev/(?:cu|tty)\.usbmodem[\w.-]+")
ROBOT_RE = re.compile(r"strands_robots|peer_id", re.I)


def classify_holder(
    *,
    pid: int,
    ppid: int,
    cmdline: str,
    live_pids: frozenset[int] = frozenset(),
) -> tuple[bool, str]:
    """Whether this port holder may be killed, and the reason either way.

    Pure: every fact it needs is an argument, so the decision is testable without a rig.
    """
    if pid == os.getpid():
        return False, "this script itself"
    if pid in live_pids:
        return False, "a LIVE child of the running dashboard — someone's robot is using it"
    if ppid != 1:
        return False, f"not orphaned (parent is {ppid}) — it belongs to a running process"
    if not ROBOT_RE.search(cmdline):
        return False, "does not look like a robot child — refusing to kill an unrelated process"
    return True, "orphaned robot child (ppid=1) holding an arm port"


def port_holders() -> list[dict[str, object]]:
    """Every process holding a usbmodem serial port, with its ppid and command line."""
    try:
        lsof = subprocess.run(
            ["/usr/sbin/lsof", "-n"], capture_output=True, text=True, timeout=60
        ).stdout
    except Exception as e:  # noqa: BLE001
        print(f"could not run lsof: {e}", file=sys.stderr)
        return []

    ports: dict[int, set[str]] = {}
    for line in lsof.splitlines():
        m = PORT_RE.search(line)
        if not m:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            ports.setdefault(int(parts[1]), set()).add(m.group(0))
        except ValueError:
            continue

    ps = subprocess.run(
        ["/bin/ps", "-ax", "-o", "pid=,ppid=,command="], capture_output=True, text=True, timeout=30
    ).stdout
    meta: dict[int, tuple[int, str]] = {}
    for line in ps.splitlines():
        bits = line.split(None, 2)
        if len(bits) == 3:
            try:
                meta[int(bits[0])] = (int(bits[1]), bits[2])
            except ValueError:
                continue

    return [
        {"pid": pid, "ppid": meta.get(pid, (0, ""))[0], "cmdline": meta.get(pid, (0, ""))[1],
         "ports": sorted(held)}
        for pid, held in sorted(ports.items())
    ]


def live_dashboard_children(dashboard_pid: int | None) -> frozenset[int]:
    """Descendants of the running dashboard — never candidates, whatever they hold."""
    if not dashboard_pid:
        return frozenset()
    ps = subprocess.run(
        ["/bin/ps", "-ax", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=30
    ).stdout
    kids: dict[int, list[int]] = {}
    for line in ps.splitlines():
        bits = line.split()
        if len(bits) == 2:
            try:
                kids.setdefault(int(bits[1]), []).append(int(bits[0]))
            except ValueError:
                continue
    out, stack = set(), list(kids.get(dashboard_pid, ()))
    while stack:
        pid = stack.pop()
        if pid in out:
            continue
        out.add(pid)
        stack.extend(kids.get(pid, ()))
    return frozenset(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", action="store_true", help="actually kill the orphans (park the arms first)")
    ap.add_argument("--dashboard-pid", type=int, default=None, help="protect this process's descendants")
    args = ap.parse_args()

    live = live_dashboard_children(args.dashboard_pid)
    holders = port_holders()
    if not holders:
        print("no process is holding an arm serial port — nothing to clear.")
        return 0

    kill, keep = [], []
    for h in holders:
        ok, why = classify_holder(
            pid=int(h["pid"]), ppid=int(h["ppid"]), cmdline=str(h["cmdline"]), live_pids=live
        )
        (kill if ok else keep).append((h, why))

    print(f"{len(holders)} process(es) hold an arm serial port: {len(kill)} orphaned, {len(keep)} kept.")
    for h, why in keep:
        print(f"  KEEP pid {h['pid']:>7}  {', '.join(h['ports'])}  — {why}")
    if kill:
        ports = sorted({p for h, _ in kill for p in h["ports"]})
        print(f"  KILL {len(kill)} orphaned robot children holding {', '.join(ports)}")

    if not args.confirm:
        print("\nDRY RUN — nothing was killed. A robot child may hold its motors energised, and killing")
        print("it can let a powered arm go limp and FALL. Park the arms, then re-run with --confirm.")
        return 0

    failed = 0
    for h, _ in kill:
        try:
            os.kill(int(h["pid"]), 9)
        except ProcessLookupError:
            pass
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  could not kill {h['pid']}: {e}")
    print(f"killed {len(kill) - failed} orphan(s). Respawn the arms afterwards — a robot that lost the")
    print("port race is not repaired by the port becoming free, it has to be started again.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
