"""Does another PROCESS already hold this arm's serial bus? (BUGS.md Q84)

The dashboard has always refused to spawn a peer_id it is already running -- an in-process check
against ``self.robots``. That check is structurally blind to the failure that actually happened:
185 parentless processes, arriving in triples every ~5 minutes for 10 hours, each holding
/dev/cu.usbmodem* and reading registers, ~50 concurrent readers per Feetech bus. Every one of them
was invisible to ``self.robots`` because none of them belonged to this dashboard.

What the operator saw was not an error but an ABSENCE: a fresh spawn cannot get clean bus access
against that crowd, so it publishes no joints, so ageing prunes its peer, so the next spawn tries
again and leaks three more holders. The fleet simply had no arms in it, and 39 consecutive caretaker
runs called that healthy.

So the question this module asks is the one no in-process lock can: who else, in the whole machine,
has this device open right now? It is deliberately a REFUSAL and not a consent gate -- a second owner
on a half-duplex serial bus corrupts both conversations, so there is nothing for the operator to
agree to. What it owes them instead is the cure, named in the message.
"""
from __future__ import annotations

import os
import shutil
import subprocess


def sibling_devices(port: str) -> list[str]:
    """A macOS serial port is two device files for one piece of hardware.

    /dev/cu.usbmodemX (call-up, non-blocking open) and /dev/tty.usbmodemX are separate paths that
    reach the SAME UART, so a process holding one blocks the other while lsof on our path alone
    reports nothing. Checking both is what makes the answer true rather than merely reassuring.
    """
    out = [port]
    base = os.path.basename(port)
    for a, b in (("cu.", "tty."), ("tty.", "cu.")):
        if base.startswith(a):
            out.append(os.path.join(os.path.dirname(port), b + base[len(a):]))
    return out


def bus_holders(port: str, *, _run=None) -> list[int]:
    """Pids with this bus (or its sibling device) open, excluding ourselves.

    lsof is not on the agent shell's PATH on this Mac, hence the explicit /usr/sbin probe. A missing
    lsof returns [] -- no evidence is not evidence of a conflict, and refusing a spawn because we
    could not look would break the machine we are trying to protect.
    """
    run = _run or (lambda argv: subprocess.run(argv, capture_output=True, text=True, timeout=8))
    exe = shutil.which("lsof") or ("/usr/sbin/lsof" if os.path.exists("/usr/sbin/lsof") else None)
    if not exe:
        return []
    pids: set[int] = set()
    for dev in sibling_devices(port):
        try:
            res = run([exe, "-nP", "-t", dev])
        except Exception:
            continue
        for line in (getattr(res, "stdout", "") or "").split():
            if line.strip().isdigit():
                pids.add(int(line))
    pids.discard(os.getpid())
    return sorted(pids)


def bus_conflict(port: str, holders: list[int], tracked: dict[int, str]) -> str | None:
    """The refusal text, or None when the bus is ours to take. Pure -- the whole judgement is here.

    ``tracked`` maps pid -> peer_id for the children THIS dashboard is running, because the two cases
    need opposite advice: our own child means "despawn it or use another name" (a normal, expected
    collision), while a stranger means "something is holding your hardware that nobody is managing"
    and the reaper is the cure. Conflating them would send the operator to the despawn button for a
    process the dashboard cannot see, which is the confusion Q84 lived in for ten hours.
    """
    if not holders:
        return None
    ours = {p: tracked[p] for p in holders if p in tracked}
    strangers = [p for p in holders if p not in tracked]
    if not strangers:
        names = ", ".join(sorted({f"{peer} (pid {pid})" for pid, peer in ours.items()}))
        return (
            f"{port} is already held by this dashboard's own robot: {names}. Despawn it first, or "
            f"spawn the new one on a different port."
        )
    listed = ", ".join(str(p) for p in strangers)
    extra = ""
    if ours:
        extra = f" This dashboard also runs {', '.join(sorted(set(ours.values())))} on it."
    return (
        f"{port} is held by {len(strangers)} process(es) this dashboard does not manage "
        f"(pid {listed}).{extra} A second owner on a half-duplex serial bus corrupts both "
        f"conversations, so the spawn is refused rather than started blind. If those are leftovers "
        f"(BUGS.md Q84 -- parentless spawn children pile up unnoticed and the only symptom is an arm "
        f"missing from the fleet), run reap_orphan_buses.sh; it kills only parentless holders and "
        f"moves no arm. If a holder survives that, unplug and replug the arm."
    )
