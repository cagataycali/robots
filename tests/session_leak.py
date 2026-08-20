"""Q32: make a test run that stays joined to the live fleet VISIBLE.

A pytest process is supposed to end when the report is printed. Three of them
were found still attached to the mesh hub hours (one of them three days) after
their suites finished: the interpreter hung in
``Py_FinalizeEx -> wait_for_thread_shutdown``, joining a non-daemon pool worker
stuck inside a zenoh call on a session no test ever closed. Meanwhile each ghost
published presence, so it appeared as a live ``gateway-*`` peer on the very
screen an operator reads to judge whether the fleet is healthy.

The report below is pure so it can be tested: it takes what was found and
returns the lines to print, or nothing at all when the run was clean.
"""

from __future__ import annotations

from collections.abc import Iterable


def leak_report(*, session_open: bool, threads: Iterable[str]) -> list[str]:
    """Lines describing what would keep this interpreter (and its mesh link) alive.

    ``threads`` is the name of every *non-daemon* thread still alive besides the
    main one - those are what ``wait_for_thread_shutdown`` will join, so they are
    the difference between a run that exits and a ghost peer on the fleet screen.
    """
    names = sorted({t for t in threads if t and t != "MainThread"})
    if not session_open and not names:
        return []
    lines = ["", "=" * 72, "LEAK: this pytest process is not ready to exit (BUGS.md Q32)"]
    if session_open:
        lines += [
            "  * a global mesh session is STILL OPEN - while it is, this process is a",
            "    peer on the live fleet and anything it publishes reaches real hardware.",
            "    Closing it now; the test that opened it should close it itself.",
        ]
    if names:
        lines += [
            f"  * {len(names)} non-daemon thread(s) still alive: {', '.join(names)}",
            "    Python joins these at shutdown, so a stuck one hangs the run forever",
            "    (that is how the ghosts survived 4 hours and 3 days).",
        ]
    lines += ["=" * 72, ""]
    return lines
