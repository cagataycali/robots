"""A process-global settings override must not survive the test that made it (BUGS.md Q62).

The failure this pins was invisible for weeks: test_dashboard_ws_chat_frames passed alone and
failed in every sweep, because test_dashboard_lan_hint's fixture overrode security.auth_token for
the whole PROCESS and monkeypatch cannot revert a call it never made. The victim's websocket
handshake was then rejected (WebSocketDisconnect) in a file that never mentions auth.

So the regression test has to be about ORDER, which means running pytest inside pytest: the pair,
in the order that reproduced it, with shuffling off.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_override_leaked_into_this_test():
    """Cheap half: whatever ran before us in this process left our settings clean."""
    from strands_robots.dashboard import settings

    assert settings._overrides == {}, (  # noqa: SLF001
        f"a previous test leaked a settings override: {settings._overrides}"  # noqa: SLF001
    )


def test_the_leaker_and_its_victim_pass_in_that_order():
    """The real pin: the exact reproduction, order fixed on the command line."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_dashboard_lan_hint.py",
            "tests/test_dashboard_ws_chat_frames.py",
            "-q", "--no-header", "-p", "no:cacheprovider", "-p", "no:randomly",
            "--no-cov",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        # A clean-ish env: the fixture under test must not depend on this machine's live dashboard
        # environment to do its job.
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(Path.home())},
    )
    tail = "\n".join(proc.stdout.strip().splitlines()[-6:])
    assert proc.returncode == 0, f"the sweep-only failure is back:\n{tail}"
