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


def test_two_apps_in_one_process_do_not_share_the_camera_close_log_budget(monkeypatch, tmp_path):
    """Q63: the second leak of this class was a PRODUCT global, not a test's mistake.

    The close-log throttle and the churn guard must outlive individual SOCKETS — that is the whole
    requirement. At module level they also outlived the APP, so a reopen storm against one app
    silenced close lines for another (and a later test read that silence as "the verdict never
    reached the log"). A dashboard process serves one app, so per-app state is identical in
    production and merely honest here.
    """
    from strands_robots.dashboard import auth
    from strands_robots.dashboard import settings as dsettings
    from strands_robots.dashboard.server import create_app

    monkeypatch.setenv("STRANDS_MESH", "false")
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.setattr(dsettings, "SETTINGS_FILE", tmp_path / "settings.json")
    dsettings._cache = None  # noqa: SLF001
    auth._cache_key = None  # noqa: SLF001
    auth._cache = {}  # noqa: SLF001

    first, second = create_app(), create_app()
    assert first.state.camera_close_log is not second.state.camera_close_log
    assert first.state.camera_churn is not second.state.camera_churn

    name = "so101-arm-1/top"
    for _ in range(50):  # spend the first app's whole budget for that camera
        first.state.camera_close_log.should_log(name)
    assert first.state.camera_close_log.should_log(name)[0] is False, "budget should be spent"
    assert second.state.camera_close_log.should_log(name)[0] is True, (
        "a storm against one app must not silence another app's close verdict"
    )


def test_the_churn_storm_and_the_close_log_pass_in_that_order():
    """The ordered pin, same technique as Q62: pytest inside pytest."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_dashboard_churn_wiring.py",
            "tests/test_dashboard_ws_close_log.py",
            "-q", "--no-header", "-p", "no:cacheprovider", "-p", "no:randomly", "--no-cov",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(Path.home())},
    )
    tail = "\n".join(proc.stdout.strip().splitlines()[-6:])
    assert proc.returncode == 0, f"the sweep-only close-log failure is back:\n{tail}"
