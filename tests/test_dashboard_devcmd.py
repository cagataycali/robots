"""``strands-robots dev`` — the pure rules the process manager stands on."""

import sys

from strands_robots.dashboard.dev import (
    calibration_verdicts,
    guard_ok,
    logs_to_prune,
    package_root,
    server_argv,
)


def test_server_argv_uses_this_interpreter_and_carries_the_token_file():
    argv = server_argv(8090, "/tmp/tok")
    assert argv[0] == sys.executable  # never a second venv
    assert argv[1:4] == ["-m", "strands_robots", "dashboard"]
    assert "--force" in argv
    assert argv[argv.index("--auth-token-file") + 1] == "/tmp/tok"


def test_server_argv_omits_token_flag_when_no_file():
    assert "--auth-token-file" not in server_argv(8090, None)


def test_package_root_is_the_imported_packages_parent():
    import strands_robots

    root = package_root()
    assert (root / "strands_robots" / "__init__.py").exists()
    assert str(root) in strands_robots.__file__


def test_guard_ok_requires_both_halves():
    assert guard_ok(401, 200)
    assert not guard_ok(200, 200)  # anonymous accepted = guard down
    assert not guard_ok(401, 401)  # token refused = wrong token
    assert not guard_ok(0, 0)      # server not answering


def test_logs_to_prune_keeps_the_newest_five():
    names = [f"dashboard_2026082{i}_000000.log" for i in range(7)]
    assert logs_to_prune(names) == names[:2]
    assert logs_to_prune(names[:5]) == []


def test_calibration_verdicts_flag_only_real_uncalibrated_profiles():
    profiles = {
        "a": {"name": "follower", "mode": "real", "robot_id": "so101f"},
        "b": {"name": "leader", "mode": "real", "robot_id": "ghost"},
        "c": {"name": "sim", "mode": "sim", "robot_id": "ignored"},
        "d": {"name": "no-id", "mode": "real"},
    }
    lines = calibration_verdicts(profiles, lambda rid: rid == "so101f")
    assert len(lines) == 2  # sim and id-less profiles say nothing
    assert any("ok: follower" in ln for ln in lines)
    assert any("MISSING: leader" in ln and "NO JOINTS" in ln for ln in lines)
