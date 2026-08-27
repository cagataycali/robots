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
    assert not guard_ok(0, 0)  # server not answering


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


# ---------------------------------------------------------------- arm buses
#
# The lsof fixtures below are real ``lsof -nP`` output captured on a Linux host
# (lsof 4.95.0), not invented strings: the header row, the field layout, and the
# false-positive line are verbatim. Only the arm device NAME is substituted where
# no arm was plugged in, which is exactly the axis under test.

LSOF_HEADER = (
    "COMMAND       PID     TID TASKCMD               USER   FD      TYPE  "
    "           DEVICE      SIZE/OFF      NODE NAME"
)
LSOF_LINUX_ARM = (
    "python3    543393                            cagatay    3u      CHR  "
    "           166,0           0t0       119 /dev/ttyACM0"
)
LSOF_LINUX_BRIDGE = (
    "python3    543394                            cagatay    3u      CHR  "
    "           188,0           0t0       120 /dev/ttyUSB0"
)
LSOF_MACOS_ARM = (
    "python3    543395                            cagatay    3u      CHR  "
    "            18,7           0t0       620 /dev/cu.usbmodem14201"
)
# Captured verbatim while a probe held a REGULAR FILE named like a bus.
LSOF_LOOKALIKE_FILE = (
    "python3    543393                            cagatay    4r      REG  "
    "           259,1             0  55592852 /tmp/busmeasure-al3axs7k/ttyACM0"
)


def test_the_bus_sweep_names_the_device_families_of_both_platforms():
    """The sweep is a substring match, so a device family it does not name is a
    silent no-op on that platform -- the sweep reports "arm buses free" without
    having looked. Both families have to be named."""
    from strands_robots.dashboard import dev

    named = " ".join(dev.ARM_BUS_MARKERS)
    for family in ("cu.usbmodem", "ttyACM", "ttyUSB"):
        assert family in named, (
            f"the arm-bus sweep names no {family} device: on that platform it "
            f"matches nothing and still reports the buses free (named: {named})"
        )


def test_arm_bus_holder_pids_reads_a_linux_arm_bus():
    from strands_robots.dashboard.dev import arm_bus_holder_pids

    assert arm_bus_holder_pids(f"{LSOF_HEADER}\n{LSOF_LINUX_ARM}") == {543393}
    assert arm_bus_holder_pids(f"{LSOF_HEADER}\n{LSOF_LINUX_BRIDGE}") == {543394}


def test_arm_bus_holder_pids_still_reads_a_macos_arm_bus():
    """Additive, not a swap: the macOS marker keeps matching what it matched."""
    from strands_robots.dashboard.dev import arm_bus_holder_pids

    assert arm_bus_holder_pids(f"{LSOF_HEADER}\n{LSOF_MACOS_ARM}") == {543395}


def test_arm_bus_holder_pids_reads_a_mixed_capture_whole():
    from strands_robots.dashboard.dev import arm_bus_holder_pids

    out = "\n".join([LSOF_HEADER, LSOF_MACOS_ARM, LSOF_LINUX_ARM, LSOF_LINUX_BRIDGE])
    assert arm_bus_holder_pids(out) == {543393, 543394, 543395}


def test_arm_bus_holder_pids_ignores_a_regular_file_named_like_a_bus():
    """A file called ttyACM0 is not a bus. The sweep SIGKILLs whatever it
    returns, so matching the name anywhere on the line would kill a process for
    holding a log file."""
    from strands_robots.dashboard.dev import arm_bus_holder_pids

    assert arm_bus_holder_pids(f"{LSOF_HEADER}\n{LSOF_LOOKALIKE_FILE}") == set()


def test_arm_bus_holder_pids_reads_only_rows_shaped_like_lsof_file_rows():
    """The helper parses another program's stdout, so a row that is not a file
    row contributes nothing rather than raising: the header (second field is the
    word PID) and a truncated capture both carry a device path."""
    from strands_robots.dashboard.dev import arm_bus_holder_pids

    assert arm_bus_holder_pids(LSOF_HEADER) == set()
    assert arm_bus_holder_pids("") == set()
    assert arm_bus_holder_pids("/dev/ttyACM0") == set()
    assert arm_bus_holder_pids("python3 notapid cagatay 3u CHR /dev/ttyACM0") == set()


def test_tty_refusal_names_macos_only_where_that_is_the_reason():
    """The refusal stands everywhere, but only macOS has the TCC mechanism to
    cite; elsewhere it must not borrow another operating system's reason."""
    from strands_robots.dashboard.dev import tty_refusal_reason

    assert "macOS" in tty_refusal_reason("darwin")
    for other in ("linux", "win32"):
        reason = tty_refusal_reason(other)
        assert "macOS" not in reason, f"{other} refusal cites macOS: {reason}"
        assert reason, "the refusal still has to say why"


def test_stop_sweeps_a_linux_arm_bus_it_finds_held(monkeypatch):
    """The behavioural half: drive the real ``stop`` over a capture that shows a
    Linux arm bus held by an orphan, and check the sweep actually reaches it.

    Pre-fix ``stop`` matched ``cu.usbmodem`` against the line, so on Linux it
    swept nothing, killed nothing, and still printed "ports and arm buses free"
    -- a report of a check it had not performed.
    """
    from strands_robots.dashboard import dev

    capture = f"{LSOF_HEADER}\n{LSOF_LINUX_ARM}\n{LSOF_LINUX_BRIDGE}"
    killed: list[int] = []

    monkeypatch.setattr(dev, "_pgrep", lambda: [])  # nothing to SIGTERM
    monkeypatch.setattr(dev, "_port_holder", lambda port: None)  # ports already free
    monkeypatch.setattr(dev, "_lsof", lambda: "/usr/bin/lsof")
    monkeypatch.setattr(
        dev.subprocess,
        "run",
        lambda *a, **k: type("R", (), {"stdout": capture, "returncode": 0})(),
    )
    monkeypatch.setattr(dev.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(dev.time, "sleep", lambda _s: None)

    assert dev.stop(8090) == 0
    assert sorted(killed) == [543393, 543394], (
        f"stop swept {sorted(killed)} -- a Linux arm bus was held by an orphan and the sweep did not reach it"
    )
