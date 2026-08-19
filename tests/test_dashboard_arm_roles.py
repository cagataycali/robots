"""An arm's role is measured off its servo bus, not inherited from a label (U2).

The operator's report: "the leader arm is follower, follower is leader on the
dashboard at the moment". Nothing in the dashboard measured the difference - the
role came from whatever name a profile carried, so a swap was invisible exactly
where it matters (a record session drives the follower).

Ground truth on an SO-100/SO-101 pair: follower bus = 12V, leader = 7.4V, and
every Feetech servo reports its own supply on the read-only Present_Voltage
register (address 62 in lerobot's table).
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from strands_robots.dashboard import arm_roles as ar
from strands_robots.dashboard import device_manager as dm


# ------------------------------------------------------------ classification


def test_twelve_volt_bus_is_the_follower():
    role, reason = ar.classify_role(12.1)
    assert role == "follower" and "12V" in reason


def test_seven_four_volt_bus_is_the_leader():
    role, reason = ar.classify_role(7.6)
    assert role == "leader" and "7.4V" in reason


def test_an_unpowered_arm_is_never_called_a_leader():
    """MEASURED LIVE, and it corrected this module: an SO-101 with its power
    supply OFF answers 5.5-5.6V on all six servos - the USB logic rail, not a
    battery near zero. The first threshold (5.5V floor) therefore called an
    unpowered arm "leader" by a tenth of a volt, which is how a follower gets
    driven as a teleoperator - the exact swap U2 exists to catch."""
    role, reason = ar.classify_role(5.55)
    assert role == "unpowered"
    assert "USB logic rail" in reason and "6.6-8.4V" in reason

    # A genuinely dead bus is still unpowered, with the plainer sentence.
    role, reason = ar.classify_role(0.4)
    assert role == "unpowered" and "not on its power supply" in reason


def test_the_live_readings_from_this_rig():
    """The two real arms, verbatim from /api/devices/arm-role."""
    arm2 = ar.role_verdict({"m1": 12.6, "m2": 12.6, "m3": 12.6, "m4": 12.7, "m5": 12.6, "m6": 12.7})
    assert arm2["role"] == "follower" and arm2["volts"] == 12.6

    arm1 = ar.role_verdict({"m1": 5.6, "m2": 5.6, "m3": 5.5, "m4": 5.5, "m5": 5.6, "m6": 5.5})
    assert arm1["role"] == "unpowered", "5.55V must never read as a role"
    assert "power supply" in arm1["remedy"]


def test_no_reading_is_unknown_not_a_guess():
    assert ar.classify_role(None)[0] == "unknown"


# ------------------------------------------------------------------- verdict


def test_one_bad_servo_does_not_rename_the_arm():
    """A single 0.0 (or a byte read gone wrong) must not drag a 12V bus down."""
    v = ar.role_verdict({"m1": 12.0, "m2": 12.1, "m3": 12.0, "m4": 0.0, "m5": 12.0})
    assert v["role"] == "mixed"  # the spread is a fault, and it is reported as one
    assert "fault, not a role" in v["reason"] and v["remedy"]


def test_a_clean_twelve_volt_bus_reads_follower():
    v = ar.role_verdict({f"m{i}": 12.0 + (i % 2) * 0.1 for i in range(1, 7)})
    assert v["role"] == "follower" and v["motors_answered"] == 6
    assert v["volts"] == pytest.approx(12.05, abs=0.06) and v["spread"] <= 0.11


def test_a_clean_leader_bus_reads_leader():
    assert ar.role_verdict({"m1": 7.5, "m2": 7.4, "m3": 7.5})["role"] == "leader"
    # The bottom of a real 7.4V pack still reads leader; the USB rail does not.
    assert ar.role_verdict({"m1": 6.7})["role"] == "leader"
    assert ar.role_verdict({"m1": 6.4})["role"] == "unpowered"


def test_silence_is_a_refusal_with_a_remedy():
    v = ar.role_verdict({"m1": None, "m2": None})
    assert v["role"] == "unknown" and "no servo answered" in v["reason"]
    assert "power supply" in v["remedy"]


# --------------------------------------------------------------- disagreement


def test_a_swapped_label_is_named_explicitly():
    d = ar.disagreement("leader", ar.role_verdict({"m1": 12.0, "m2": 12.0}))
    assert d and d["labelled"] == "leader" and d["measured"] == "follower"
    assert "12.0V" in d["message"] and "relabel it follower" in d["remedy"]


def test_a_matching_label_says_nothing():
    assert ar.disagreement("follower", ar.role_verdict({"m1": 12.0})) is None


def test_a_faulty_measurement_never_accuses_the_label():
    """mixed/unknown means our reading is bad - not that the operator is wrong."""
    assert ar.disagreement("leader", {"role": "mixed", "volts": 6.0}) is None
    assert ar.disagreement("leader", {"role": "unknown"}) is None
    assert ar.disagreement(None, ar.role_verdict({"m1": 12.0})) is None


# ----------------------------------------------------------------- bus access


def _mgr(tmp_path):
    return dm.DeviceManager(profiles_path=str(tmp_path / "profiles.json"))


def test_a_live_child_owns_its_port_and_the_read_is_refused(tmp_path):
    mgr = _mgr(tmp_path)
    mgr.robots["so101-arm-1"] = SimpleNamespace(
        peer_id="so101-arm-1", port="/dev/cu.usbmodem1", alive=lambda: True,
    )
    with pytest.raises(PermissionError, match="held by so101-arm-1"):
        mgr.read_bus_role("/dev/cu.usbmodem1")
    # A dead child does not own anything.
    mgr.robots["so101-arm-1"].alive = lambda: False
    assert mgr.port_owner("/dev/cu.usbmodem1") is None


def test_the_read_runs_in_a_child_and_parses_its_voltages(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return SimpleNamespace(stdout=json.dumps({"m1": 12.0, "m2": 12.1}), stderr="")

    monkeypatch.setattr(dm.subprocess, "run", fake_run)
    v = _mgr(tmp_path).read_bus_role("/dev/cu.usbmodem1")
    assert v["role"] == "follower" and v["port"] == "/dev/cu.usbmodem1"
    # Reads only: the child script must not contain a single write verb.
    src = seen["argv"][2]
    assert "Present_Voltage" in src
    for forbidden in ("write", "Goal_Position", "Torque_Enable", "sync_write"):
        assert forbidden not in src


def test_a_hung_bus_becomes_a_verdict_not_an_exception(tmp_path, monkeypatch):
    def boom(argv, **kw):
        raise subprocess.TimeoutExpired("python", 25)

    monkeypatch.setattr(dm.subprocess, "run", boom)
    v = _mgr(tmp_path).read_bus_role("/dev/cu.usbmodem1", timeout=25)
    assert v["role"] == "unknown" and "did not answer within 25s" in v["reason"]
    assert "replug" in v["remedy"]


def test_the_sdks_own_words_survive_a_failed_read(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dm.subprocess, "run",
        lambda argv, **kw: SimpleNamespace(stdout="{}", stderr="m1: No response from motor 1\n"),
    )
    v = _mgr(tmp_path).read_bus_role("/dev/cu.usbmodem1")
    assert v["role"] == "unknown" and v["detail"] == "m1: No response from motor 1"


def test_garbage_on_stdout_does_not_crash_the_read(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dm.subprocess, "run",
        lambda argv, **kw: SimpleNamespace(stdout="not json at all", stderr=""),
    )
    assert _mgr(tmp_path).read_bus_role("/dev/cu.usbmodem1")["role"] == "unknown"


def test_profiles_are_looked_up_by_serial_not_by_port(tmp_path, monkeypatch):
    """A /dev name is reassigned by the OS; the serial is the board. Looking up
    by port would silently return None forever - a mismatch check that checks
    nothing."""
    mgr = _mgr(tmp_path)
    mgr.profiles.save("5AB0158428", {"robot_name": "so101", "role": "leader"})
    monkeypatch.setattr(
        dm, "scan_serial_ports",
        lambda: [{"device": "/dev/cu.usbmodem5AB01584281", "serial_number": "5AB0158428"}],
    )
    assert mgr.profile_for_port("/dev/cu.usbmodem5AB01584281")["role"] == "leader"
    assert mgr.profile_for_port("/dev/cu.usbmodem-other") is None
