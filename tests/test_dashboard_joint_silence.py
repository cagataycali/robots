"""A connected arm with no joints must say WHY (Q80).

Live on 2026-08-20 both of cagatay's arms reported `hardware connected`, kept a fresh heartbeat,
listed cameras -- and omitted every joint from every snapshot for hours. The reason sat in each
child's log, in words, where the fleet view never looks: one arm's serial port was held by another
process (179 orphaned children from earlier spawns), the other board had no calibration at all.
From outside the two are identical; their remedies are opposite.
"""
import pytest

from strands_robots.dashboard import joint_silence

IN_USE = (
    "13:58:52 WARNING:strands_robots.mesh.core:[mesh] so101-follower: state probe 'hw_joints' "
    "failed, that section of the snapshot is omitted (further failures logged at debug): "
    "ConnectionError(\"Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] after 3 "
    "tries. [TxRxResult] Port is in use!\")"
)
UNCALIBRATED = (
    "13:59:22 WARNING:strands_robots.mesh.core:[mesh] so101-leader: state probe 'hw_joints' "
    "failed, that section of the snapshot is omitted (further failures logged at debug): "
    "RuntimeError(\"FeetechMotorsBus(\\n Port: '/dev/cu.usbmodem5AB01818061',\\n Motors: \\n"
    + "{'shoulder_pan': Motor(id=1, model='sts3215'), " * 12
    + "}\\n)',\\n has no calibration registered.\")"
)
ODD = (
    "10:00:00 WARNING:strands_robots.mesh.core:[mesh] so101: state probe 'hw_joints' failed, that "
    "section of the snapshot is omitted (further failures logged at debug): OSError(5, 'Input/output error')"
)


def test_a_contended_port_is_named_as_such():
    v = joint_silence.classify(["hardware connected", IN_USE])
    assert v["kind"] == "port_in_use"
    assert "holding this arm's serial port" in v["headline"]
    assert "lsof" in v["remedy"] and "recalibrat" in v["remedy"]  # says what NOT to try


def test_an_uncalibrated_board_is_not_reported_as_contention():
    v = joint_silence.classify([UNCALIBRATED])
    assert v["kind"] == "uncalibrated"
    assert "calibrat" in v["remedy"].lower()
    assert "lsof" not in v["remedy"], "opposite remedy -- the two must never be conflated"


def test_the_motor_dump_is_trimmed_to_something_readable():
    v = joint_silence.classify([UNCALIBRATED])
    assert len(v["detail"]) <= 240, "a multi-page motor dump is not a badge"
    assert v["detail"].startswith("RuntimeError"), "the exception type must survive the trim"
    assert "\n" not in v["detail"]


def test_an_unrecognised_failure_still_points_at_the_log():
    v = joint_silence.classify([ODD])
    assert v["kind"] == "probe_failed"
    assert "log" in v["remedy"]
    assert "Input/output error" in v["detail"], "never hide the raw reason"


def test_the_newest_complaint_wins():
    v = joint_silence.classify([UNCALIBRATED, IN_USE])
    assert v["kind"] == "port_in_use"


def test_silence_stays_silent():
    assert joint_silence.classify(["hardware connected", "so101 online"]) is None
    assert joint_silence.classify(None) is None
    assert joint_silence.classify([]) is None


def test_live_joints_beat_a_past_complaint():
    # mesh.core logs a degraded probe once and NEVER logs the recovery, so the line outlives the
    # fault. Without this gate the badge would be permanent -- and then ignored.
    peer = {"state": {"shoulder_pan.pos": 12.0, "gripper.pos": 3.0}}
    fields = {"role": "follower", "joint_problem": {"kind": "port_in_use"}}
    out = joint_silence.merge(peer, fields)
    assert "joint_problem" not in out
    assert out["role"] == "follower", "the gate must not touch anything else"


def test_the_complaint_survives_while_joints_are_missing():
    peer = {"state": {"peer_id": "so101", "t": 1.0}}      # presence, no positions
    fields = {"joint_problem": {"kind": "uncalibrated"}}
    assert joint_silence.merge(peer, fields)["joint_problem"]["kind"] == "uncalibrated"


@pytest.mark.parametrize("state", [None, {}, "nonsense", {"cameras": {"top": {}}}])
def test_no_joints_means_no_joints(state):
    assert joint_silence.has_joints(state) is False


# --------------------------------------------------------------- wiring (Q80)
# The U2 lesson: a claim about the fleet view must be proven where the view reads from, not on the
# route that happens to be easiest to call. Both halves are checked here -- the annotation the
# DeviceManager contributes, and the gate MeshBridge applies to it.

def test_the_device_manager_contributes_the_verdict_from_the_childs_own_log():
    from strands_robots.dashboard.device_manager import DeviceManager, ManagedRobot

    dm = DeviceManager.__new__(DeviceManager)          # no ports, no processes, no threads
    dm.robots = {"so101-follower": ManagedRobot(
        peer_id="so101-follower", robot_name="so101", mode="real",
        port="/dev/cu.usbmodem5AB01584281", cameras={}, process=None, started_at=0.0,
    )}
    dm.robots["so101-follower"].logs.extend(["hardware connected", IN_USE])
    dm.roles_by_peer = lambda: {}

    ann = dm.annotations_by_peer()
    assert ann["so101-follower"]["joint_problem"]["kind"] == "port_in_use"


def test_the_bridge_snapshot_carries_it_and_the_gate_clears_it():
    from strands_robots.dashboard.mesh_bridge import MeshBridge

    br = MeshBridge.__new__(MeshBridge)
    br.peer_annotations = lambda: {"so101": {"joint_problem": {"kind": "port_in_use"}}}

    silent = {"so101": {"state": {"peer_id": "so101", "t": 1.0}}}
    assert MeshBridge._peer_annotations(br)["so101"]["joint_problem"]
    got = {**silent["so101"], **joint_silence.merge(silent["so101"],
                                                    br.peer_annotations()["so101"])}
    assert got["joint_problem"]["kind"] == "port_in_use"

    reading = {"state": {"shoulder_pan.pos": 1.0}}
    assert "joint_problem" not in joint_silence.merge(reading, br.peer_annotations()["so101"])


# --- the peer's OWN report is better evidence than this module's log reading (Q85/Q86) -----------


def test_has_joints_reads_the_nested_shape_the_fleet_actually_publishes() -> None:
    # mesh.core publishes state["joints"], and JointStrip renders state.joints. Scanning the TOP
    # level for '*.pos' (the old behaviour) answered False for every healthy arm, so merge() could
    # never clear a badge -- a permanent complaint is the one failure mode this module forbids.
    assert joint_silence.has_joints({"joints": {"shoulder_pan.pos": 12.0}}) is True
    assert joint_silence.has_joints({"joints": {}}) is False
    assert joint_silence.has_joints({"peer_id": "x"}) is False


def test_has_joints_still_tolerates_a_flat_shape() -> None:
    assert joint_silence.has_joints({"shoulder_pan.pos": 1.0}) is True


def test_live_joints_clear_a_log_derived_badge() -> None:
    out = joint_silence.merge(
        {"state": {"joints": {"shoulder_pan.pos": 3.0}}},
        {"joint_problem": {"kind": "port_in_use"}},
    )
    assert "joint_problem" not in out


def test_a_peer_reported_fault_carries_the_matching_remedy_and_its_duration() -> None:
    got = joint_silence.classify_state(
        {"degraded": {"hw_joints": {"reason": "RuntimeError: has no calibration registered.",
                                    "failures": 900, "since": 1.0, "for_seconds": 12600.0}}}
    )
    assert got is not None
    assert got["kind"] == "uncalibrated"
    assert "Calibrate this arm" in got["remedy"]  # same table as the log path, never a second one
    assert got["for_seconds"] == 12600.0 and got["failures"] == 900
    assert got["source"] == "peer"


def test_a_peer_reported_port_conflict_says_find_the_other_owner() -> None:
    got = joint_silence.classify_state(
        {"degraded": {"hw_joints": {"reason": "ConnectionError: Failed to sync read "
                                              "'Present_Position' ... [TxRxResult] Port is in use!"}}}
    )
    assert got["kind"] == "port_in_use" and "lsof" in got["remedy"]


def test_an_unrecognised_reason_still_points_at_the_log() -> None:
    got = joint_silence.classify_state({"degraded": {"hw_joints": {"reason": "OSError: something new"}}})
    assert got["kind"] == "probe_failed" and "devices > logs" in got["remedy"]


def test_silence_and_junk_stay_silent() -> None:
    for state in (None, {}, {"degraded": {}}, {"degraded": {"hw_joints": {}}},
                  {"degraded": {"hw_joints": {"reason": "   "}}}, {"degraded": "yes"}, "nope"):
        assert joint_silence.classify_state(state) is None


def test_the_peers_own_report_beats_a_stale_log_verdict() -> None:
    # The log keeps the FIRST fault forever (mesh.core never logs a recovery). If the peer now says
    # the fault is a different one, the operator must be told the current one.
    out = joint_silence.merge(
        {"state": {"degraded": {"hw_joints": {"reason": "RuntimeError: has no calibration registered."}}}},
        {"joint_problem": {"kind": "port_in_use", "headline": "from an old log line"}},
    )
    assert out["joint_problem"]["kind"] == "uncalibrated"
    assert out["joint_problem"]["source"] == "peer"


def test_a_child_older_than_the_degraded_field_keeps_its_log_verdict() -> None:
    out = joint_silence.merge({"state": {"task": {"status": "idle"}}},
                              {"joint_problem": {"kind": "uncalibrated"}})
    assert out["joint_problem"] == {"kind": "uncalibrated"}
