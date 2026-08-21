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
    # Wording covers both measured cases: another process (179 orphaned children, 2026-08-19) and
    # the owning child's own aborted read (single holder, measured 2026-08-20).
    assert "serial port is held" in v["headline"]
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


def test_a_never_ran_probe_is_not_told_to_read_the_log() -> None:
    # The log is EMPTY in this case -- nothing failed. Sending the operator there is the one wrong
    # answer, and it is what the generic fallback would have said.
    got = joint_silence.classify_state({"degraded": {"hw_joints": {
        "reason": "the joint probe did not run: this peer's hardware object (SO101Follower) has no "
                  "get_observation(), so positions cannot be read", "failures": 4}}})
    assert got["kind"] == "not_probed"
    assert "nothing in the log to find" in got["remedy"]
    assert "devices > logs" not in got["remedy"]


def test_a_recovery_line_ends_the_complaint() -> None:
    # Newest-first: the last word on this probe is "it works", so there is nothing to report. Before
    # mesh/core logged recoveries this was unknowable, which is what the old tooltip admitted.
    lines = [
        "13:58 state probe 'hw_joints' failed: RuntimeError(no calibration registered)",
        "14:02 INFO:...:[mesh] so101-leader: state probe 'hw_joints' recovered after 2 failures over 41.0s",
    ]
    assert joint_silence.classify(lines) is None


def test_a_failure_AFTER_a_recovery_is_still_reported() -> None:
    # The fault came back. Order is what matters, not the mere presence of a recovery line.
    lines = [
        "13:58 state probe 'hw_joints' failed: RuntimeError(no calibration registered)",
        "14:02 [mesh] so101-leader: state probe 'hw_joints' recovered after 2 failures over 41.0s",
        "14:30 state probe 'hw_joints' failed: ConnectionError(... Port is in use!)",
    ]
    got = joint_silence.classify(lines)
    assert got is not None and got["kind"] == "port_in_use"


def test_a_recovery_line_is_never_read_as_a_failure() -> None:
    # It says "state probe 'hw_joints'" like a failure does; the failure pattern requires
    # failed/still failing, and this pins that so a future wording change cannot invert the meaning.
    assert joint_silence.classify(["14:02 state probe 'hw_joints' recovered after 1 failures over 3.0s"]) is None


def test_port_in_use_does_not_send_the_operator_hunting_a_process_that_does_not_exist() -> None:
    """MEASURED 2026-08-20 on cagatay's rig, and it contradicted this module's own advice.

    Both real arms had published ZERO joints for 5.5 hours while presence, heartbeat and cameras all
    looked healthy. The child's log carried exactly this line, and ``classify`` named it correctly::

        state probe 'hw_joints' failed ... ConnectionError("Failed to sync read 'Present_Position'
        on ids=[1, 2, 3, 4, 5, 6] after 3 tries. [TxRxResult] Port is in use!")

    But ``/usr/sbin/lsof`` on both ttys showed exactly ONE holder each - the arm's own child process.
    The old remedy said "find the other owner and stop it", so the operator would go looking for a
    second process that does not exist, find nothing, and be left with a silent arm and no next step.
    The port is busy INSIDE the owning process (a read that died mid-exchange leaves the Feetech bus
    marked in-use, and nothing in this codebase clears that flag), so the cure is respawning that same
    child. Both cases stay possible, so the remedy now tells the operator how to TELL THEM APART.
    """
    verdict = joint_silence.classify([
        "13:58:52 WARNING:strands_robots.mesh.core:[mesh] so101-follower: state probe 'hw_joints' "
        "failed, that section of the snapshot is omitted (further failures logged at debug): "
        "ConnectionError(\"Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] after 3 "
        "tries. [TxRxResult] Port is in use!\")",
    ])
    assert verdict is not None, "the real measured line must still be classified"
    assert verdict["kind"] == "port_in_use"
    remedy = verdict["remedy"]
    assert "lsof" in remedy, "the operator needs the command that decides which case this is"
    assert "ONE holder" in remedy and "TWO holders" in remedy, (
        "a single-holder port is the case measured on real hardware; a remedy that assumes a second "
        "process sends the operator hunting something that does not exist"
    )
    assert "respawn" in remedy.lower(), "the single-holder cure is a respawn of this same child"


# ---------------------------------------------------------------------------
# Q81 aftermath: the port-busy verdict must describe the world it now lives in
# ---------------------------------------------------------------------------

_PROBE_BUSY = (
    "WARNING:strands_robots.mesh.core:state probe 'hw_joints' failed: "
    "ConnectionError: Failed to sync read 'Present_Position' after 3 tries. [TxRxResult] Port is in use!"
)
_CURE_RAN = (
    "WARNING:strands_robots.bus_access:so101-arm-1: the motor bus was left marked in-use by an "
    "exchange that never finished; cleared that flag while holding this arm's bus lock and read again "
    "(2 time(s) this session)."
)


def test_port_busy_no_longer_claims_nothing_clears_the_flag() -> None:
    """The old remedy said "nothing clears it", which stopped being true at commit 665c4617.

    Advice that describes a world the code left behind sends the operator to respawn an arm that would
    have healed on its own - and worse, teaches them that respawning is what fixed it.
    """
    verdict = joint_silence.classify([_PROBE_BUSY])
    assert verdict is not None and verdict["kind"] == "port_in_use"
    remedy = verdict["remedy"]
    assert "nothing clears it" not in remedy
    assert "CLEARS that by itself" in remedy, "say that the flag heals itself now"
    assert "older build" in remedy, "an arm that stays silent may simply predate the cure"


def test_a_log_showing_the_cure_RAN_retires_the_stale_flag_explanation() -> None:
    """Evidence in the same log changes the diagnosis, which is the whole point of reading it.

    If the child has already cleared a stranded flag, the flag is spent as an explanation: the
    recovery ran, and the port was busy again immediately. What remains is a real second owner or a
    bus that stopped answering - a completely different set of actions.
    """
    verdict = joint_silence.classify([_CURE_RAN, _PROBE_BUSY])
    assert verdict is not None and verdict["kind"] == "port_in_use"
    remedy = verdict["remedy"]
    assert "no longer the explanation" in remedy
    assert "REAL second owner" in remedy
    assert "lsof" in remedy, "still name the command that identifies a holder"
    assert "failing hardware, not bad luck" in remedy
    assert "Respawning masks it" in remedy, "do not recommend the action that hides the fault"
    assert "older build" not in remedy, "this child demonstrably runs the curing code"


def test_the_cure_line_only_sharpens_the_port_busy_verdict() -> None:
    """A recovery in the log must not rewrite an unrelated diagnosis.

    The two events are independent: an arm can heal a stranded flag at 12:00 and lose its calibration
    at 12:05, and the calibration answer is still the right one.
    """
    uncal = joint_silence.classify(
        [_CURE_RAN, "state probe 'hw_joints' failed: no calibration registered for so101"]
    )
    assert uncal is not None and uncal["kind"] == "uncalibrated"
    assert "Calibrate this arm" in uncal["remedy"]


def test_a_healthy_log_with_a_recovery_still_reports_nothing() -> None:
    """A cured flag is not a fault: healing and then working is the success case, badge-free."""
    assert joint_silence.classify([_PROBE_BUSY, _CURE_RAN, "state probe 'hw_joints' recovered"]) is None


def _busy_state(**extra):
    return {
        "peer_id": "so101-arm-1",
        "degraded": {"hw_joints": {"reason": "[TxRxResult] Port is in use!", "failures": 40,
                                   "for_seconds": 900}},
        **extra,
    }


def test_the_snapshot_path_sharpens_from_the_published_count_not_just_the_log() -> None:
    """Both sources must answer "what do I do about this arm" the same way.

    The log path learned to retire the stale-flag explanation when it sees the cure's fingerprint. The
    snapshot path has its own fingerprint - the peer publishes how many stranded flags it has cleared -
    and it is the ONLY path external peers have, so without this they would get the weaker answer
    exactly where there is no log to correct it.
    """
    weak = joint_silence.classify_state(_busy_state())
    assert weak is not None and "CLEARS that by itself" in weak["remedy"], "no evidence: stay cautious"

    sharp = joint_silence.classify_state(_busy_state(bus_recoveries=3))
    assert sharp is not None and sharp["kind"] == "port_in_use"
    assert "no longer the explanation" in sharp["remedy"]
    assert "REAL second owner" in sharp["remedy"]
    assert sharp["failures"] == 40 and sharp["for_seconds"] == 900, "keep the peer's own numbers"
    assert sharp["source"] == "peer"


def test_a_published_count_of_zero_or_nonsense_is_no_evidence_at_all() -> None:
    """Absent, zero, negative, a string and a bool all mean "no evidence" - never a sharpened claim.

    Absent covers both an older build and a peer that simply never stranded a flag; those are
    indistinguishable from here and both correctly leave the cautious verdict in place. Sharpening a
    diagnosis on an invented count would be worse than the staleness this replaced.
    """
    for value in (0, -4, "7", True, None, [1]):
        verdict = joint_silence.classify_state(_busy_state(bus_recoveries=value))
        assert verdict is not None
        assert "CLEARS that by itself" in verdict["remedy"], f"{value!r} must not sharpen the verdict"


def test_the_published_count_does_not_touch_other_faults() -> None:
    """A cleared flag says nothing about calibration - the remedy for that must not move."""
    state = {
        "degraded": {"hw_joints": {"reason": "no calibration registered for so101"}},
        "bus_recoveries": 9,
    }
    verdict = joint_silence.classify_state(state)
    assert verdict is not None and verdict["kind"] == "uncalibrated"
    assert "Calibrate this arm" in verdict["remedy"]


# --- Q113: a remembered verdict, because the log window is 10 lines long -------------------------
# The reason a peer publishes no joints is derived from ManagedRobot.logs, a deque(maxlen=10). A
# child that prints ANYTHING after its hw_joints failure pushes the explanation out of that window
# while the fault is unchanged, and the card silently returns to the healthy-looking-arm state Q80
# exists to end. So the verdict is remembered on the record, and only two things may clear it: the
# arm publishing joints again (joint_silence.merge, already tested above) or the child's own
# recovery line. `recovered` is the public half of that second rail.
def test_recovered_is_the_only_silence_that_may_clear_a_remembered_verdict() -> None:
    healed = [_PROBE_BUSY, "INFO:strands_robots.mesh.core:state probe 'hw_joints' recovered"]
    assert joint_silence.recovered(healed) is True
    # THE MISTAKE THIS TEST CAUGHT: the first version aliased `recovered` to `_self_healed`, the
    # stale-flag CURE line. That line only sharpens the busy verdict - the flag is cleared, so a
    # real owner or a dead bus remains - and clearing a badge on it would hide a live fault.
    assert joint_silence.recovered([_PROBE_BUSY, _CURE_RAN]) is False
    assert joint_silence._self_healed([_CURE_RAN]) is True, "the cure line is still recognised, as itself"
    # The failure scrolling away is NOT recovery: an empty window is no evidence of health.
    assert joint_silence.recovered([]) is False
    assert joint_silence.recovered(["hardware connected", "so101-leader (real @ /dev/x) online"]) is False
    # and it must agree with what classify itself honours: same lines, no complaint.
    assert joint_silence.classify(healed, {}) is None


def test_recovered_tolerates_rubbish_lines_like_the_rest_of_the_module() -> None:
    """A log ring can hold anything a child printed; a diagnostic must not raise on it."""
    for junk in ([None], [123], [{"a": 1}], [b"bytes"], []):
        assert joint_silence.recovered(junk) is False


def test_the_verdict_survives_its_own_log_line_scrolling_out_of_the_window() -> None:
    """The Q113 defect, at the seam where the fleet view actually reads.

    `ManagedRobot.logs` is a deque(maxlen=LOG_TAIL_LINES) = 200 lines. The two real arms on this
    fleet happen to keep their hw_joints failure visible only because they printed 10 lines in 49
    hours; a chatty child (a record run logging per episode, a retry loop) pushes the explanation
    out of a 200-line window without difficulty. MEASURED, so the premise is not assumed: the
    assertion below fails if the deque still holds the line. Before this, the badge
    then vanished and the card was back to a connected arm with an empty joint history and no
    reason -- indistinguishable from a slow probe, which is the exact confusion Q80 removed.
    """
    from strands_robots.dashboard.device_manager import DeviceManager, ManagedRobot

    dm = DeviceManager.__new__(DeviceManager)
    dm.robots = {"arm": ManagedRobot(
        peer_id="arm", robot_name="so101", mode="real",
        port="/dev/cu.usbmodem1", cameras={}, process=None, started_at=0.0,
    )}
    dm.roles_by_peer = lambda: {}
    arm = dm.robots["arm"]

    arm.logs.extend(["hardware connected", IN_USE])
    assert dm.annotations_by_peer()["arm"]["joint_problem"]["kind"] == "port_in_use"

    # Enough ordinary lines to fill the window: the failure is now GONE from it, the fault is not.
    arm.logs.extend(f"episode {i} recorded" for i in range(210))
    assert IN_USE not in list(arm.logs), "the premise: the log no longer contains the reason"
    still = dm.annotations_by_peer()["arm"]["joint_problem"]
    assert still["kind"] == "port_in_use", "a scrolled-away reason must not retire the badge"

    # ...and the child's OWN recovery line does retire it.
    arm.logs.append("INFO:strands_robots.mesh.core:state probe 'hw_joints' recovered")
    assert "joint_problem" not in dm.annotations_by_peer().get("arm", {})


def test_a_respawned_arm_does_not_inherit_the_dead_processs_complaint() -> None:
    """Why the memory lives on the RECORD and not in a dict keyed by peer id.

    A respawn under the same peer id is a NEW ManagedRobot. If the verdict were remembered by peer
    id, the fresh process would be born wearing the old one's badge -- and the operator who just
    fixed the port would be told it is still contended, which teaches them to ignore badges.
    """
    from strands_robots.dashboard.device_manager import DeviceManager, ManagedRobot

    dm = DeviceManager.__new__(DeviceManager)
    dm.roles_by_peer = lambda: {}
    dm.robots = {"arm": ManagedRobot(
        peer_id="arm", robot_name="so101", mode="real",
        port="/dev/cu.usbmodem1", cameras={}, process=None, started_at=0.0,
    )}
    dm.robots["arm"].logs.extend(["hardware connected", IN_USE])
    assert dm.annotations_by_peer()["arm"]["joint_problem"]["kind"] == "port_in_use"

    dm.robots["arm"] = ManagedRobot(                     # the respawn
        peer_id="arm", robot_name="so101", mode="real",
        port="/dev/cu.usbmodem1", cameras={}, process=None, started_at=1.0,
    )
    dm.robots["arm"].logs.append("hardware connected")
    assert "joint_problem" not in dm.annotations_by_peer().get("arm", {})
