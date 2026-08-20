"""A failed state probe must reach the SNAPSHOT, not just the log (Q85/Q86).

Two live SO-101s sat for 3.5 hours as "connected" peers with no joints: one had no calibration
registered, the other was losing its own bus ("Port is in use!"). Both reasons were logged once and
then suppressed at debug, so the fleet view showed two healthy-looking idle arms and the operator had
no way to tell them apart -- or from an arm that genuinely was not moving.

Every Mesh here is built with ``__new__``: constructing a real one opens a session, and a test that
touches the process-global session is how a pytest run once e-stopped the live fleet (Q30).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.mesh.core import PROBE_REASON_MAX, Mesh, degraded_report, summarise_probe_error

LEADER_REPR = (
    "FeetechMotorsBus(\n    Port: '/dev/cu.usbmodem5AB01818061',\n    Motors: \n"
    "{       'shoulder_pan': Motor(id=1,\n                              model='sts3215'),\n"
    "        'gripper': Motor(id=6,\n                         model='sts3215')},\n)',\n"
    " has no calibration registered."
)
FOLLOWER_MSG = (
    "Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] after 3 tries. "
    "[TxRxResult] Port is in use!"
)


def _mesh(robot: Any) -> Mesh:
    m = Mesh.__new__(Mesh)
    m.peer_id = "so101-leader"
    m.robot = robot
    return m


class _Arm:
    """The shape _read_state probes: an outer robot wrapping a connected inner bus."""

    def __init__(self) -> None:
        self.robot = self
        self.is_connected = True
        self.config = type("C", (), {"cameras": {}})()

    def get_observation(self) -> dict[str, Any]:  # pragma: no cover - patched per test
        return {}


# --- the summariser: publishable, human, and it names the action ---------------------------------


def test_the_leaders_1_5kb_motor_dump_becomes_one_sentence() -> None:
    out = summarise_probe_error(RuntimeError(LEADER_REPR))
    assert out == "RuntimeError: has no calibration registered."
    assert len(out) <= PROBE_REASON_MAX


def test_the_type_survives_because_it_decides_what_the_operator_does() -> None:
    # Q85 (calibrate this arm) and Q86 (something owns the bus) need different actions, and the
    # message alone does not say which family it belongs to.
    assert summarise_probe_error(ConnectionError(FOLLOWER_MSG)).startswith("ConnectionError:")
    assert "Port is in use!" in summarise_probe_error(ConnectionError(FOLLOWER_MSG))


def test_an_exception_with_no_message_still_says_something() -> None:
    assert summarise_probe_error(TimeoutError()) == "TimeoutError"


def test_a_very_long_single_line_is_capped_not_dropped() -> None:
    out = summarise_probe_error(RuntimeError("x" * 900))
    assert len(out) <= PROBE_REASON_MAX and out.endswith("\u2026")


def test_the_summariser_never_raises() -> None:
    class Hostile(Exception):
        def __str__(self) -> str:
            raise ValueError("nope")

    assert summarise_probe_error(Hostile()) == "Hostile"


# --- the report: how long, how often, and JSON-safe ----------------------------------------------


def test_for_seconds_is_computed_because_a_consumer_shares_no_clock() -> None:
    rep = degraded_report({"hw_joints": {"reason": "R", "failures": 7, "since": 1000.0}}, 1030.5)
    assert rep["hw_joints"]["for_seconds"] == 30.5
    assert rep["hw_joints"]["failures"] == 7


def test_a_clock_that_went_backwards_never_reports_negative_age() -> None:
    rep = degraded_report({"hw_joints": {"reason": "R", "since": 2000.0}}, 1000.0)
    assert rep["hw_joints"]["for_seconds"] == 0.0


# --- the wiring: a failing probe publishes, a working one clears ----------------------------------


def test_the_snapshot_carries_the_reason_the_joints_are_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _mesh(_Arm())
    monkeypatch.setattr(mesh_core, "read_joints", lambda inner: (_ for _ in ()).throw(RuntimeError(LEADER_REPR)))
    snap = m._read_state()
    assert snap is not None, "a snapshot whose only content is a degradation must still be sent"
    assert "joints" not in snap
    assert snap["degraded"]["hw_joints"]["reason"] == "RuntimeError: has no calibration registered."
    assert snap["degraded"]["hw_joints"]["failures"] == 1


def test_repeated_failure_counts_up_and_keeps_the_first_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _mesh(_Arm())
    monkeypatch.setattr(mesh_core, "read_joints", lambda inner: (_ for _ in ()).throw(ConnectionError(FOLLOWER_MSG)))
    first = m._read_state()
    time.sleep(0.01)
    later = m._read_state()
    assert later["degraded"]["hw_joints"]["failures"] == 2
    assert later["degraded"]["hw_joints"]["since"] == first["degraded"]["hw_joints"]["since"]


def test_recovery_removes_the_badge_because_a_false_one_costs_belief(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _mesh(_Arm())
    monkeypatch.setattr(mesh_core, "read_joints", lambda inner: (_ for _ in ()).throw(ConnectionError(FOLLOWER_MSG)))
    assert "degraded" in m._read_state()
    monkeypatch.setattr(mesh_core, "read_joints", lambda inner: {"shoulder_pan": 12.0})
    good = m._read_state()
    assert good["joints"] == {"shoulder_pan": 12.0}
    assert "degraded" not in good, "one transient error must not libel an arm until it is restarted"


def test_a_fault_that_returns_is_warned_about_again(monkeypatch: pytest.MonkeyPatch) -> None:
    # The once-only log suppression is keyed per category; recovery must reset it, or a fault that
    # comes back an hour later hides behind an occurrence nobody can still see.
    m = _mesh(_Arm())
    boom = lambda inner: (_ for _ in ()).throw(ConnectionError(FOLLOWER_MSG))  # noqa: E731
    monkeypatch.setattr(mesh_core, "read_joints", boom)
    m._read_state()
    monkeypatch.setattr(mesh_core, "read_joints", lambda inner: {"shoulder_pan": 1.0})
    m._read_state()
    assert "hw_joints" not in getattr(m, "_read_state_warned", set())
    monkeypatch.setattr(mesh_core, "read_joints", boom)
    m._read_state()
    assert m._read_state_degraded["hw_joints"]["failures"] == 1, "the second episode counts from 1"


def test_a_new_cause_replaces_the_old_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _mesh(_Arm())
    monkeypatch.setattr(mesh_core, "read_joints", lambda inner: (_ for _ in ()).throw(ConnectionError(FOLLOWER_MSG)))
    m._read_state()
    monkeypatch.setattr(mesh_core, "read_joints", lambda inner: (_ for _ in ()).throw(RuntimeError(LEADER_REPR)))
    snap = m._read_state()
    assert snap["degraded"]["hw_joints"]["reason"].startswith("RuntimeError:")


# --- silence with NO error: the state both of cagatay's arms were actually in --------------------


class _NoObservation:
    """A hardware object that is connected but cannot be asked for positions."""

    def __init__(self) -> None:
        self.robot = self
        self.is_connected = True
        self.config = type("C", (), {"cameras": {}})()


class _Disconnected(_Arm):
    def __init__(self) -> None:
        super().__init__()
        self.is_connected = False


def test_a_probe_that_never_ran_says_which_precondition_stopped_it() -> None:
    # No exception means no log line, nothing for the log-reading diagnosis to find, and no degraded
    # entry -- so a peer can report connected:true with no joints forever and no rail can say why.
    # (I originally justified this test with cagatay's two silent arms; that was wrong, their probes
    # DO throw and I had searched the wrong log. The hole is real, the anecdote was not.)
    snap = _mesh(_NoObservation())._read_state()
    reason = snap["degraded"]["hw_joints"]["reason"]
    assert "did not run" in reason and "get_observation" in reason
    assert "_NoObservation" in reason, "name the object, so the operator can see what was spawned"
    assert snap["degraded"]["hw_joints"]["skipped"] is True


def test_a_disconnected_hardware_object_is_named_as_such() -> None:
    snap = _mesh(_Disconnected())._read_state()
    assert "is_connected false" in snap["degraded"]["hw_joints"]["reason"]


def test_an_observation_with_no_scalar_joint_counts_what_came_back(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _mesh(_Arm())
    monkeypatch.setattr(mesh_core, "read_joints", lambda inner: {})
    reason = m._read_state()["degraded"]["hw_joints"]["reason"]
    assert "0 keys came back" in reason, "'the arm answered with nothing' is not 'we never asked'"


def test_a_peer_with_no_hardware_object_stays_silent() -> None:
    # A sim world or a gateway is not a broken arm; complaining would give every such peer a
    # permanent badge for not being hardware.
    class Bare:
        pass

    assert _mesh(Bare())._read_state() is None


def test_a_skip_that_resolves_clears_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    arm = _Arm()
    m = _mesh(arm)
    monkeypatch.setattr(mesh_core, "read_joints", lambda inner: {})
    assert "degraded" in m._read_state()
    monkeypatch.setattr(mesh_core, "read_joints", lambda inner: {"shoulder_pan.pos": 4.0})
    assert "degraded" not in m._read_state()


def test_a_thrown_failure_carries_no_skipped_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # Absent, not False: a consumer must not be told something the publisher never said.
    m = _mesh(_Arm())
    monkeypatch.setattr(mesh_core, "read_joints", lambda inner: (_ for _ in ()).throw(RuntimeError(LEADER_REPR)))
    assert "skipped" not in m._read_state()["degraded"]["hw_joints"]


def test_a_recovered_probe_says_so_exactly_once(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    # The log's last word on a healed arm used to be "failed", which makes every past fault look
    # permanent to a human AND to this dashboard's own log-derived diagnosis.
    arm = _Arm()
    m = _mesh(arm)
    monkeypatch.setattr(mesh_core, "read_joints", lambda inner: (_ for _ in ()).throw(RuntimeError(LEADER_REPR)))
    m._read_state()
    m._read_state()
    monkeypatch.setattr(mesh_core, "read_joints", lambda inner: {"shoulder_pan.pos": 4.0})
    with caplog.at_level(logging.INFO, logger="strands_robots.mesh.core"):
        m._read_state()
        recovered = [r for r in caplog.records if "recovered" in r.message or "recovered" in r.getMessage()]
        assert len(recovered) == 1, "exactly one line, naming the category"
        assert "hw_joints" in recovered[0].getMessage()
        assert "2 failures" in recovered[0].getMessage(), "how bad it was, and for how long"
        caplog.clear()
        # A healthy cycle is not news: logging a recovery every second would bury the failures it
        # exists to qualify.
        m._read_state()
        m._read_state()
        assert [r for r in caplog.records if "recovered" in r.getMessage()] == []


def test_a_degraded_entry_never_creates_a_snapshot_on_its_own() -> None:
    # Contract older than this feature (tests/mesh/test_mesh.py: a hostile robot yields None, never
    # an exception). If EVERY probe failed, the peer's problem is not its joints -- there is no
    # working robot behind it, and presence already carries connected/hw. My first version published
    # a state whose only content was a complaint, which would also give every gateway a permanent
    # broadcast; it regressed that test for three iterations because my -k filter never selected
    # tests/mesh/.
    class Hostile:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("inner robot unavailable")

    assert _mesh(Hostile())._read_state() is None
