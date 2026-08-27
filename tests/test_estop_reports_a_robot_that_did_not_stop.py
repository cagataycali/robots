"""An emergency stop that the robot refused is reported, not discarded.

``RobotDeviceDriver.onEmergencyStop`` called ``self._robot.stop_task()`` and
dropped the envelope. Two of the four robot surfaces the driver can wrap answer
a stop with an affirmative "I did not stop": ``G1Driver.stop_task`` reports
``status="error"`` with ``stopped=False`` when its control loop outlasts the
join budget, and ``ReachyDriver.stop_task`` reports a daemon that refused. So
an operator's emergency stop was answered by a single WARNING line saying the
stop was being attempted, while the loop kept writing frames and nothing was
recorded above WARNING.

``Mesh.emergency_stop`` grades exactly that verdict for every peer it fans out
to and logs one that did not stop at CRITICAL, quoting a hardware cutoff as the
remedy. A stop that arrives over Device Connect rather than over the mesh is
the same operator request, so it gets the same accounting.

Why the existing suites were silent: the three files that drive
``onEmergencyStop`` assert ``stop_task.assert_called_once()`` or
``robot.stopped is True`` -- that the CALL WAS MADE, never what it ANSWERED --
and their robot doubles are ``MagicMock``s whose ``stop_task`` returns a
``MagicMock``, which carries no verdict to read.
"""

import ast
import asyncio
import inspect
import logging

import pytest

# The autouse fixture below is what makes this import safe. A sibling test file
# replaces the device_connect_edge submodules with MagicMocks at import time,
# and this helper restores the real ones. Importing the helper does NOT bring
# the sibling's autouse fixture with it -- an autouse fixture is bound to the
# module that declares it -- so this file declares its own.
from tests.test_device_connect_hardening import _force_real_device_connect_edge

# The two shapes a real driver produces, stated here rather than imported so a
# failure names the disagreement instead of inheriting it. The one cell that
# reads the drivers is the premise below.
_G1_JOIN_TIMEOUT = {
    "status": "error",
    "content": [
        {
            "json": {
                "stopped": False,
                "running": True,
                "steps": 4211,
                "reason": "stop_task: control loop did not join within timeout; policy is likely blocking",
            }
        }
    ],
}
_REACHY_DAEMON_REFUSED = {
    "status": "error",
    "content": [{"text": "stop_task: daemon refused the stop: connection reset"}],
}
_CLEAN_STOP = {
    "status": "success",
    "content": [{"json": {"stopped": True, "running": False, "steps": 12}}],
}
_NOTHING_TO_STOP = {
    "status": "success",
    "content": [{"text": "No task running to stop (current: idle)"}],
}

_CUTOFF_REMEDY = "hardware cutoff"
_LOGGER = "strands_robots.device_connect.robot_driver"


@pytest.fixture(autouse=True)
def _real_device_connect(monkeypatch):
    """Restore the real device_connect_edge and clear the estop allowlist."""
    _force_real_device_connect_edge()
    for var in ("DEVICE_CONNECT_RPC_ALLOW", "DEVICE_CONNECT_ESTOP_ALLOW", "DEVICE_CONNECT_ALLOW_INSECURE"):
        monkeypatch.delenv(var, raising=False)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _Robot:
    """A robot whose ``stop_task`` answers with a chosen envelope."""

    tool_name_str = "g1"

    def __init__(self, envelope):
        self._envelope = envelope
        self.calls = 0

    def stop_task(self):
        self.calls += 1
        return self._envelope


def _estop(envelope, caplog, device_id="safety-controller-1"):
    """Deliver an emergency stop and return (robot, records)."""
    from strands_robots.device_connect.robot_driver import RobotDeviceDriver

    robot = _Robot(envelope)
    driver = RobotDeviceDriver(robot)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        _run(driver.onEmergencyStop(device_id, "emergencyStop", {"reason": "operator pressed"}))
    return robot, [r for r in caplog.records if r.name == _LOGGER]


def _above_warning(records):
    return [r for r in records if r.levelno >= logging.ERROR]


class TestTheRefusalShapesAreReal:
    """The two envelopes graded below are the ones the drivers actually build.

    Premises: they hold before and after the fix.
    """

    def test_a_refused_stop_carries_text_and_no_stopped_flag(self):
        """``_refuse`` is how both drivers report a refusal, and it carries no json block."""
        from strands_robots.drivers.g1 import _refuse as g1_refuse
        from strands_robots.drivers.reachy import _refuse as reachy_refuse

        for refuse in (g1_refuse, reachy_refuse):
            envelope = refuse("stop_task: daemon refused the stop: connection reset")
            assert envelope["status"] == "error"
            assert [b for b in envelope["content"] if "json" in b] == [], (
                "a refusal carries text only, which is why a stopped-flag reader alone misses it"
            )

    def test_the_g1_stop_reports_the_join_outcome_in_its_payload(self):
        """G1's ``stop_task`` puts the join outcome under ``stopped``."""
        from strands_robots.drivers.g1 import G1Driver

        source = inspect.getsource(G1Driver.stop_task)
        assert '"stopped"' in source, "the graded json shape requires G1 to report a stopped flag"

    def test_the_teleop_reader_alone_would_miss_a_refused_stop(self):
        """The design reason a ``stop_task`` reader is not the teleop reader.

        ``teleop_mixin._stop_reported_stopped`` answers ``True`` for an envelope
        with no ``json`` block. That is right for ``stop_teleoperate`` (nothing
        was teleoperating) and wrong here, because it is the shape a refused
        ``stop_task`` arrives in.
        """
        from strands_robots.teleop_mixin import _stop_reported_stopped

        assert _stop_reported_stopped(_G1_JOIN_TIMEOUT) is False
        assert _stop_reported_stopped(_REACHY_DAEMON_REFUSED) is True


class TestARobotThatDidNotStopIsReported:
    """The affirmative refusal reaches an operator."""

    @pytest.mark.parametrize(
        ("label", "envelope"),
        [("join-timeout", _G1_JOIN_TIMEOUT), ("daemon-refused", _REACHY_DAEMON_REFUSED)],
    )
    def test_the_refusal_is_recorded_above_warning(self, label, envelope, caplog):
        robot, records = _estop(envelope, caplog)
        assert robot.calls == 1, "the stop is still attempted"
        assert _above_warning(records), f"a {label} stop was recorded at WARNING or below"

    @pytest.mark.parametrize(
        ("label", "envelope", "quote"),
        [
            ("join-timeout", _G1_JOIN_TIMEOUT, "did not join within timeout"),
            ("daemon-refused", _REACHY_DAEMON_REFUSED, "daemon refused the stop"),
        ],
    )
    def test_the_report_names_the_reason_the_robot_gave(self, label, envelope, quote, caplog):
        _robot, records = _estop(envelope, caplog)
        loud = " ".join(r.getMessage() for r in _above_warning(records))
        assert quote in loud, f"the {label} report does not carry the robot's own reason"

    def test_the_report_names_the_source_of_the_stop(self, caplog):
        _robot, records = _estop(_G1_JOIN_TIMEOUT, caplog, device_id="safety-controller-7")
        loud = " ".join(r.getMessage() for r in _above_warning(records))
        assert "safety-controller-7" in loud

    def test_the_report_names_the_hardware_cutoff_remedy(self, caplog):
        """The mesh's own wording: a robot that may still be executing needs a cutoff."""
        _robot, records = _estop(_G1_JOIN_TIMEOUT, caplog)
        loud = " ".join(r.getMessage() for r in _above_warning(records))
        assert _CUTOFF_REMEDY in loud


class TestAStopThatHappenedIsQuiet:
    """Over-reach controls: a false "did not stop" on the safety path is worse than none."""

    @pytest.mark.parametrize(
        ("label", "envelope"),
        [("clean stop", _CLEAN_STOP), ("nothing to stop", _NOTHING_TO_STOP)],
    )
    def test_an_accounted_stop_records_nothing_above_warning(self, label, envelope, caplog):
        _robot, records = _estop(envelope, caplog)
        assert _above_warning(records) == [], f"a {label} must not be reported as a failure"

    def test_an_error_beside_a_successful_stop_is_not_flagged(self, caplog):
        """An explicit ``stopped=True`` is authoritative over the envelope's status.

        This is why the flag is read ahead of ``status`` rather than instead of
        it: a driver reporting an error about something else, while its own
        payload says the loop stopped, has not failed to stop.
        """
        envelope = {
            "status": "error",
            "content": [{"json": {"stopped": True, "running": False, "note": "zero-torque publish retried"}}],
        }
        _robot, records = _estop(envelope, caplog)
        assert _above_warning(records) == []

    def test_a_return_that_is_not_an_envelope_is_not_flagged(self, caplog):
        """A driver returning ``None`` reports nothing; it is not an affirmative failure."""
        _robot, records = _estop(None, caplog)
        assert _above_warning(records) == []

    def test_a_test_double_that_answers_nothing_is_not_flagged(self, caplog):
        """The shape the three pre-existing suites hand this handler."""
        from unittest.mock import MagicMock

        from strands_robots.device_connect.robot_driver import RobotDeviceDriver

        robot = MagicMock()
        robot.tool_name_str = "so100"
        driver = RobotDeviceDriver(robot)
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            _run(driver.onEmergencyStop("safety-1", "emergencyStop", {}))
        robot.stop_task.assert_called_once()
        assert _above_warning([r for r in caplog.records if r.name == _LOGGER]) == []


class TestTheAuthorizationGuardStillComesFirst:
    """Reading the verdict must not widen who can trigger a stop."""

    def test_an_unauthorized_source_never_reaches_stop_task(self, caplog, monkeypatch):
        monkeypatch.setenv("DEVICE_CONNECT_ESTOP_ALLOW", "safety-controller-1")
        robot, records = _estop(_G1_JOIN_TIMEOUT, caplog, device_id="rogue-device")
        assert robot.calls == 0, "a spoofed emergency stop must not halt the task"
        assert _above_warning(records) == [], "a rejected source is not a robot that did not stop"

    def test_an_authorized_source_still_reaches_stop_task(self, caplog, monkeypatch):
        monkeypatch.setenv("DEVICE_CONNECT_ESTOP_ALLOW", "safety-controller-1")
        robot, _records = _estop(_G1_JOIN_TIMEOUT, caplog)
        assert robot.calls == 1


class TestTheHandlerReadsTheVerdictRatherThanDiscardingIt:
    """Structural: the call's value is consumed, and the mesh sibling agrees."""

    def test_the_stop_call_is_not_a_bare_statement(self):
        from strands_robots.device_connect import robot_driver

        tree = ast.parse(inspect.getsource(robot_driver))
        handler = next(
            node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "onEmergencyStop"
        )
        discarded = [
            ast.unparse(node.value)
            for node in ast.walk(handler)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and "stop_task" in ast.unparse(node.value)
        ]
        assert discarded == [], f"the stop's verdict is discarded: {discarded}"

    def test_the_mesh_estop_path_flags_the_same_envelope(self):
        """The in-tree control: one fleet, one envelope, one verdict.

        ``Mesh.emergency_stop`` already reports a peer that did not stop. If the
        two paths disagreed about the same envelope, an operator's accounting
        would depend on which transport the stop arrived over.
        """
        from strands_robots.mesh.core import _peers_that_did_not_stop

        flagged = _peers_that_did_not_stop([{**_G1_JOIN_TIMEOUT, "responder_id": "g1-01"}])
        assert flagged == {"g1-01"}
        assert _peers_that_did_not_stop([{**_CLEAN_STOP, "responder_id": "g1-01"}]) == set()
