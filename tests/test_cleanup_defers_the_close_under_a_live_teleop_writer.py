# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""``cleanup()`` must not close the devices under a teleop loop that is still writing.

``stop_teleoperate`` reports its join outcome honestly: a leader whose
``get_action()`` blocks past the join budget - a serial read on a wedged bus is
the ordinary case - yields ``status="error"`` with ``stopped: False``, and that
branch deliberately leaves the attached devices connected. Its docstring says
why, and names the precedent: "Tearing the bus down under a live writer is what
``G1Driver.cleanup`` refuses for the same reason."

``cleanup()`` called ``stop_teleoperate()`` and discarded that envelope, then
closed the motors bus. Measured on a one-camera arm whose leader is wedged, with
the port's exclusivity modelled:

    call                       bus closed   port held   thread alive
    stop_teleoperate() alone       no         yes           yes
    cleanup()                     yes          no           yes

So the two functions disagreed about the same state, and ``cleanup()`` undid the
protection ``stop_teleoperate`` had just provided.

Closing there is worse than deferring on both of the counts ``cleanup()`` cares
about, which is why this is a defect rather than a trade-off:

- the release does not hold. ``send_action`` re-opens the robot lazily on a
  command that finds it disconnected, and the live loop's next write goes
  through it, so the port is open again (``connect_calls`` 1 -> 2) with nothing
  left to close it - ``cleanup()`` is terminal and the executor is already down.
  That is exactly the harm the comment above ``_disconnect_devices()`` describes,
  and the teleop loop is first in its own list of command sources.
- the torque disable is undone. ``_disconnect_devices`` prefers the driver's own
  ``disconnect()`` because that is where torque disable and gripper release
  live; the loop's write then lands *after* it, leaving the arm energised at a
  fresh commanded position.

What these tests pin:

    - a live teleop writer defers the close: the bus stays connected, the port
      stays held, and no re-open cycle happens;
    - the reason is recorded at ERROR with the remedy, because ``cleanup()``
      returns ``None`` and has no envelope to report through;
    - a clean stop still closes every device exactly once, and a robot with no
      teleop session at all is untouched by the change;
    - a ``stop_teleoperate`` that *raises* still closes the devices - the
      outcome is unknown there, and the pre-existing "a teleop teardown failure
      must not block the rest of hardware cleanup" contract is preserved;
    - the outcome is read from ``stopped`` rather than from ``status``, because
      the status is derived from the session counters: a session whose every
      frame errored reports ``"error"`` after a perfectly clean join.

No serial port and no camera is opened. The device doubles are the ones
``test_hardware_cleanup_disconnects`` uses, so port exclusivity is modelled and
the consequence is asserted rather than the call.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pytest

import strands_robots.teleop_mixin as teleop_mixin
from strands_robots.teleop_mixin import _stop_reported_stopped
from tests.test_hardware_cleanup_disconnects import _arm, _Bridge, _make_robot, _Mesh, _Port

# Short enough to keep every cell fast. The budget itself is pre-existing and
# pinned by ``test_stop_teleoperate_reports_the_join_outcome``; what matters here
# is only that a join can fail.
_FAST_JOIN_S = 0.3


class _WedgedLeader:
    """A leader whose ``get_action()`` blocks, as a serial read on a stuck bus does."""

    def __init__(self, released: threading.Event) -> None:
        self.is_connected = True
        self._released = released
        self.polls = 0

    def get_action(self) -> dict[str, float]:
        self.polls += 1
        self._released.wait(30.0)
        return {"j0.pos": 0.5}

    def disconnect(self) -> None:
        self.is_connected = False


class _Session:
    """A hardware robot with a teleop loop parked in its leader's read."""

    def __init__(self, *, wedged: bool) -> None:
        self.port = _Port()
        self.log: list[str] = []
        self.driver = _arm(self.port, self.log)
        self.robot = _make_robot(self.driver)
        self.driver.connect()
        self.robot._ensure_teleop_state()
        self.released = threading.Event()
        if not wedged:
            self.released.set()
        self.leader = _WedgedLeader(self.released)
        self.robot._teleops = {"leader": type("_Att", (), {"device": self.leader})()}
        self.robot._teleop_robot_name = "test_arm"
        self.robot._teleop_running = True
        self.robot._teleop_start_mono = time.monotonic()
        self.writes: list[dict[str, Any]] = []
        entered = threading.Event()

        def loop() -> None:
            entered.set()
            while self.robot._teleop_running and not self.robot._teleop_stop_event.is_set():
                action = self.leader.get_action()
                self.writes.append(self.robot.send_action(action))
                break

        self.thread = threading.Thread(target=loop, daemon=True)
        self.robot._teleop_thread = self.thread
        self.thread.start()
        assert entered.wait(5.0), "the teleop loop never started"
        if wedged:
            time.sleep(0.05)  # let it reach the blocking read
        else:
            self.thread.join(5.0)

    def release(self) -> None:
        """Unblock the wedged leader and let the loop take its one write."""
        self.released.set()
        self.thread.join(5.0)


@pytest.fixture(autouse=True)
def _fast_join(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(teleop_mixin, "_TELEOP_JOIN_TIMEOUT_S", _FAST_JOIN_S)


class TestThePremiseStopTeleoperateAlreadyDeclinesTheTeardown:
    """What ``cleanup()`` has to read: the outcome is reported, and it is acted on."""

    def test_a_wedged_leader_makes_the_join_fail_and_says_so(self) -> None:
        session = _Session(wedged=True)
        envelope = session.robot.stop_teleoperate()
        payload = next(b["json"] for b in envelope["content"] if "json" in b)
        assert envelope["status"] == "error"
        assert payload["stopped"] is False
        assert session.thread.is_alive()
        session.release()

    def test_stop_teleoperate_leaves_the_devices_open_on_a_failed_join(self) -> None:
        session = _Session(wedged=True)
        session.robot.stop_teleoperate()
        assert session.driver.bus.is_connected
        assert session.port.held_by == "arm"
        assert session.driver.bus.disconnect_calls == []
        session.release()


class TestTheCloseIsDeferredUnderALiveWriter:
    """The regression: ``cleanup()`` reads the outcome and holds the devices."""

    def test_the_bus_is_not_closed_while_the_loop_is_alive(self) -> None:
        session = _Session(wedged=True)
        session.robot.cleanup()
        assert session.thread.is_alive(), "premise: the loop must still be running"
        assert session.driver.bus.is_connected
        assert session.driver.bus.disconnect_calls == []

    def test_the_port_stays_held_so_nothing_else_can_take_it(self) -> None:
        session = _Session(wedged=True)
        session.robot.cleanup()
        assert session.port.held_by == "arm"

    def test_no_camera_is_closed_either(self) -> None:
        session = _Session(wedged=True)
        session.robot.cleanup()
        assert all(cam.is_connected for cam in session.driver.cameras.values())
        assert session.log == []

    def test_the_drivers_own_disconnect_is_not_called(self) -> None:
        session = _Session(wedged=True)
        session.robot.cleanup()
        assert session.driver.disconnect_calls == 0

    def test_the_live_write_does_not_have_to_re_open_the_port(self) -> None:
        """The whole point: the release did not hold, so do not perform it."""
        session = _Session(wedged=True)
        session.robot.cleanup()
        opened_before = session.driver.bus.connect_calls
        session.release()
        assert session.writes and session.writes[0]["status"] == "success"
        assert session.driver.bus.connect_calls == opened_before, (
            "the loop's write had to re-open a port cleanup() closed"
        )

    def test_the_command_lands_on_a_bus_whose_torque_was_never_disabled(self) -> None:
        session = _Session(wedged=True)
        session.robot.cleanup()
        session.release()
        assert len(session.driver.commands) == 1
        assert session.driver.bus.disconnect_calls == [], "a fresh command landed after the driver disabled torque"

    def _cleanup_errors(self, caplog: pytest.LogCaptureFixture) -> list[str]:
        session = _Session(wedged=True)
        with caplog.at_level(logging.ERROR, logger="strands_robots.hardware_robot"):
            session.robot.cleanup()
        session.release()
        return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]

    def test_the_remedy_the_error_names_actually_works(self) -> None:
        """Re-join the loop, then cleanup() again: the devices close."""
        session = _Session(wedged=True)
        session.robot.cleanup()
        assert session.driver.bus.is_connected, "premise: the first cleanup deferred"
        session.release()
        session.robot.stop_teleoperate()
        session.robot.cleanup()
        assert not session.driver.bus.is_connected
        assert session.port.held_by is None

    def test_the_deferral_is_recorded_at_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """``cleanup()`` returns ``None``, so the log is the only report it has."""
        messages = self._cleanup_errors(caplog)
        assert any("devices left open" in m for m in messages), messages

    def test_the_error_names_the_remedy(self, caplog: pytest.LogCaptureFixture) -> None:
        messages = self._cleanup_errors(caplog)
        assert any("stop_teleoperate()" in m for m in messages), messages
        assert any("cleanup() again" in m for m in messages), messages


class TestDeferringTheCloseCostsNothingElse:
    """These hold either way: deferring must not skip the software teardown."""

    def test_the_shutdown_is_still_latched(self) -> None:
        session = _Session(wedged=True)
        session.robot.cleanup()
        assert session.robot._shutdown_event.is_set()
        session.release()

    def test_the_mesh_and_the_ros_bridge_are_still_torn_down(self) -> None:
        """Deferring the *device* close must not defer the software teardown.

        ``TestTheDevicesCloseLast`` in ``test_hardware_cleanup_disconnects``
        pins the mesh and the bridge going down before the devices; holding the
        devices back must not turn that ordering into a skip.
        """
        session = _Session(wedged=True)
        session.robot.mesh = _Mesh(session.log)
        session.robot._ros_bridge = _Bridge(session.log)
        session.robot.cleanup()
        assert "mesh.stop" in session.log
        assert "ros_bridge.shutdown" in session.log
        session.release()


class TestACleanStopStillClosesEverything:
    """Over-reach guard: the deferral must not cost the ordinary teardown."""

    def test_a_joined_loop_closes_the_bus_and_the_camera(self) -> None:
        session = _Session(wedged=False)
        assert not session.thread.is_alive(), "premise: this loop joins"
        session.robot.cleanup()
        assert not session.driver.bus.is_connected
        assert session.port.held_by is None
        assert session.driver.disconnect_calls == 1

    def test_a_robot_with_no_teleop_session_is_unaffected(self) -> None:
        port = _Port()
        log: list[str] = []
        driver = _arm(port, log)
        robot = _make_robot(driver)
        driver.connect()
        robot.cleanup()
        assert not driver.bus.is_connected
        assert port.held_by is None

    def test_a_clean_stop_logs_no_deferral(self, caplog: pytest.LogCaptureFixture) -> None:
        session = _Session(wedged=False)
        with caplog.at_level(logging.ERROR, logger="strands_robots.hardware_robot"):
            session.robot.cleanup()
        assert [r.getMessage() for r in caplog.records if "devices left open" in r.getMessage()] == []


class TestAnUnknownOutcomeKeepsThePreExistingContract:
    """A raise leaves the outcome unknown; warn and continue, as before."""

    def test_a_stop_that_raises_still_closes_the_devices(self) -> None:
        session = _Session(wedged=False)

        def boom() -> dict[str, Any]:
            raise RuntimeError("teleop teardown failed")

        session.robot.stop_teleoperate = boom  # type: ignore[method-assign]
        session.robot.cleanup()
        assert not session.driver.bus.is_connected
        assert session.port.held_by is None

    def test_a_stop_that_raises_is_warned_not_propagated(self, caplog: pytest.LogCaptureFixture) -> None:
        session = _Session(wedged=False)

        def boom() -> dict[str, Any]:
            raise RuntimeError("teleop teardown failed")

        session.robot.stop_teleoperate = boom  # type: ignore[method-assign]
        with caplog.at_level(logging.WARNING, logger="strands_robots.hardware_robot"):
            session.robot.cleanup()
        assert any("stop_teleoperate() raised" in r.getMessage() for r in caplog.records)


class TestTheOutcomeIsReadFromStoppedNotFromStatus:
    """``status`` is derived from the session counters, so it is the wrong key."""

    @pytest.mark.parametrize(
        ("envelope", "expected"),
        [
            pytest.param({"status": "error", "content": [{"json": {"stopped": False}}]}, False, id="failed-join"),
            pytest.param({"status": "success", "content": [{"json": {"stopped": True}}]}, True, id="clean-join"),
            pytest.param(
                {"status": "error", "content": [{"json": {"stopped": True, "frames": 4, "errors": 4}}]},
                True,
                id="every-frame-errored-but-joined-cleanly",
            ),
            pytest.param(
                {"status": "degraded", "content": [{"json": {"stopped": True}}]},
                True,
                id="degraded-but-joined-cleanly",
            ),
            pytest.param(
                {"status": "success", "content": [{"text": "No active teleoperation."}]},
                True,
                id="nothing-to-stop-carries-no-json",
            ),
            pytest.param({"status": "success", "content": []}, True, id="empty-content"),
            pytest.param({}, True, id="no-content-key"),
        ],
    )
    def test_the_reader_keys_on_the_stopped_flag(self, envelope: dict[str, Any], expected: bool) -> None:
        assert _stop_reported_stopped(envelope) is expected

    def test_an_errored_session_that_joined_still_closes_the_devices(self) -> None:
        """The end-to-end form of the same discrimination."""
        session = _Session(wedged=False)
        session.robot._teleop_errors = 7
        session.robot._teleop_frames = 7
        envelope = session.robot.stop_teleoperate()
        assert envelope["status"] == "error", "premise: the counters make this an error"
        assert _stop_reported_stopped(envelope) is True

    def test_a_json_block_without_the_flag_is_not_read_as_a_live_loop(self) -> None:
        assert _stop_reported_stopped({"content": [{"json": {"frames": 3}}]}) is True


class TestTheDecisionIsSingleSourced:
    """``cleanup()`` must not re-derive "did it stop" from the thread handle."""

    def test_cleanup_reads_the_envelope_rather_than_the_thread(self) -> None:
        import ast
        import inspect
        import textwrap

        from strands_robots.hardware_robot import Robot as HwRobot

        source = inspect.getsource(HwRobot.cleanup)
        tree = ast.parse(textwrap.dedent(source))
        calls = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and "_stop_reported_stopped" in ast.unparse(node.func)
        ]
        assert len(calls) == 1, calls
        assert "is_alive" not in source, "the join outcome is stop_teleoperate's to report"
