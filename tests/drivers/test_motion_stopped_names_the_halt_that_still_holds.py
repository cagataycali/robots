"""``motion_stopped`` names the halt that holds now, not the last one asked for.

Two shipped drivers publish a ``motion_stopped`` flag from ``get_status``, and
they publish it under the same key on purpose - the Mini's ``get_status``
docstring says the shape "matches the lerobot driver's ``get_status`` envelope so
the mesh publishes both peers identically". It is the field an operator reads to
decide whether a robot is safe to approach:
``test_a_declined_stop_hook_does_not_claim_the_motion_stopped`` in the
Microduck's suite states it in those words, and
``test_a_daemon_that_refuses_the_stop_does_not_report_it_as_halted`` in the
Mini's carries the comment "Reporting a halt that did not happen is an
affirmative lie on a safety path".

Both suites graded that flag in one direction only: it may not become true
without an accepted stop. Nothing graded the other direction, and the two
drivers had diverged there.
:meth:`~strands_robots.drivers.microduck.MicroduckDriver.send_action` clears the
latch once its intents are on the wire. The Mini set it in
:meth:`~strands_robots.drivers.reachy.ReachyDriver.stop` and
:meth:`~strands_robots.drivers.reachy.ReachyDriver.stop_task` and cleared it
only in :meth:`~strands_robots.drivers.reachy.ReachyDriver.connect_eagerly`, so
after any stop the Mini reported ``motion_stopped=True`` for the rest of the
session - through a commanded head pose, through a played emotion, through the
wake-up move - and only a reconnect made it honest again. Claiming a halt that
has since been superseded is the same lie as claiming one that never happened,
told one step later, and it was told about a robot whose head was moving.

The population below is *derived* rather than listed: every shipped native
driver whose ``get_status`` publishes ``motion_stopped`` has to clear it in
``send_action``, which is a
:data:`~strands_robots.drivers.base.DRIVER_SURFACE` member every driver has. A
third driver that starts publishing the flag is graded the moment it arrives,
without this file naming it.

What is deliberately *not* in the clearing population: the Mini's
:meth:`~strands_robots.drivers.reachy.ReachyDriver.set_motors`. Torque coming
back is not this driver committing motion, and the Microduck's ``enable_torque``
and ``relax`` do not clear the flag either - widening the rule on one driver
only would re-open the divergence this closes.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

import strands_robots.drivers.reachy as reachy_mod
from strands_robots.drivers.microduck import MicroduckDriver
from strands_robots.drivers.reachy import ReachyDriver
from strands_robots.drivers.registry import get_native_driver_class, list_native_drivers

# The Mini's own doubles: the daemon stand-in, the recording link, the connect
# helper and the agent-verb driver. Plain functions, so importing them binds no
# fixture name into this module.
from tests.drivers.test_reachy_driver import (
    _connected,
    _RecordingLink,
    _run_tool,
    _text,
)


def _flag(driver: ReachyDriver | MicroduckDriver) -> bool:
    """The ``motion_stopped`` value ``driver`` publishes right now."""
    status = asyncio.run(driver.get_status())
    published = status["content"][0]["json"]
    assert "motion_stopped" in published, "this driver does not publish the flag under test"
    return bool(published["motion_stopped"])


def _halted(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> tuple[ReachyDriver, Any, Any]:
    """A connected Mini whose daemon has accepted a stop.

    Returns:
        ``(driver, daemon, link)`` with ``motion_stopped`` established true, so
        every test below starts from the state the latch is supposed to leave.
    """
    driver, daemon, link = _connected(monkeypatch, **kwargs)
    asyncio.run(driver.stop())
    assert _flag(driver) is True, "the fixture needs an accepted stop to start from"
    return driver, daemon, link


class _RobotdDouble:
    """A live robotd client that accepts every request and notification."""

    def __init__(self) -> None:
        self.alive = True
        self.methods: list[str] = []

    def call(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.methods.append(method)
        return {"result": {}}

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.methods.append(method)

    def close(self) -> None:
        return None


#: Every Mini path that commits motion to the wire, with the call that drives it.
#: ``send_action`` is the protocol verb; the other three hand the daemon a whole
#: recorded choreography, and are exactly the paths that already forget the
#: cached head-yaw target because they move the head.
_MINI_MOTION_PATHS: tuple[tuple[str, Any], ...] = (
    ("send_action", lambda d: d.send_action({"head_yaw": 20.0})),
    ("play_move", lambda d: d.play_move("happy")),
    ("wake_up", lambda d: d.wake_up()),
    ("goto_sleep", lambda d: d.goto_sleep()),
)

#: The three daemon move players with the REST path whose verdict gates each, so
#: a double can make the daemon decline the move itself rather than the driver's
#: own argument gates refusing before the request is ever posted.
_MINI_DAEMON_MOVES: tuple[tuple[str, Any, str], ...] = (
    (
        "play_move",
        lambda d: d.play_move("happy"),
        reachy_mod._PATH_MOVE_PLAY.format(dataset=reachy_mod._MOVE_LIBRARIES["emotions"], move="happy"),
    ),
    ("wake_up", lambda d: d.wake_up(), reachy_mod._PATH_WAKE),
    ("goto_sleep", lambda d: d.goto_sleep(), reachy_mod._PATH_SLEEP),
)


class TestTheMiniClearsTheHaltWhenItCommitsMotion:
    """A halt the driver itself superseded is not still reported as holding."""

    @pytest.mark.parametrize(("verb", "call"), _MINI_MOTION_PATHS, ids=[p[0] for p in _MINI_MOTION_PATHS])
    def test_motion_after_a_stop_clears_the_flag(self, monkeypatch: pytest.MonkeyPatch, verb: str, call: Any) -> None:
        driver, _, _ = _halted(monkeypatch)
        assert call(driver)["status"] == "success", f"{verb} was expected to reach the robot here"
        assert _flag(driver) is False, f"{verb} moved the robot and the flag still reported a halt"

    def test_the_flag_an_agent_reads_is_the_one_that_clears(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole point is what the ``status`` verb publishes, not an attribute."""

        def published(driver: ReachyDriver) -> bool:
            # The ``status`` verb nests the whole ``get_status`` envelope inside
            # one json block, so the flag an agent reads is two levels down.
            outer = _run_tool(driver, "status")["content"][0]["json"]
            return bool(outer["content"][0]["json"]["motion_stopped"])

        driver, _, _ = _connected(monkeypatch)
        assert _run_tool(driver, "stop")["status"] == "success"
        assert published(driver) is True
        assert driver.play_move("happy")["status"] == "success"
        assert published(driver) is False

    def test_a_stop_task_halt_is_cleared_by_motion_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``stop_task`` is the other setter, so it needs the same clearing."""
        driver, _, _ = _connected(monkeypatch)
        assert driver.stop_task()["status"] == "success"
        assert _flag(driver) is True
        assert driver.send_action({"body_yaw": 10.0})["status"] == "success"
        assert _flag(driver) is False


class TestARefusedCommandLeavesTheHaltStanding:
    """Only a *successful* write supersedes a halt; a refusal moved nothing."""

    def test_a_link_that_refuses_the_command_does_not_clear_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Refusing(_RecordingLink):
            async def send_cmd(self, cmd: dict[str, Any]) -> None:
                raise RuntimeError("socket closed")

        driver, _, _ = _halted(monkeypatch, link=_Refusing())
        result = driver.send_action({"head_yaw": 20.0})
        assert result["status"] == "error"
        assert "socket closed" in _text(result)
        assert _flag(driver) is True, "a command the link refused was reported as superseding the halt"

    def test_a_library_the_driver_refuses_does_not_clear_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Refused by the driver's own gate, before anything is posted."""
        driver, _, _ = _halted(monkeypatch)
        result = driver.play_move("happy", library="nonesuch")
        assert result["status"] == "error"
        assert _flag(driver) is True

    @pytest.mark.parametrize(("verb", "call", "path"), _MINI_DAEMON_MOVES, ids=[m[0] for m in _MINI_DAEMON_MOVES])
    def test_a_move_the_daemon_declines_does_not_clear_it(
        self, monkeypatch: pytest.MonkeyPatch, verb: str, call: Any, path: str
    ) -> None:
        """The request was posted and came back refused: nothing moved.

        This is the half the driver's argument gates cannot stand in for - they
        refuse before the post, so they would pass a driver that cleared the
        latch the moment it decided to send.
        """
        driver, daemon, _ = _halted(monkeypatch)
        daemon._responses[path] = {"error": "another move is already playing"}
        result = call(driver)
        assert result["status"] == "error", f"{verb} was expected to report the daemon's refusal"
        assert (path, "POST") == (daemon.calls[-1][2], daemon.calls[-1][3]), "the request must have been posted"
        assert _flag(driver) is True, f"{verb} was declined and the halt was reported as superseded"

    def test_an_action_naming_nothing_sendable_does_not_clear_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _halted(monkeypatch)
        assert driver.send_action({"elbow": 1.0})["status"] == "error"
        assert _flag(driver) is True


class TestTheDirectionAlreadyGradedIsUndisturbed:
    """Recorded because they pass before and after: that is what they assert."""

    def test_motion_alone_never_reports_a_halt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch)
        assert driver.send_action({"head_yaw": 20.0})["status"] == "success"
        assert _flag(driver) is False

    def test_a_daemon_that_refuses_the_stop_still_reports_no_halt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch, stop_result={"error": "busy"})
        asyncio.run(driver.stop())
        assert _flag(driver) is False

    def test_torque_coming_back_is_not_this_driver_committing_motion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The deliberate exclusion, pinned so widening it is a decision."""
        driver, _, _ = _halted(monkeypatch)
        assert driver.set_motors("enabled")["status"] == "success"
        assert _flag(driver) is True


def _duck(client: Any) -> MicroduckDriver:
    """Build a connected Microduck driver holding ``client`` as its robotd link.

    ``client`` is annotated ``Any`` because the double is structural rather than
    a subclass of the real client - the same shape the sibling stop-verb suite
    uses to stand a robotd connection up without a socket.
    """
    driver = MicroduckDriver(port="/run/robotd-under-test.sock")
    driver._client = client
    driver._connected = True
    return driver


class TestTheMicroduckHoldsTheSameContract:
    """The peer that was already right, pinned so it cannot drift back."""

    def _duck(self) -> tuple[MicroduckDriver, _RobotdDouble]:
        client = _RobotdDouble()
        return _duck(client), client

    def test_send_action_clears_a_halt_it_supersedes(self) -> None:
        driver, client = self._duck()
        asyncio.run(driver.stop())
        assert _flag(driver) is True
        assert driver.send_action({"vx": 0.1})["status"] == "success"
        assert "robot.stop" in client.methods
        assert _flag(driver) is False

    def test_an_action_naming_nothing_sendable_does_not_clear_it(self) -> None:
        driver, _ = self._duck()
        asyncio.run(driver.stop())
        assert driver.send_action({"elbow": 1.0})["status"] == "error"
        assert _flag(driver) is True


class TestEveryDriverPublishingTheFlagClearsItOnItsWritePath:
    """Derived from the shipped drivers, so a third one is graded on arrival."""

    @staticmethod
    def _publishers() -> dict[str, Any]:
        """Shipped native driver classes whose ``get_status`` publishes the flag."""
        classes: dict[str, Any] = {}
        for robot in list_native_drivers():
            cls = get_native_driver_class(robot)
            if cls is not None:
                classes[cls.__name__] = cls
        assert len(classes) > 5, f"the driver table did not load: {sorted(classes)}"
        return {name: cls for name, cls in classes.items() if "motion_stopped" in inspect.getsource(cls.get_status)}

    def test_the_population_is_the_two_daemon_drivers(self) -> None:
        assert sorted(self._publishers()) == ["MicroduckDriver", "ReachyDriver"]

    def test_every_publisher_clears_the_flag_in_send_action(self) -> None:
        adrift = [
            name
            for name, cls in self._publishers().items()
            if "self._stopped = False" not in inspect.getsource(cls.send_action)
        ]
        assert adrift == [], f"these drivers publish motion_stopped but never clear it on the write path: {adrift}"

    @pytest.mark.parametrize("setter", ["stop", "stop_task"])
    def test_both_setters_are_still_the_only_way_to_report_a_halt(self, setter: str) -> None:
        """The flag becomes true on an accepted stop, and nowhere else."""
        source = inspect.getsource(ReachyDriver)
        assert source.count("self._stopped = True") == 2, "a new path started reporting a halt"
        assert f"def {setter}" in source
