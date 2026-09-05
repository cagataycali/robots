# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``stop_teleoperate`` reached on the loop's own thread must still release the devices.

``Robot.__del__`` calls ``cleanup()``, and ``cleanup()`` calls
``stop_teleoperate()``. The loop's ``Thread`` target is a closure over the robot
(``teleoperate`` builds ``loop = lambda: self._teleop_loop(...)`` and hands it to
``Thread(target=loop)``), so that closure is the last reference to the robot once
the caller drops its handle. When the body returns, ``Thread.run`` does
``del self._target``, the reference goes, and the finalizer -- with the whole
terminal teardown behind it -- runs *on the teleop thread*.

``thread.join()`` there raises ``RuntimeError: cannot join current thread``. The
raise leaves ``stop_teleoperate`` from the middle of the method, so everything
after the join is skipped: the mesh publishers are not stopped, the attached
teleoperators are not disconnected, and the thread handle is not cleared.
``cleanup()`` catches it, warns, and -- because a raise leaves the outcome
unknown rather than positively reporting a live loop -- goes on to close the
robot's own devices and log ``cleanup completed``.

Measured on a one-camera arm driven by a leader holding its own port, with both
device nodes' exclusivity modelled:

    teardown route                       publishers stopped   leader disconnect   leader port
    finalized on its own teleop thread          no                   0             held
    explicit cleanup() (control)                yes                  1             released

So the robot's own bus and cameras were released and the *teleoperator's* port
was not -- which is why this went unnoticed: the leak is in the one device group
whose disconnect lives inside ``stop_teleoperate`` rather than in
``_disconnect_devices``. ``cleanup()``'s docstring says it disconnects "the
motors bus and every camera -- so no device node stays held"; on this route one
stayed held for the life of the process, under a ``cleanup completed`` report.

There is nothing to wait for on that route, which is what makes carrying on the
right answer rather than a guess: ``_teleop_loop`` never calls this verb (graded
below), so the only way control reaches it on the loop's thread is after the
body has already returned, and no other thread can be writing -- so the "do not
tear the bus down under a live writer" reason the failed-join branch exists for
cannot apply. The decision ``cleanup()`` makes about the robot's *own* devices is
deliberately unchanged: it closed them on the raise path and it closes them now.

The records the raise produced are also how this surfaced. A finalizer runs at
garbage-collection time, so its two records land in whatever code happens to be
executing -- an unrelated test, in a suite with daemon threads. A cell that means
"this function warned about nothing" and asserts ``caplog.text == ""`` grades
every record in the process at every level for the whole test, so it reads a
finalizer on another thread as its own subject speaking. The three cells that
made that claim now grade the records of the logger they name; the pytest
behaviour that makes the distinction necessary is pinned here.
"""

from __future__ import annotations

import ast
import inspect
import logging
import pathlib
import threading
import time
from typing import Any

import pytest

import strands_robots.teleop_mixin as tm
from strands_robots.hardware_robot import Robot as HwRobot
from tests.test_hardware_cleanup_disconnects import _arm, _make_robot, _Port

# Short enough to keep every cell fast. The shipped budget is pinned by
# ``test_stop_teleoperate_reports_the_join_outcome``; what matters here is only
# that a join can be attempted and can fail.
_FAST_JOIN_S = 0.3

# The cells that claim a named logger stayed quiet. Held as a literal so a
# fourth one cannot join the claim without joining the rule.
QUIET_LOGGER_CLAIMS = (
    "tests/test_dashboard_auth_enabled_flag_domain.py",
    "tests/mesh/test_non_hub_session_topology.py",
    "tests/mesh/test_stream_hz_domain.py",
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fast_join(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tm, "_TELEOP_JOIN_TIMEOUT_S", _FAST_JOIN_S)


class _LeaderPort:
    """The teleoperator's own device node: exclusive, like the follower's."""

    def __init__(self) -> None:
        self.held_by: str | None = None

    def open(self, holder: str) -> None:
        if self.held_by is not None:
            raise OSError(f"[Errno 16] Device or resource busy: held by {self.held_by}")
        self.held_by = holder

    def close(self) -> None:
        self.held_by = None


class _Leader:
    """A teleoperator holding a port, released only inside ``stop_teleoperate``."""

    def __init__(self, port: _LeaderPort) -> None:
        self.port = port
        self.is_connected = False
        self.disconnects = 0

    def connect(self) -> None:
        self.port.open("leader")
        self.is_connected = True

    def get_action(self) -> dict[str, float]:
        return {"j0.pos": 0.5}

    def disconnect(self) -> None:
        self.disconnects += 1
        self.port.close()
        self.is_connected = False


class _Observables:
    """Everything a cell asserts on, none of it referring back to the robot."""

    def __init__(self) -> None:
        self.follower_port = _Port()
        self.leader_port = _LeaderPort()
        self.leader = _Leader(self.leader_port)
        self.driver: Any = None
        self.thread: threading.Thread | None = None
        self.publishers_stopped: list[str] = []
        self.body_returned = threading.Event()
        self.release = threading.Event()


def _register_session(obs: _Observables, *, keep: bool, stop_from_the_loop: list[Any] | None = None) -> Any:
    """Start a teleop session in its own frame, as ``teleoperate`` does.

    The robot is built and the thread started here so that when this frame
    returns the loop's closure is the only reference left -- which is the
    reference topology ``teleoperate`` creates and the reason the finalizer can
    land on the teleop thread.

    Args:
        obs: Handles the caller keeps; none of them reference the robot.
        keep: Return the robot too, so the caller pins it alive (the ordinary
            route, where the finalizer never runs on the loop's thread).
        stop_from_the_loop: When given, the loop body calls
            ``stop_teleoperate()`` itself and appends the envelope -- or the
            exception -- here. That is the same arrival as the finalizer's,
            deterministically and without a garbage-collection step.

    Returns:
        The robot when ``keep``, else ``None``.
    """
    driver = _arm(obs.follower_port)
    self = _make_robot(driver)
    driver.connect()
    obs.driver = driver
    self._ensure_teleop_state()
    obs.leader.connect()
    self._teleops = {"leader": type("_Att", (), {"device": obs.leader})()}
    self._teleop_robot_name = "test_arm"
    self._teleop_running = True
    self._teleop_start_mono = time.monotonic()

    body_returned, release = obs.body_returned, obs.release

    def body(robot: Any) -> None:
        while robot._teleop_running and not robot._teleop_stop_event.is_set():
            release.wait(10.0)
            break
        if stop_from_the_loop is not None:
            try:
                stop_from_the_loop.append(robot.stop_teleoperate())
            except Exception as exc:  # noqa: BLE001 - the raise IS what this cell grades
                stop_from_the_loop.append(exc)
        body_returned.set()

    # teleop_mixin.teleoperate: a local closure over the robot, handed to Thread.
    loop = lambda: body(self)  # noqa: E731
    self._teleop_thread = threading.Thread(target=loop, name=f"teleop-{self.tool_name_str}", daemon=True)
    obs.thread = self._teleop_thread
    self._teleop_thread.start()
    return self if keep else None


def _drop_and_finalize(obs: _Observables) -> None:
    """Let the loop return with its closure holding the last robot reference."""
    obs.release.set()
    assert obs.body_returned.wait(5.0), "premise: the loop body never returned"
    assert obs.thread is not None
    obs.thread.join(5.0)
    assert not obs.thread.is_alive(), "premise: the teleop thread never exited"


def _records(caplog: pytest.LogCaptureFixture, name: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == name]


def _watch_publisher_stop(monkeypatch: pytest.MonkeyPatch, seen: list[str]) -> None:
    """Record which thread stopped the session's publishers, and let it run.

    ``_stop_publishers`` is the first thing after the join, so it is the step a
    ``RuntimeError`` from ``join()`` silently takes with it.
    """
    original = HwRobot._stop_publishers

    def recorded(self: Any) -> None:
        seen.append(threading.current_thread().name)
        original(self)

    monkeypatch.setattr(HwRobot, "_stop_publishers", recorded)


class TestThePremise:
    """Why a self-join means the body has returned, and how it is reached."""

    def test_the_loop_never_calls_the_stop_verb_itself(self) -> None:
        """So control on the loop's thread can only be post-return."""
        tree = ast.parse(inspect.getsource(tm))
        loops = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_teleop_loop"]
        assert len(loops) == 1, "the loop body moved; this premise needs re-deriving"
        called = {
            (c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", ""))
            for c in ast.walk(loops[0])
            if isinstance(c, ast.Call)
        }
        assert called, "no calls found in the loop body: this scan is looking in the wrong place"
        assert "stop_teleoperate" not in called

    def test_the_finalizer_runs_the_terminal_teardown(self) -> None:
        """``__del__`` -> ``cleanup()`` -> ``stop_teleoperate()`` is the route."""
        assert "cleanup" in inspect.getsource(HwRobot.__del__)
        assert "stop_teleoperate" in inspect.getsource(HwRobot.cleanup)


class TestTheDevicesAreReleasedWhenTheLoopIsTheCaller:
    """The regression: reached on its own thread, the teardown still runs."""

    def test_the_stop_verb_returns_an_envelope_instead_of_raising(self) -> None:
        """``join()`` on the calling thread raised ``RuntimeError`` from mid-method."""
        obs = _Observables()
        outcome: list[Any] = []
        robot = _register_session(obs, keep=True, stop_from_the_loop=outcome)
        assert robot is not None
        obs.release.set()
        assert obs.body_returned.wait(5.0)
        assert obs.thread is not None
        obs.thread.join(5.0)

        assert len(outcome) == 1, "the loop never reached the stop verb"
        assert not isinstance(outcome[0], Exception), f"the verb raised: {outcome[0]!r}"
        assert outcome[0]["status"] in {"success", "error"}
        assert robot._teleop_thread is None, "the handle was left pointing at a finished loop"

    def test_the_teleoperator_port_is_released(self) -> None:
        obs = _Observables()
        _register_session(obs, keep=False)
        _drop_and_finalize(obs)
        assert obs.leader.disconnects == 1, "the teleoperator was never disconnected"
        assert obs.leader_port.held_by is None, "the teleoperator's device node is still held"

    def test_the_session_publishers_are_stopped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        obs = _Observables()
        _watch_publisher_stop(monkeypatch, obs.publishers_stopped)
        _register_session(obs, keep=False)
        _drop_and_finalize(obs)
        assert obs.publishers_stopped, "the publishers the session started were never stopped"

    def test_the_teardown_does_not_report_a_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        obs = _Observables()
        with caplog.at_level(logging.DEBUG):
            _register_session(obs, keep=False)
            _drop_and_finalize(obs)
        raised = [r.getMessage() for r in caplog.records if "raised during cleanup" in r.getMessage()]
        assert raised == []

    def test_the_robots_own_devices_are_still_closed(self) -> None:
        """Unchanged: ``cleanup()`` closed them on the raise path and still does."""
        obs = _Observables()
        _register_session(obs, keep=False)
        _drop_and_finalize(obs)
        assert obs.follower_port.held_by is None
        assert obs.driver.bus.is_connected is False


class TestTheCrossThreadPathIsUnchanged:
    """The ordinary route keeps reporting its join outcome honestly."""

    def test_a_clean_stop_still_joins_and_disconnects(self) -> None:
        obs = _Observables()
        robot = _register_session(obs, keep=True)
        assert robot is not None
        obs.release.set()
        assert obs.body_returned.wait(5.0)
        envelope = robot.stop_teleoperate()
        payload: dict[str, Any] = next((b["json"] for b in envelope["content"] if "json" in b), {})
        assert payload.get("stopped", True) is True
        assert robot._teleop_thread is None
        assert obs.leader.disconnects == 1
        assert obs.leader_port.held_by is None

    def test_a_wedged_loop_still_reports_that_it_did_not_stop(self) -> None:
        obs = _Observables()
        robot = _register_session(obs, keep=True)
        assert robot is not None
        envelope = robot.stop_teleoperate()  # the body is parked in release.wait()
        payload = next(b["json"] for b in envelope["content"] if "json" in b)
        assert envelope["status"] == "error"
        assert payload["stopped"] is False
        assert robot._teleop_thread is not None, "the handle must survive a failed join"
        assert obs.leader_port.held_by == "leader", "a live writer keeps its devices"
        obs.release.set()
        assert obs.body_returned.wait(5.0)


class TestAQuietLoggerClaimGradesItsOwnLogger:
    """Why the three quiet-logger cells cannot be written against ``caplog.text``."""

    def test_caplog_text_carries_every_logger_from_every_thread(self, caplog: pytest.LogCaptureFixture) -> None:
        """The pytest behaviour that makes the scoping necessary."""
        subject, foreign = "strands_robots.test_subject", "strands_robots.test_foreign"
        with caplog.at_level(logging.WARNING, logger=subject):
            pass  # the subject is asked nothing, so it says nothing
        # A finalizer's record, on the thread a finalizer runs on. ``cleanup()``
        # emits exactly this shape: a WARNING and an INFO on another thread, at
        # garbage-collection time, inside whatever test is executing.
        emitted = threading.Thread(target=lambda: logging.getLogger(foreign).warning("from a finalizer"))
        emitted.start()
        emitted.join(5.0)

        assert caplog.text != "", "premise: an unrelated logger's record is captured"
        assert _records(caplog, foreign), "premise: the foreign record reached caplog"
        assert _records(caplog, subject) == [], "the subject said nothing, and that is the claim"

    @pytest.mark.parametrize("relative", QUIET_LOGGER_CLAIMS)
    def test_the_claim_is_graded_against_a_named_logger(self, relative: str) -> None:
        path = _REPO_ROOT / relative
        assert path.exists(), f"{relative} moved; this rule needs re-pointing"
        text = path.read_text(encoding="utf-8")
        assert "caplog.records" in text, f"{relative} makes a quiet-logger claim without naming one"
        assert 'caplog.text == ""' not in text, (
            f"{relative} grades every record in the process, so a finalizer on another "
            "thread reads as its own subject speaking"
        )
