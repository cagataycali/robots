"""Guard: ``stop_teleoperate`` reports whether the loop actually stopped.

``threading.Thread.join(timeout=...)`` returns ``None`` whether or not the
thread finished, so the liveness read after it is the only thing that
distinguishes a stopped loop from one that outlasted the budget. Without that
read the verb answered ``status="success"`` for a leader whose ``get_action()``
was still blocking - a serial read on a wedged bus is the ordinary case the
timeout exists for - and then disconnected the devices under the live loop and
set the thread handle to ``None``, so no later call and no status read could
discover the loop was still polling that leader and writing to the follower.

Two properties are graded, because either alone leaves the other reachable:

* the envelope: an unjoined stop is an ``error`` naming the budget, carrying
  ``stopped=False``, and it leaves the devices connected rather than tearing the
  bus down mid-write;
* the handle: it is cleared only on a real join, so a second call re-joins the
  same loop and ``get_teleoperate_status`` can report ``thread_alive``.

The counters ``_teleop_stats`` derives its status from cannot express this. A
session that ran cleanly until its leader wedged carries healthy ones, which is
why the join outcome is reported beside them rather than through them.

The budget is shortened through the module constant in most cells so the suite
does not spend the shipped 3 s per case; one cell drives the real constant.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

import strands_robots.teleop_mixin as tm

# Stated locally rather than imported, so these cells grade the shipped budget
# instead of following it. Exactly one cell asserts the two agree.
SHIPPED_JOIN_TIMEOUT_S = 3.0

# Short budget for the cells whose subject is the verdict, not the wait.
FAST_JOIN_TIMEOUT_S = 0.2


class WedgedLeader:
    """A leader whose ``get_action()`` blocks, as a serial read on a hung bus does."""

    name = "so101_leader"
    id = "leader"

    def __init__(self) -> None:
        self.is_connected = True
        self.entered = threading.Event()
        self.release = threading.Event()
        self.disconnect_calls = 0
        self.get_action_calls = 0

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False

    def get_action(self) -> dict[str, float]:
        self.get_action_calls += 1
        self.entered.set()
        self.release.wait(20.0)
        return {"shoulder_pan.pos": 1.0}


class PromptLeader:
    """A leader that answers immediately - the healthy control."""

    name = "so101_leader"
    id = "leader"

    def __init__(self) -> None:
        self.is_connected = True
        self.disconnect_calls = 0

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False

    def get_action(self) -> dict[str, float]:
        return {"shoulder_pan.pos": 1.0}


class Host(tm.TeleopMixin):
    """Minimal teleop host: the mixin needs only ``send_action``."""

    tool_name_str = "probe_robot"

    def __init__(self) -> None:
        self.sent: list[dict[str, float]] = []
        self._send_lock = threading.Lock()

    def send_action(self, action: dict[str, float], robot_name: str | None = None) -> dict[str, Any]:  # noqa: ARG002
        with self._send_lock:
            self.sent.append(dict(action))
        return {"status": "success", "content": [{"text": "ok"}]}


def _json(result: dict[str, Any]) -> dict[str, Any]:
    return next(block["json"] for block in result["content"] if "json" in block)


def _text(result: dict[str, Any]) -> str:
    return " ".join(block["text"] for block in result["content"] if "text" in block)


@pytest.fixture
def wedged(monkeypatch: pytest.MonkeyPatch):
    """A running session whose leader is blocked inside ``get_action()``."""
    # raising=False so this fixture installs against a tree that has not
    # named the budget yet: a pre-fix run then reports which behaviour is
    # missing instead of an AttributeError from the fixture itself.
    monkeypatch.setattr(tm, "_TELEOP_JOIN_TIMEOUT_S", FAST_JOIN_TIMEOUT_S, raising=False)
    host = Host()
    device = WedgedLeader()
    host.attach_teleop(device, name="leader")
    assert host.teleoperate(hz=100.0, block=False)["status"] == "success"
    assert device.entered.wait(5.0), "the loop never reached the leader's get_action()"
    thread = host._teleop_thread
    assert thread is not None and thread.is_alive()
    try:
        yield host, device, thread
    finally:
        device.release.set()
        thread.join(timeout=5.0)


class TestThePremise:
    """What the graded behaviour rests on."""

    def test_a_timed_join_cannot_report_whether_the_thread_finished(self) -> None:
        """``join(timeout=)`` returns None either way, so liveness is the only route."""
        blocked = threading.Event()
        thread = threading.Thread(target=lambda: blocked.wait(20.0), daemon=True)
        thread.start()
        try:
            assert thread.join(timeout=0.05) is None
            assert thread.is_alive() is True, "the probe thread was supposed to outlast the join"
        finally:
            blocked.set()
            thread.join(timeout=5.0)

    def test_the_wedged_leader_really_outlasts_the_budget(self, wedged) -> None:
        """Without this the regression cells would grade a loop that simply exited."""
        _host, _device, thread = wedged
        thread.join(timeout=FAST_JOIN_TIMEOUT_S)
        assert thread.is_alive() is True


class TestAnUnjoinedStopIsReportedHonestly:
    """The envelope a caller reads must not claim a stop that did not happen."""

    def test_the_status_is_an_error(self, wedged) -> None:
        host, _device, _thread = wedged
        assert host.stop_teleoperate()["status"] == "error"

    def test_the_payload_says_it_did_not_stop(self, wedged) -> None:
        host, _device, _thread = wedged
        assert _json(host.stop_teleoperate())["stopped"] is False

    def test_the_reason_names_the_budget_that_elapsed(self, wedged) -> None:
        host, _device, _thread = wedged
        assert f"{tm._TELEOP_JOIN_TIMEOUT_S:.1f}s" in _text(host.stop_teleoperate())

    def test_the_reason_names_what_is_still_being_driven(self, wedged) -> None:
        """The leader still being polled and the follower still being written."""
        host, _device, _thread = wedged
        text = _text(host.stop_teleoperate())
        assert "leader" in text
        assert "probe_robot" in text

    def test_the_shipped_budget_is_honoured_when_it_is_not_shortened(self) -> None:
        """One cell drives the real constant, so the budget itself is graded."""
        host = Host()
        device = WedgedLeader()
        host.attach_teleop(device, name="leader")
        host.teleoperate(hz=100.0, block=False)
        assert device.entered.wait(5.0)
        thread = host._teleop_thread
        try:
            started = time.monotonic()
            result = host.stop_teleoperate()
            waited = time.monotonic() - started
            assert result["status"] == "error"
            assert waited >= SHIPPED_JOIN_TIMEOUT_S
        finally:
            device.release.set()
            if thread is not None:
                thread.join(timeout=5.0)

    def test_a_frame_can_still_reach_the_follower_after_the_stop_returned(self, wedged) -> None:
        """The consequence the success claim hid: the loop is still writing."""
        host, device, thread = wedged
        result = host.stop_teleoperate()
        assert result["status"] == "error"
        before = len(host.sent)
        device.release.set()
        thread.join(timeout=5.0)
        assert len(host.sent) > before, "the loop was supposed to still be mid-write"


class TestTheHandleSurvivesAnUnjoinedStop:
    """The handle is the only route by which the live loop stays discoverable."""

    def test_the_thread_handle_is_kept(self, wedged) -> None:
        host, _device, thread = wedged
        host.stop_teleoperate()
        assert host._teleop_thread is thread

    def test_the_status_verb_reports_the_thread_as_alive(self, wedged) -> None:
        host, _device, _thread = wedged
        host.stop_teleoperate()
        assert _json(host.get_teleoperate_status())["thread_alive"] is True

    def test_a_second_call_re_joins_the_same_loop_and_succeeds(self, wedged) -> None:
        host, device, thread = wedged
        assert host.stop_teleoperate()["status"] == "error"
        device.release.set()
        thread.join(timeout=5.0)
        again = host.stop_teleoperate()
        assert again["status"] == "success"
        assert _json(again)["stopped"] is True
        assert host._teleop_thread is None


class TestTheBusIsNotTornDownMidWrite:
    """A device is disconnected once the loop has joined, not before."""

    def test_the_devices_stay_connected_while_the_loop_lives(self, wedged) -> None:
        host, device, _thread = wedged
        host.stop_teleoperate()
        assert device.disconnect_calls == 0
        assert device.is_connected is True


class TestAJoinedStopIsUnchanged:
    """Over-reach controls: every expectation here held before the change too."""

    def test_a_healthy_session_still_reports_success(self) -> None:
        host = Host()
        device = PromptLeader()
        host.attach_teleop(device, name="leader")
        host.teleoperate(hz=200.0, block=False)
        deadline = time.monotonic() + 5.0
        while not host.sent and time.monotonic() < deadline:
            time.sleep(0.01)
        assert host.stop_teleoperate()["status"] == "success"

    def test_a_healthy_session_still_disconnects_its_devices(self) -> None:
        host = Host()
        device = PromptLeader()
        host.attach_teleop(device, name="leader")
        host.teleoperate(hz=200.0, block=False)
        host.stop_teleoperate()
        assert device.disconnect_calls == 1
        assert device.is_connected is False

    def test_a_healthy_session_still_clears_the_handle(self) -> None:
        host = Host()
        host.attach_teleop(PromptLeader(), name="leader")
        host.teleoperate(hz=200.0, block=False)
        host.stop_teleoperate()
        assert host._teleop_thread is None

    def test_stopping_an_idle_host_is_still_a_no_op_success(self) -> None:
        result = Host().stop_teleoperate()
        assert result["status"] == "success"
        assert "No active teleoperation" in _text(result)


class TestTheLivenessReadIsStructurallyPresent:
    """Pin the read itself, so a revert to a bare timed join fails here.

    Scoped to this module rather than derived tree-wide: three other timed
    thread joins in the package answer different contracts (a ``-> None``
    teardown, a stats property, and a recording flush), so a tree-wide rule
    would grade surfaces this change does not touch.
    """

    def test_the_budget_is_a_named_constant(self) -> None:
        """The single cell that couples this file to the shipped constant.

        Named so the docstring, the refusal text and these cells read one value -
        and so a case whose subject is the verdict can shorten it.
        """
        assert tm._TELEOP_JOIN_TIMEOUT_S == SHIPPED_JOIN_TIMEOUT_S

    def test_the_timed_join_is_followed_by_a_liveness_read(self) -> None:
        import ast
        import inspect
        import textwrap

        body = ast.parse(textwrap.dedent(inspect.getsource(tm.TeleopMixin.stop_teleoperate)))
        joins = [
            node
            for node in ast.walk(body)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and (node.args or node.keywords)
        ]
        assert len(joins) == 1, "expected exactly one timed join to grade"
        alive = [
            node
            for node in ast.walk(body)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "is_alive"
        ]
        assert alive, "the timed join's outcome is never read"
        assert joins[0].lineno < alive[0].lineno, "liveness must be read after the join"
