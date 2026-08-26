"""Tests for the G1 control loop wired by harness#361 PR-C.

These grade the loop's transport primitive: 500 Hz cadence, per-step re-gate,
zero-torque frame on every terminal path except a wire refusal, exit reasons
named on every branch.  The unitree_sdk2py imports are lazy on both the
builders and the loop, so the tests run without the SDK on the box - the
publisher is a callable double the driver records writes on.

The loop's shutdown path is verified two ways: by inspecting the exit reason
in the snapshot, and by counting the frames the publisher recorded.  A stop
always leaves the driver with ``_loop = None`` so a subsequent ``run_policy``
starts fresh.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.drivers import g1 as g1_mod
from strands_robots.drivers.g1 import (
    _CONTROL_LOOP_DT,
    _CONTROL_LOOP_HZ,
    _ControlLoop,
    _refusal_text,
)

# ---------------------------------------------------------------------------
# Fakes.  The loop reaches into the driver's cached observations and its
# ``_pubs``; a MagicMock driver with those attributes exercises the loop
# without needing a real DDS bus or an SDK on the box.
# ---------------------------------------------------------------------------


class _RecordingPublisher:
    """Stand-in for ``DDSPublisher`` that records every ``publish`` call.

    Enough of the interface to grade the loop's write path.  ``publish``
    returns ``None`` on success and the string on failure - the same shape
    the real publisher returns.
    """

    def __init__(self, refuse_after: int | None = None, reason: str = "publish refused") -> None:
        self.calls: list[Any] = []
        self._refuse_after = refuse_after
        self._reason = reason

    def publish(self, topic: str, klass: Any, cmd: Any) -> str | None:
        self.calls.append((topic, klass, cmd))
        if self._refuse_after is not None and len(self.calls) > self._refuse_after:
            return self._reason
        return None


def _fake_driver(
    mode_machine: int | None = 9,
    fsm_id: int | None = 500,
    gate_result: dict[str, Any] | None = None,
    publisher: _RecordingPublisher | None = None,
) -> Any:
    """Return a MagicMock driver good enough for the loop.

    Every attribute the loop reads is a real value (not a Mock stand-in) so
    a typo in the loop code fails on AttributeError rather than reading a
    Mock silently.
    """
    driver = MagicMock(
        spec=[
            "_mode_machine",
            "_fsm_id",
            "_battery",
            "_imu",
            "_pubs",
            "_check_motion_gates",
            "_loop",
        ]
    )
    driver._mode_machine = mode_machine
    driver._fsm_id = fsm_id
    driver._battery = {"pct": 80.0}
    driver._imu = {"rpy": [0.0, 0.0, 0.0]}
    driver._pubs = publisher if publisher is not None else _RecordingPublisher()
    driver._check_motion_gates = MagicMock(return_value=gate_result)
    driver._loop = None
    return driver


def _wait_finished(loop: _ControlLoop, timeout: float = 2.0) -> None:
    """Poll ``is_running`` until the loop has joined its thread."""
    deadline = time.monotonic() + timeout
    while loop.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not loop.is_running, "loop did not finish within timeout"


# ---------------------------------------------------------------------------
# Cadence constants.
# ---------------------------------------------------------------------------


class TestCadenceConstants:
    """The SDK example uses 500 Hz; assert the constants match."""

    def test_500hz_matches_sdk_reference(self) -> None:
        assert _CONTROL_LOOP_HZ == 500.0

    def test_dt_is_the_reciprocal(self) -> None:
        assert _CONTROL_LOOP_DT == pytest.approx(0.002, rel=1e-6)


# ---------------------------------------------------------------------------
# Exit reasons.  Every terminal path names itself.
# ---------------------------------------------------------------------------


class TestExitReasons:
    """Every branch that leaves ``_run`` sets ``exit_reason``."""

    def test_n_steps_budget_exits_named(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=3)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "n_steps"
        assert snap["steps"] == 3
        assert driver._loop is None, "loop must clear its reference on exit"

    def test_duration_budget_exits_named(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        # Sub-tick duration so ``now >= deadline`` fires on the first check.
        loop = _ControlLoop(driver=driver, policy=policy, duration=0.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "duration"

    def test_gate_flip_exits_named_with_reason(self) -> None:
        refusal = {"status": "error", "content": [{"text": "FSM 999 refuses arm writes"}]}
        driver = _fake_driver(gate_result=refusal)

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=1.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "gate"
        assert snap["exit_detail"] == "FSM 999 refuses arm writes"

    def test_policy_returning_none_exits_named(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> None:
            return None

        loop = _ControlLoop(driver=driver, policy=policy, duration=1.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "policy"
        assert "None" in (snap["exit_detail"] or "")
        assert snap["refusals"] == 1

    def test_policy_raising_exits_named(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> None:
            raise RuntimeError("boom")

        loop = _ControlLoop(driver=driver, policy=policy, duration=1.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "policy"
        assert "RuntimeError" in (snap["exit_detail"] or "")
        assert "boom" in (snap["exit_detail"] or "")

    def test_stop_task_exits_named(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=None)
        loop.start()
        # Give it a moment to take at least one step.
        time.sleep(0.05)
        loop.stop("stop_task")
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "stop_task"

    def test_missing_publisher_exits_named(self) -> None:
        driver = _fake_driver()
        driver._pubs = None

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=1.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "publish"
        assert "not connected" in (snap["exit_detail"] or "")


# ---------------------------------------------------------------------------
# Zero-torque shutdown.  Every exit publishes it except a wire refusal.
# ---------------------------------------------------------------------------


class TestZeroTorqueShutdown:
    """The last frame the loop publishes is the zero-torque frame."""

    def _last_frame(self, pub: _RecordingPublisher) -> Any:
        assert pub.calls, "expected at least one publish call"
        return pub.calls[-1][2]

    def test_stop_publishes_zero_torque(self) -> None:
        pub = _RecordingPublisher()
        driver = _fake_driver(publisher=pub)

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=None)
        loop.start()
        time.sleep(0.05)
        loop.stop("stop_task")
        _wait_finished(loop)

        # The last publish must be the zero-torque frame.  Distinguish it
        # from the loop's action frames by checking that the commanded slot
        # for left_knee (index looked up via the module's mapping) carries
        # q=0.0/tau=0.0.  Any post-stop frame beyond the last action counts.
        cmd = self._last_frame(pub)
        left_knee_slot = g1_mod._G1_JOINT_INDEX["left_knee"]
        assert cmd.motor_cmd[left_knee_slot].q == 0.0
        assert cmd.motor_cmd[left_knee_slot].tau == 0.0
        assert cmd.motor_cmd[left_knee_slot].kp == 0.0

    def test_gate_flip_publishes_zero_torque(self) -> None:
        # First step passes the gate, second step refuses.
        refusal = {"status": "error", "content": [{"text": "FSM 999 refuses arm writes"}]}
        gate_returns = [None, refusal]
        driver = _fake_driver()
        driver._check_motion_gates = MagicMock(side_effect=gate_returns)

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        pub = driver._pubs
        loop = _ControlLoop(driver=driver, policy=policy, duration=1.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        # First action frame, then zero-torque on gate refusal.
        assert len(pub.calls) >= 2
        left_knee_slot = g1_mod._G1_JOINT_INDEX["left_knee"]
        stop_frame = pub.calls[-1][2]
        assert stop_frame.motor_cmd[left_knee_slot].kp == 0.0
        assert loop.snapshot()["exit_reason"] == "gate"

    def test_publish_refusal_does_not_double_stamp(self) -> None:
        """When the wire refuses, the loop does not stamp another frame.

        A second publish would clobber the reason with a fresh wire error
        rather than surfacing the original refusal.
        """
        # Refuse immediately - the first action publish fails.
        pub = _RecordingPublisher(refuse_after=0, reason="dds refused")
        driver = _fake_driver(publisher=pub)

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=1.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "publish"
        assert snap["exit_detail"] == "dds refused"
        # Exactly one call - the action attempt.  No zero-torque follow-up.
        assert len(pub.calls) == 1


# ---------------------------------------------------------------------------
# Per-step re-gate.  The gate is consulted on every step, not once at start.
# ---------------------------------------------------------------------------


class TestPerStepReGate:
    """The FSM gate runs on every iteration, not just at start."""

    def test_gate_is_called_once_per_step(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=5)
        loop.start()
        _wait_finished(loop)

        # Called ``n_steps`` times (one per step; the ``n_steps`` budget
        # check fires before the gate on the 6th iteration).
        assert driver._check_motion_gates.call_count == 5

    def test_gate_scope_is_motion(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=2)
        loop.start()
        _wait_finished(loop)

        # ``_check_motion_gates("motion")`` on every call.
        for call in driver._check_motion_gates.call_args_list:
            args, _ = call
            assert args == ("motion",)


# ---------------------------------------------------------------------------
# Snapshot invariants.
# ---------------------------------------------------------------------------


class TestSnapshot:
    """``snapshot()`` returns a consistent shape from any thread."""

    def test_snapshot_before_start_is_stable(self) -> None:
        driver = _fake_driver()
        loop = _ControlLoop(driver=driver, policy=lambda _o: None, duration=1.0, n_steps=1)
        snap = loop.snapshot()
        assert snap["running"] is False
        assert snap["steps"] == 0
        assert snap["exit_reason"] is None
        assert snap["elapsed_s"] is None
        assert snap["hz"] == _CONTROL_LOOP_HZ

    def test_snapshot_shape_after_exit(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=1)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        # Every documented field is present.
        assert set(snap) == {
            "running",
            "steps",
            "refusals",
            "elapsed_s",
            "duration_budget_s",
            "n_steps_budget",
            "exit_reason",
            "exit_detail",
            "hz",
        }
        assert snap["running"] is False
        assert snap["elapsed_s"] is not None and snap["elapsed_s"] >= 0.0


# ---------------------------------------------------------------------------
# Refusal text helper.
# ---------------------------------------------------------------------------


class TestRefusalText:
    """The helper extracts the first text entry, or a default."""

    def test_extracts_text_from_content(self) -> None:
        env = {"status": "error", "content": [{"text": "the reason"}]}
        assert _refusal_text(env) == "the reason"

    def test_returns_default_when_content_is_empty(self) -> None:
        assert _refusal_text({"status": "error", "content": []}) == "refused"

    def test_skips_non_text_entries(self) -> None:
        env = {"status": "error", "content": [{"json": {}}, {"text": "found"}]}
        assert _refusal_text(env) == "found"
