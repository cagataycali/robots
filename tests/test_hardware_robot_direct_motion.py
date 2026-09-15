"""``Robot(mode="real")`` can move one joint a little, with the operator's yes, and reads it back.

The real-hardware tool's motion verbs were ``execute`` and ``start`` - a policy
rollout. Asked to "move the wrist 5 degrees" an agent had no action for it.
These tests grade the direct path on a fake lerobot bus that records every
write, using the same answering-agent shape as
``tests/test_hardware_robot_stream_gates_real_dispatch.py``:

* a declined or unanswered call writes NOTHING (the plan is read-only);
* an approved call enables torque on the named joints only, writes one
  ``Goal_Position``, waits, reads back and reports reached/error per joint;
* over-cap travel and unknown joints are refused BEFORE the operator is asked,
  naming the joint, the travel, the cap and the remedy;
* an uncalibrated arm is commanded in the encoder frame ``get_state`` reports;
  a calibrated one through lerobot's normalised write;
* ``set_torque enabled=false`` is never gated; ``enabled=true`` is;
* the pre-approval env var pre-approves by action name.

One ``hardware``-marked test moves a real wrist 5 degrees and back when
``STRANDS_HW_PORT`` is set (HARDWARE.md rules: <=10 deg, torque off after).
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any, cast

import pytest
from strands.interrupt import Interrupt, _InterruptState
from strands.types._events import ToolInterruptEvent, ToolResultEvent
from strands.types.tools import ToolUse

from strands_robots import hardware_motion
from strands_robots import hardware_robot as hardware_robot_module
from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_robot import RobotTaskState
from tests._daemon_executor import DaemonThreadExecutor
from tests.test_hardware_robot_observe_actions import FakeBus, FakeLeRobot, _Mode

# The autouse fixture below stubs ``time.sleep`` (the module attribute the motion
# code reads); the hardware test needs the real one for a real settle.
_REAL_SLEEP = time.sleep


class MotionBus(FakeBus):
    """The fake bus plus the write surface a move touches: torque, Goal_Position, Operating_Mode."""

    def __init__(self, *, calibrated: bool) -> None:
        super().__init__(calibrated=calibrated)
        self.modes = {name: 0 for name in self.motors}
        self.goal_writes: list[tuple[dict[str, Any], bool]] = []
        self.torque_calls: list[tuple[str, list[str] | None]] = []

    def read(self, register: str, motor: str, *, normalize: bool = True) -> int:
        if register == "Operating_Mode":
            self.reads.append((register, motor))
            return self.modes[motor]
        return super().read(register, motor, normalize=normalize)

    def enable_torque(self, motors=None, num_retry: int = 0) -> None:
        names = list(self.motors) if motors is None else list(motors)
        self.torque_calls.append(("enable", names))
        for n in names:
            self.torque[n] = 1

    def disable_torque(self, motors=None, num_retry: int = 0) -> None:
        names = list(self.motors) if motors is None else list(motors)
        self.torque_calls.append(("disable", names))
        for n in names:
            self.torque[n] = 0

    def sync_write(self, register: str, values: dict[str, Any], *, normalize: bool = True, num_retry: int = 0) -> None:
        assert register == "Goal_Position"
        self.writes.append((register, values))
        self.goal_writes.append((dict(values), normalize))
        for name, v in values.items():
            if normalize:
                # inverse of FakeBus.sync_read's stand-in calibration
                self.ticks[name] = round((v - 1.0) / 360 * 4096 + 2048) if name != "gripper" else self.ticks[name]
            else:
                self.ticks[name] = int(v)
        # a servo that lands 3 ticks short, so read-back is measured not echoed
        for name in values:
            self.ticks[name] -= 3


class MotionRobot(FakeLeRobot):
    """The fake arm carrying the write-recording bus, which is what a move reads back."""

    bus: MotionBus

    def __init__(self, *, calibrated: bool) -> None:
        super().__init__(calibrated=calibrated)
        self.bus = MotionBus(calibrated=calibrated)


def _robot(*, calibrated: bool) -> MotionRobot:
    return MotionRobot(calibrated=calibrated)


def _normalised_body(robot: MotionRobot) -> MotionRobot:
    """The arm shape lerobot's ``koch``/``omx`` follower has by default.

    Their config's ``use_degrees`` is false, so every body joint is normalised
    to ``-100..100`` percent of its calibrated range instead of degrees - the
    gripper is already ``0-100``. Nothing but the servo's declared mode changes.
    """
    for motor in robot.bus.motors.values():
        if motor.norm_mode is _Mode.DEGREES:
            motor.norm_mode = _Mode.RANGE_M100_100
    return robot


def _make_hw(robot: FakeLeRobot) -> HwRobot:
    hw = HwRobot.__new__(HwRobot)
    hw.tool_name_str = "arm"
    hw.action_horizon = 8
    hw.data_config = None
    hw.control_frequency = 30.0
    hw.action_sleep_time = 1.0 / 30.0
    hw._task_state = RobotTaskState()
    hw._executor = DaemonThreadExecutor(max_workers=1, thread_name_prefix="arm_executor")
    hw._shutdown_event = threading.Event()
    hw._stop_requested = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    hw.mesh = None
    hw.peer_id = None
    hw.robot = robot
    return hw


class _Answering(dict):
    def __init__(self, response: object) -> None:
        super().__init__()
        self._response = response

    def setdefault(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if key not in self:
            self[key] = Interrupt(default.id, default.name, default.reason, self._response)
        return self[key]


class _FakeAgent:
    def __init__(self, response: object | None) -> None:
        self._interrupt_state = _InterruptState(_Answering(response) if response is not None else {})
        self.cancel_signal = threading.Event()


def _state(response: object | None) -> dict[str, Any]:
    return {"agent": _FakeAgent(response)}


def _stream(hw: HwRobot, state: dict[str, Any], **tool_input: Any) -> list:
    tool_use = cast(ToolUse, {"toolUseId": "tu-motion", "input": tool_input})

    async def _run() -> list:
        return [ev async for ev in hw.stream(tool_use, state)]

    return asyncio.run(_run())


def _result(events: list) -> dict[str, Any]:
    assert isinstance(events[-1], ToolResultEvent), events[-1]
    return dict(events[-1].tool_result)


def _text(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


@pytest.fixture(autouse=True)
def _clean_gate_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    for name in ("BYPASS_TOOL_CONSENT", hardware_robot_module.COMMAND_ALLOW_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setattr(hardware_motion.time, "sleep", lambda s: None)


class TestNothingMovesWithoutAYes:
    def test_a_declined_move_writes_nothing(self) -> None:
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        result = _result(_stream(hw, _state("n"), action="set_joint_positions", positions={"elbow_flex": 95.0}))
        assert result["status"] == "error"
        assert robot.bus.goal_writes == [] and robot.bus.torque_calls == []
        assert robot.bus.writes == []

    def test_an_unanswered_move_is_an_interrupt_and_writes_nothing(self) -> None:
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        events = _stream(hw, _state(None), action="set_joint_positions", positions={"elbow_flex": 95.0})
        assert isinstance(events[-1], ToolInterruptEvent)
        reason = events[-1].interrupts[0].reason
        # The operator is shown the travel that would be written, joint by joint.
        assert "elbow_flex: 90.0 → 95.0° (+5.0)" in str(reason)
        assert "NOT calibrated" in str(reason)
        assert robot.bus.writes == []

    def test_headless_is_refused_naming_the_env_var(self) -> None:
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        result = _result(_stream(hw, {}, action="set_joint_positions", positions={"elbow_flex": 95.0}))
        assert result["status"] == "error"
        assert hardware_robot_module.COMMAND_ALLOW_ENV in _text(result)
        assert robot.bus.writes == []

    def test_set_torque_on_is_gated(self) -> None:
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        result = _result(_stream(hw, _state("n"), action="set_torque", enabled=True))
        assert result["status"] == "error"
        assert robot.bus.torque_calls == []


class TestRefusalsComeBeforeTheOperator:
    def test_unknown_joint_lists_the_real_ones_and_asks_nobody(self) -> None:
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        events = _stream(hw, _state(None), action="set_joint_positions", positions={"wrist": 5.0})
        result = _result(events)  # a result, not an interrupt
        assert "Unknown joint(s) ['wrist']" in _text(result)
        assert "['elbow_flex', 'gripper', 'shoulder_pan']" in _text(result)
        assert robot.bus.writes == []

    def test_over_cap_travel_is_refused_with_the_remedy(self) -> None:
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        result = _result(_stream(hw, _state(None), action="set_joint_positions", positions={"elbow_flex": 135.0}))
        text = _text(result)
        assert "elbow_flex: asked to travel +45.0°" in text
        assert "at most 20.0°" in text and "Split the move" in text and "max_relative_target" in text
        assert robot.bus.writes == []

    def test_declared_max_relative_target_is_the_cap(self) -> None:
        robot = _robot(calibrated=False)
        robot.config.max_relative_target = 5.0
        hw = _make_hw(robot)
        result = _result(_stream(hw, _state(None), action="set_joint_positions", positions={"elbow_flex": 96.0}))
        assert "at most 5.0°" in _text(result)

    def test_non_finite_and_empty_requests_are_refused(self) -> None:
        hw = _make_hw(_robot(calibrated=False))
        assert "must be finite" in _text(
            _result(_stream(hw, _state(None), action="set_joint_positions", positions={"elbow_flex": float("nan")}))
        )
        assert "needs `positions`" in _text(_result(_stream(hw, _state(None), action="set_joint_positions")))
        assert "needs `position`" in _text(_result(_stream(hw, _state(None), action="set_gripper")))
        assert "needs `enabled`" in _text(_result(_stream(hw, _state(None), action="set_torque")))

    def test_a_servo_not_in_position_mode_is_refused(self) -> None:
        robot = _robot(calibrated=False)
        robot.bus.modes["elbow_flex"] = 1
        hw = _make_hw(robot)
        result = _result(_stream(hw, _state("y"), action="set_joint_positions", positions={"elbow_flex": 95.0}))
        assert result["status"] == "error"
        assert "not in position mode" in _text(result) and "configure()" in _text(result)
        assert robot.bus.goal_writes == []


class TestAnApprovedMoveIsWrittenAndReadBack:
    def test_uncalibrated_arm_is_commanded_in_the_encoder_frame(self) -> None:
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)

        result = _result(_stream(hw, _state("y"), action="set_joint_positions", positions={"elbow_flex": 95.0}))

        assert result["status"] == "success", _text(result)
        assert robot.bus.torque_calls == [("enable", ["elbow_flex"])]
        assert robot.bus.goal_writes == [({"elbow_flex": hardware_motion.degrees_to_ticks(95.0)}, False)]
        plan = result["content"][1]["json"]
        assert plan["frame"] == "encoder_estimate"
        assert plan["current"] == {"elbow_flex": 90.0}
        assert plan["targets"] == {"elbow_flex": 95.0}
        # landed 3 ticks (~0.26 deg, rounded through the tick grid) short: measured, within tolerance
        assert plan["reached"] == {"elbow_flex": True}
        assert plan["error"]["elbow_flex"] == pytest.approx(-0.26, abs=0.02)
        assert plan["torque_left_on"] == ["elbow_flex"]
        text = _text(result)
        assert "1/1 reached" in text
        assert "elbow_flex: 90.0° → target 95.0°, actual 94.8° - reached" in text
        assert "Torque is ON on elbow_flex" in text and "set_torque enabled=false" in text
        assert "encoder estimate" in text

    def test_calibrated_arm_goes_through_the_normalised_write(self) -> None:
        robot = _robot(calibrated=True)
        hw = _make_hw(robot)

        result = _result(_stream(hw, _state("y"), action="set_joint_positions", positions={"elbow_flex": 96.0}))

        assert result["status"] == "success", _text(result)
        assert robot.bus.goal_writes == [({"elbow_flex": 96.0}, True)]
        assert result["content"][1]["json"]["frame"] == "calibration"
        assert "encoder estimate" not in _text(result)

    def test_a_stalled_servo_is_reported_not_reached(self) -> None:
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        real_write = robot.bus.sync_write

        def _stall(register, values, *, normalize=True, num_retry=0):
            robot.bus.writes.append((register, values))
            robot.bus.goal_writes.append((dict(values), normalize))  # commanded, but the servo does not move

        robot.bus.sync_write = _stall  # type: ignore[method-assign]
        result = _result(_stream(hw, _state("y"), action="set_joint_positions", positions={"elbow_flex": 100.0}))
        assert result["status"] == "success"
        plan = result["content"][1]["json"]
        assert plan["reached"] == {"elbow_flex": False}
        assert plan["error"]["elbow_flex"] == pytest.approx(-10.0, abs=0.01)
        assert "NOT reached (error -10.0°)" in _text(result)
        robot.bus.sync_write = real_write  # type: ignore[method-assign]

    def test_raw_ticks_path(self) -> None:
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        result = _result(
            _stream(hw, _state("y"), action="set_joint_positions", positions={"elbow_flex": 3100}, raw=True)
        )
        assert result["status"] == "success", _text(result)
        assert robot.bus.goal_writes == [({"elbow_flex": 3100}, False)]
        assert result["content"][1]["json"]["frame"] == "ticks"
        assert "3072.0 ticks → target 3100.0 ticks" in _text(result)

    def test_set_gripper_is_a_move_of_the_gripper(self) -> None:
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        result = _result(_stream(hw, _state("y"), action="set_gripper", position=15.0))
        assert result["status"] == "success", _text(result)
        assert robot.bus.torque_calls == [("enable", ["gripper"])]
        assert list(robot.bus.goal_writes[0][0]) == ["gripper"]

    def test_env_pre_approval_by_action_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(hardware_robot_module.COMMAND_ALLOW_ENV, "set_joint_positions")
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        assert (
            _result(_stream(hw, {}, action="set_joint_positions", positions={"elbow_flex": 95.0}))["status"]
            == "success"
        )
        # ...and only that name: set_gripper is still asked.
        gripper = _result(_stream(hw, {}, action="set_gripper", position=15.0))
        assert gripper["status"] == "error" and hardware_robot_module.COMMAND_ALLOW_ENV in _text(gripper)
        assert len(robot.bus.goal_writes) == 1

    def test_the_plan_read_does_not_hold_a_torque_write(self) -> None:
        """Reading before the gate is a read: torque is touched only after the yes."""
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        _stream(hw, _state(None), action="set_joint_positions", positions={"elbow_flex": 95.0})
        assert robot.bus.torque_calls == []
        assert robot.bus.is_connected, "the plan opened the bus to read it"
        assert robot.configure_calls == 0


class TestTorque:
    def test_release_is_never_gated(self) -> None:
        robot = _robot(calibrated=False)
        robot.bus.torque = {n: 1 for n in robot.bus.motors}
        hw = _make_hw(robot)

        result = _result(_stream(hw, {}, action="set_torque", enabled=False))  # headless, no env var

        assert result["status"] == "success", _text(result)
        assert robot.bus.torque_calls == [("disable", list(robot.bus.motors))]
        assert result["content"][1]["json"]["torque_enabled"] == {n: False for n in robot.bus.motors}
        assert "torque OFF (arm can be moved by hand)" in _text(result)

    def test_release_named_joints_only(self) -> None:
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        result = _result(_stream(hw, {}, action="set_torque", enabled=False, joints=["gripper"]))
        assert result["status"] == "success"
        assert robot.bus.torque_calls == [("disable", ["gripper"])]

    def test_hold_with_a_yes(self) -> None:
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        result = _result(_stream(hw, _state("y"), action="set_torque", enabled=True, joints=["elbow_flex"]))
        assert result["status"] == "success", _text(result)
        assert robot.bus.torque_calls == [("enable", ["elbow_flex"])]
        assert "torque ON (holding position) on elbow_flex" in _text(result)

    def test_unknown_joint_in_torque_is_refused(self) -> None:
        robot = _robot(calibrated=False)
        hw = _make_hw(robot)
        result = _result(_stream(hw, {}, action="set_torque", enabled=False, joints=["nope"]))
        assert result["status"] == "error" and "Unknown joint(s) ['nope']" in _text(result)


class TestUnits:
    @pytest.mark.parametrize(("deg", "ticks"), [(0.0, 2048), (90.0, 3072), (-180.0, 0), (180.0, 4095), (5.0, 2105)])
    def test_degrees_to_ticks(self, deg: float, ticks: int) -> None:
        assert hardware_motion.degrees_to_ticks(deg) == ticks

    def test_motion_actions_roster(self) -> None:
        assert hardware_motion.DIRECT_MOTION_ACTIONS <= hardware_robot_module.MOTION_ACTIONS
        assert "set_torque" in hardware_robot_module.MOTION_ACTIONS
        assert "stop" not in hardware_robot_module.MOTION_ACTIONS


class TestEveryTextQuotesTheUnitTheJointReportsIn:
    """A target is quoted in the unit the ARM reports that joint in, in every text.

    lerobot normalises each joint per its ``MotorNormMode``, so "degrees" is one
    of three answers: a calibrated gripper is ``0-100``, and every body joint of
    a ``koch``/``omx`` arm is ``-100..100`` (their ``use_degrees`` defaults to
    false). ``read_joint_state`` is the only thing that knows which, so the
    operator's warning, the over-cap refusal and the read-back all ask it.
    """

    @pytest.mark.parametrize(
        ("normalised_body", "joint", "target", "symbol", "mode"),
        [
            (False, "gripper", 52.0, "%", "range_0_100"),
            (False, "elbow_flex", 95.0, "°", None),
            (True, "elbow_flex", 95.0, "%", "range_m100_100"),
        ],
    )
    def test_the_operator_approves_travel_in_that_unit(
        self, normalised_body: bool, joint: str, target: float, symbol: str, mode: str | None
    ) -> None:
        robot = _robot(calibrated=True)
        if normalised_body:
            _normalised_body(robot)
        hw = _make_hw(robot)

        events = _stream(hw, _state(None), action="set_joint_positions", positions={joint: target})

        reason = str(events[-1].interrupts[0].reason)
        assert f"→ {target:.1f}{symbol}" in reason, reason
        if mode is not None:
            # A number the operator would read as degrees is the motion they did not approve.
            assert f"{target:.1f}°" not in reason
            assert mode in reason and "not degrees" in reason
        assert robot.bus.writes == []

    def test_the_gate_the_refusal_and_the_read_back_agree_on_one_joint(self) -> None:
        robot = _robot(calibrated=True)
        hw = _make_hw(robot)

        gate = str(_stream(hw, _state(None), action="set_gripper", position=52.0)[-1].interrupts[0].reason)
        refusal = _text(_result(_stream(hw, _state(None), action="set_gripper", position=99.0)))
        read_back = _text(_result(_stream(hw, _state("y"), action="set_gripper", position=52.0)))

        for text in (gate, refusal, read_back):
            assert "%" in text and "°" not in text, text

    def test_a_degrees_joint_reads_in_degrees_and_carries_no_units_note(self) -> None:
        hw = _make_hw(_robot(calibrated=True))
        text = _text(_result(_stream(hw, _state("y"), action="set_joint_positions", positions={"elbow_flex": 96.0})))
        assert "within 2° after" in text
        assert "elbow_flex: 91.0° → target 96.0°" in text
        assert "Units:" not in text and "%" not in text

    def test_a_mixed_call_gives_each_joint_its_own_unit(self) -> None:
        hw = _make_hw(_robot(calibrated=True))
        text = _text(
            _result(
                _stream(hw, _state("y"), action="set_joint_positions", positions={"elbow_flex": 96.0, "gripper": 52.0})
            )
        )
        assert "within 2, each in its own unit" in text
        assert "elbow_flex: 91.0° → target 96.0°" in text
        assert "gripper: 42.0% → target 52.0%" in text
        assert "Units: gripper in 0-100 percent of the calibrated range (range_0_100) - not degrees." in text

    @pytest.mark.parametrize(
        ("entry", "raw", "symbol", "label"),
        [
            ({"degrees": 1.0, "degrees_source": "calibration"}, False, "°", "degrees"),
            ({"degrees": 1.0, "degrees_source": "encoder_estimate"}, False, "°", "degrees"),
            ({"normalized": 42.0, "normalized_unit": "range_0_100"}, False, "%", "0-100 percent"),
            ({"normalized": 1.0, "normalized_unit": "range_m100_100"}, False, "%", "-100 to +100 percent"),
            ({"normalized": 1.0, "normalized_unit": "whatever_lerobot_adds_next"}, False, "%", "normalised"),
            ({"degrees": 1.0, "degrees_source": "calibration"}, True, " ticks", "encoder ticks"),
        ],
    )
    def test_unit_of_answers_from_the_state_entry(
        self, entry: dict[str, Any], raw: bool, symbol: str, label: str
    ) -> None:
        got_symbol, got_label = hardware_motion.unit_of(entry, raw=raw)
        assert got_symbol == symbol
        assert label in got_label


@pytest.mark.hardware
def test_real_wrist_moves_five_degrees_and_back_then_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    """HARDWARE.md: <=10 deg from the current pose, explicit pre-approval, torque OFF at the end."""
    port = os.environ.get("STRANDS_HW_PORT")
    if not port:
        pytest.skip("set STRANDS_HW_PORT=/dev/... to move a real arm")
    monkeypatch.setattr(hardware_motion.time, "sleep", _REAL_SLEEP)  # a real settle
    monkeypatch.setenv(hardware_robot_module.COMMAND_ALLOW_ENV, "set_joint_positions")
    from strands_robots import Robot

    arm = Robot("so101", mode="real", port=port)
    try:
        start = _result(_stream(arm, {}, action="get_state"))["content"][1]["json"]["joints"]
        w0 = start["wrist_roll"]["degrees"]
        up = _result(_stream(arm, {}, action="set_joint_positions", positions={"wrist_roll": w0 + 5.0}))
        assert up["status"] == "success", _text(up)
        assert up["content"][1]["json"]["reached"]["wrist_roll"] is True
        back = _result(_stream(arm, {}, action="set_joint_positions", positions={"wrist_roll": w0}))
        assert back["status"] == "success", _text(back)
        off = _result(_stream(arm, {}, action="set_torque", enabled=False))
        assert off["status"] == "success", _text(off)
        end = _result(_stream(arm, {}, action="get_state"))["content"][1]["json"]
        assert end["torque_enabled_any"] is False
        assert abs(end["joints"]["wrist_roll"]["degrees"] - w0) <= 2.0
    finally:
        arm.cleanup()
