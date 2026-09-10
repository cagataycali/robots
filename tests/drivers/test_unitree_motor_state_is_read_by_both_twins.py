"""``rt/lowstate.motor_state`` is a reading, and both Unitree twins take it.

The G1 is a 29-DoF humanoid whose driver publishes PD targets on ``rt/lowcmd``
under gains up to :data:`~strands_robots.drivers.g1._SDK_KP` at the control
loop's rate. ``LowState_.motor_state`` is the only proprioception the robot
sends back, so a driver that does not read it commands every one of those
joints without ever observing one: nothing can check a target against the
measured pose, and a policy driven by
:meth:`~strands_robots.drivers.g1._ControlLoop._call_policy` is handed an
observation with no joint in it at all.

The twin driver already read the same array the same way -
:meth:`~strands_robots.drivers.go2.Go2Driver._on_lowstate` decoded
``motor_state[slot]`` per :data:`~strands_robots.drivers.go2.GO2_JOINT_INDEX`
into ``_joints`` and published it on the ``sensors`` verb - so one SDK family
answered a joint read on the quadruped and not on the humanoid.

These cells pin the reading on both drivers, that they share one decoder
(:func:`~strands_robots.drivers.base.decode_motor_state`) rather than a copy
each, and that the decoder distinguishes a field the frame does not carry from
a joint at zero.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import pathlib
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.drivers import g1 as g1_module
from strands_robots.drivers import go2 as go2_module
from strands_robots.drivers.base import decode_motor_state
from strands_robots.drivers.g1 import _G1_JOINT_INDEX, G1Driver, _ControlLoop
from strands_robots.drivers.go2 import GO2_JOINT_INDEX, Go2Driver

#: The record every joint reports, on either robot.
_JOINT_FIELDS = frozenset({"q", "dq", "tau_est", "temperature"})

#: ``unitree_hg``'s ``LowState_`` declares 35 motor slots; the G1 names 29.
_HG_MOTOR_SLOTS = 35


def _motor(slot: int, **overrides: Any) -> SimpleNamespace:
    """One ``MotorState_``, its readings derived from its slot so they differ."""
    fields: dict[str, Any] = {
        "q": 0.1 * slot,
        "dq": 0.01 * slot,
        "tau_est": 1.0 * slot,
        "temperature": 30 + slot,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _lowstate(slots: int = _HG_MOTOR_SLOTS, **overrides: Any) -> SimpleNamespace:
    """A ``LowState_`` carrying a full motor array, an IMU and a layout id."""
    return SimpleNamespace(
        motor_state=[_motor(i, **overrides) for i in range(slots)],
        imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
        mode_machine=4,
    )


def _g1() -> G1Driver:
    return G1Driver(tool_name="g1", port="192.168.1.172")


def _go2() -> Go2Driver:
    return Go2Driver(tool_name="go2", port="192.168.1.100")


class TestTheG1ReadsItsJoints:
    """The read the humanoid was missing."""

    def test_lowstate_populates_every_named_joint(self) -> None:
        driver = _g1()
        assert driver._joints is None, "a driver that has not heard lowstate reports no joints"

        driver._on_lowstate(_lowstate())

        assert driver._joints is not None
        assert set(driver._joints) == set(_G1_JOINT_INDEX), "every named joint is read"
        assert len(driver._joints) == 29

    def test_each_joint_carries_the_full_record(self) -> None:
        driver = _g1()
        driver._on_lowstate(_lowstate())
        assert driver._joints is not None
        for name, record in driver._joints.items():
            assert set(record) == _JOINT_FIELDS, f"{name} reports a partial record"

    def test_a_joint_reads_its_own_wire_slot(self) -> None:
        """Slot-for-slot: the humanoid's index table addresses the array."""
        driver = _g1()
        driver._on_lowstate(_lowstate())
        assert driver._joints is not None
        for name, slot in _G1_JOINT_INDEX.items():
            assert driver._joints[name] == {
                "q": pytest.approx(0.1 * slot),
                "dq": pytest.approx(0.01 * slot),
                "tau_est": pytest.approx(1.0 * slot),
                "temperature": 30 + slot,
            }

    def test_the_reserved_slots_are_not_reported(self) -> None:
        """35 slots arrive; the 29 the robot actuates are the ones read."""
        driver = _g1()
        driver._on_lowstate(_lowstate())
        assert driver._joints is not None
        assert max(_G1_JOINT_INDEX.values()) == 28
        assert len(driver._joints) < _HG_MOTOR_SLOTS

    def test_the_imu_and_layout_id_still_decode(self) -> None:
        """The new read shares a callback with the two it did not replace."""
        driver = _g1()
        driver._on_lowstate(_lowstate())
        assert driver._imu is not None
        assert driver._imu["rpy"] == [0.0, 0.0, 0.0]
        assert driver._mode_machine == 4


class TestTheJointsReachTheirConsumers:
    """A reading nothing exposes is not yet an answer to anything."""

    def test_the_sensors_verb_publishes_joints(self) -> None:
        driver = _g1()
        driver._on_lowstate(_lowstate())

        async def _read() -> dict[str, Any]:
            async for result in driver.stream({"toolUseId": "t", "name": "g1", "input": {"action": "sensors"}}, {}):
                return dict(result)
            raise AssertionError("the sensors verb yielded no result")

        result = asyncio.run(_read())

        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert set(payload["joints"]) == set(_G1_JOINT_INDEX)

    def test_the_sensors_verb_is_declared_to_return_them(self) -> None:
        """The verb's own description names what it hands back."""
        described = _g1().tool_spec["inputSchema"]["json"]["properties"]["action"]["description"]
        assert "joints" in described

    def test_the_policy_observation_carries_the_measured_pose(self) -> None:
        """The whole point of the read: a proprioceptive policy can run."""
        driver = _g1()
        driver._on_lowstate(_lowstate())

        seen: list[dict[str, Any]] = []

        def recording_policy(obs: dict[str, Any]) -> dict[str, float]:
            seen.append(obs)
            return {}

        # Built through the constructor, because that is where the loop
        # resolves the policy into the step callable it invokes per tick.
        loop = _ControlLoop(driver, recording_policy, duration=1.0, n_steps=None)

        loop._call_policy()

        assert len(seen) == 1
        joints = seen[0]["joints"]
        assert joints is not None
        assert len(joints) == 29
        for record in joints.values():
            assert set(record) == _JOINT_FIELDS

    def test_the_snapshot_does_not_hand_out_the_live_cache(self) -> None:
        driver = _g1()
        driver._on_lowstate(_lowstate())
        snapshot = driver._snapshot("_joints")
        assert snapshot is not None
        snapshot["left_knee"] = "clobbered"
        assert driver._joints is not None
        assert driver._joints["left_knee"] != "clobbered"


class TestOneDecoderForBothTwins:
    """The decode is shared, so the two drivers cannot drift apart on it."""

    def test_both_drivers_call_the_shared_decoder(self) -> None:
        for module in (g1_module, go2_module):
            assert module.decode_motor_state is decode_motor_state, (
                f"{module.__name__} must route through the one owner in drivers.base"
            )

    def test_the_go2_still_reads_its_twelve_joints(self) -> None:
        driver = _go2()
        driver._on_lowstate(
            SimpleNamespace(
                motor_state=[_motor(i) for i in range(20)],
                imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
                bms_state=SimpleNamespace(soc=88.0),
            )
        )
        assert driver._joints is not None
        assert set(driver._joints) == set(GO2_JOINT_INDEX)
        assert len(driver._joints) == 12

    def test_both_peers_report_one_joint_shape(self) -> None:
        """A fleet consumer reading one SDK family sees one record shape."""
        g1_driver = _g1()
        g1_driver._on_lowstate(_lowstate())
        go2_driver = _go2()
        go2_driver._on_lowstate(
            SimpleNamespace(motor_state=[_motor(i) for i in range(20)]),
        )
        assert g1_driver._joints is not None
        assert go2_driver._joints is not None

        g1_shapes = {frozenset(r) for r in g1_driver._joints.values()}
        go2_shapes = {frozenset(r) for r in go2_driver._joints.values()}
        assert g1_shapes == go2_shapes == {_JOINT_FIELDS}

    def test_neither_twin_keeps_a_private_motor_decode(self) -> None:
        """Neither Unitree module names a motor-record field itself.

        ``tau_est`` is a name only a ``MotorState_`` record carries, so either
        twin reading it outside the shared decoder has grown a second copy of
        this decode - which is how they came to disagree about whether the
        array was read at all. Scoped to the two ``rt/lowstate`` subscribers:
        :data:`~strands_robots.drivers.booster.MOTOR_STATE_FIELDS` decodes a
        different layout (parallel per-field vectors, not per-joint records)
        and already derives its own read from one table.
        """
        offenders: list[str] = []
        for module in (g1_module, go2_module):
            path = pathlib.Path(inspect.getfile(module))
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Constant) and node.value == "tau_est":
                    offenders.append(f"{path.name}:{node.lineno}")
        assert offenders == [], f"a private motor decode reappeared: {offenders}"

    def test_each_twin_reaches_the_decoder_exactly_once(self) -> None:
        """One call site per driver: the read is not duplicated inside one."""
        for module in (g1_module, go2_module):
            source = inspect.getsource(module)
            calls = source.count("decode_motor_state(getattr(msg,")
            assert calls == 1, f"{module.__name__} decodes motor_state {calls} times"


class TestADefaultIsNotAReading:
    """``0.0`` is a valid pose; a field the frame lacks must not read as one."""

    def test_an_absent_field_reads_none_not_zero(self) -> None:
        """A renamed IDL field is silence, not a joint parked at its zero."""
        motors = [SimpleNamespace(dq=0.5, tau_est=1.0, temperature=30) for _ in range(_HG_MOTOR_SLOTS)]
        decoded = decode_motor_state(motors, _G1_JOINT_INDEX)
        assert decoded is not None
        assert decoded["left_knee"]["q"] is None, "an unread position is not the zero position"
        assert decoded["left_knee"]["dq"] == pytest.approx(0.5), "the readable fields survive"

    def test_one_bad_field_does_not_discard_the_others(self) -> None:
        motors = [_motor(i, q=object()) for i in range(_HG_MOTOR_SLOTS)]
        decoded = decode_motor_state(motors, _G1_JOINT_INDEX)
        assert decoded is not None
        record = decoded["left_knee"]
        assert record["q"] is None
        assert record["temperature"] == 33, "the frame's other readings are kept"

    def test_an_absent_array_leaves_the_cache_unread(self) -> None:
        assert decode_motor_state(None, _G1_JOINT_INDEX) is None

    def test_a_buffer_is_not_a_motor_array(self) -> None:
        """A bytes-like array indexes to integers, which carry no readings."""
        for buffer in (b"\x01\x02", bytearray(b"\x01\x02"), memoryview(b"\x01\x02"), "0.1"):
            assert decode_motor_state(buffer, _G1_JOINT_INDEX) is None, buffer

    def test_an_array_answering_no_slot_is_not_a_reading(self) -> None:
        """Zero joints decoded says the array was unreadable, not jointless."""
        assert decode_motor_state([], _G1_JOINT_INDEX) is None

    def test_a_short_array_costs_only_the_slots_it_lacks(self) -> None:
        """A firmware carrying fewer slots keeps the joints it does carry."""
        decoded = decode_motor_state([_motor(i) for i in range(5)], _G1_JOINT_INDEX)
        assert decoded is not None
        assert set(decoded) == {name for name, slot in _G1_JOINT_INDEX.items() if slot < 5}
        assert decoded["left_hip_pitch"]["q"] == pytest.approx(0.0)

    def test_a_lowstate_without_motors_leaves_the_previous_reading(self) -> None:
        """A frame missing the array does not erase the pose already read."""
        driver = _g1()
        driver._on_lowstate(_lowstate())
        first = dict(driver._joints or {})
        driver._on_lowstate(SimpleNamespace(imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0])))
        assert driver._joints == first
