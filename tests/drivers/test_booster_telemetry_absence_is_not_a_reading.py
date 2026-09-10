"""A Booster T1 field the frame does not carry is not a zero reading.

:func:`~strands_robots.drivers.booster.parse_low_state` documents the rule -
"absent fields are omitted rather than defaulted: a snapshot that reports a
zeroed IMU the robot never sent is worse than one that reports none". #3381
applied it to the three IMU vectors; the four motor vectors were still
``float(getattr(motor, <field>, 0.0))`` and ``_on_battery`` read its three
fields the same way, so one function held two conventions and a field a
firmware revision renamed, dropped or turned into a flag arrived as a plausible
number on the rows that were left:

==============================  ===========================  ================
frame                           before                       after
==============================  ===========================  ================
``q`` renamed                   every joint at ``0.0``       ``None``
``q`` is a flag                 every joint at ``1.0``       ``None``
``soc`` renamed                 ``battery_pct=0.0``          ``None``
``soc`` is a flag               ``battery_pct=1.0``          ``None``
==============================  ===========================  ================

The ``joints`` row is the one that moves the robot rather than only misreporting
it. It is the hold source: :meth:`~strands_robots.drivers.booster.BoosterDriver.send_action`
reads it as ``held_q`` and :func:`~strands_robots.drivers.booster.build_frame`
writes ``held_q[slot]`` as the position target of every *uncommanded* upper-body
slot. A full-width vector of defaulted zeros is finite and non-empty, so it
passed both of ``_on_low_state``'s guards, got cached, and the very next write
commanded all eight arm joints to exactly zero - a full-travel move on a
1.2 m biped, from a frame that carried no positions at all. The refusal the
driver already spells for that case ("no LowState frame has arrived yet, so
neither the frame width nor the hold position of an uncommanded arm joint is
known") was unreachable, because a defaulted frame is indistinguishable from an
observed one.

The rule already had an owner. :mod:`strands_robots.drivers.base`'s
``telemetry_float`` / ``telemetry_float_list`` decide what counts as a reading
for the Unitree drivers, including the two rows neither of *those* copies
guarded before they converged - a ``bool`` on a scalar and a bytes-like on a
vector. The T1 reads through the same functions and keeps the same record shape
those drivers use: the key stays and the value is ``None``, so a consumer asking
for a field always gets an answer and the answer can be "the robot did not report
this". This suite grades the rule as a table derived from the driver's own field
maps - the IMU rows included, so the three maps are graded as one roster rather
than the motor and battery halves alone - the frame surviving one bad field, the
hold source end to end, and a derivation that refuses a typed default returning
to the module.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest

from strands_robots.drivers import booster
from strands_robots.drivers.base import telemetry_float, telemetry_float_list
from strands_robots.drivers.booster import (
    BATTERY_STATE_FIELDS,
    BOOSTER_JOINT_INDEX,
    IMU_STATE_FIELDS,
    MOTOR_STATE_FIELDS,
    UPPER_BODY_SLOTS,
    parse_low_state,
)

from .test_booster_driver import _FakeSdk, _live_driver, install_booster_sdk

_WIDTH = len(BOOSTER_JOINT_INDEX)


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> _FakeSdk:
    """The same SDK double the driver suite runs against."""
    return install_booster_sdk(monkeypatch)


#: Marks a field as *not present* on the message the double builds - a firmware
#: revision that renamed or dropped it.
_ABSENT = object()

#: Values that are no reading on a *scalar* field: absent, a flag, unset, text
#: that is not a number, or a raw buffer. ``bool`` is the row ``g1`` let through
#: before the Unitree copies of this rule converged - ``float(True)`` is ``1.0``.
_SCALAR_NON_READINGS: list[Any] = [
    _ABSENT,
    True,
    None,
    "n/a",
    memoryview(b"\x01\x02"),
    bytearray(b"\x01\x02"),
]

#: Values that are no reading on a *vector* field. Everything above plus the two
#: rows that only apply to a sequence: a bytes-like value iterates as integers
#: and so decoded a buffer into a two-element attitude (the row *neither*
#: Unitree copy guarded), and text iterates as characters. ``[0.1, True, 0.3]``
#: is the all-or-nothing rule: half an attitude is worse than none, because a
#: consumer cannot tell that it is half.
_VECTOR_NON_READINGS: list[Any] = [*_SCALAR_NON_READINGS, "0.05", [0.1, True, 0.3]]


class _Fields:
    """A decoded SDK message: named attributes, minus the ones marked absent."""

    def __init__(self, defaults: dict[str, Any], **overrides: Any) -> None:
        for name, value in {**defaults, **overrides}.items():
            if value is not _ABSENT:
                setattr(self, name, value)


def _motor(**overrides: Any) -> _Fields:
    return _Fields({"q": 0.5, "dq": 0.1, "tau_est": 0.2, "temperature": 30.0}, **overrides)


def _imu(**overrides: Any) -> _Fields:
    return _Fields({"rpy": (0.01, 0.02, 0.03), "gyro": (0.1, 0.2, 0.3), "acc": (0.0, 0.0, 9.81)}, **overrides)


def _low_state(
    *,
    motor: dict[str, Any] | None = None,
    imu: dict[str, Any] | None = None,
    one_slot: tuple[int, dict[str, Any]] | None = None,
) -> Any:
    """A full-width ``LowState`` whose motors and IMU carry ``overrides``.

    ``one_slot`` overrides a *single* slot, which is the case that separates the
    all-or-nothing rule from a filter: a vector that dropped one element would
    renumber every slot after the gap.
    """
    motors = [_motor(**(motor or {})) for _ in range(_WIDTH)]
    if one_slot is not None:
        slot, overrides = one_slot
        motors[slot] = _motor(**overrides)
    return _Fields({}, motor_state_parallel=motors, motor_state_serial=motors, imu_state=_imu(**(imu or {})))


def _battery(**overrides: Any) -> _Fields:
    return _Fields({"soc": 88.0, "voltage": 48.2, "current": -3.1}, **overrides)


def _unobserved_driver(sdk: _FakeSdk, *, enable: bool = True) -> booster.BoosterDriver:
    """A connected, write-enabled driver that has been handed no frame yet.

    :func:`_live_driver` injects a good ``LowState``, which is the wrong start
    for the cells about what happens when the *first* frame does not read.
    """
    driver = booster.BoosterDriver(cmd_type="parallel")
    assert driver.connect_eagerly() is None
    if enable:
        assert driver.enable_upper_body(True)["status"] == "success"
    return driver


class TestAFieldThatIsNoReadingIsOmittedRatherThanDefaulted:
    """The rule the docstring states, graded over the driver's own field maps.

    The rosters are read from :data:`MOTOR_STATE_FIELDS`, :data:`IMU_STATE_FIELDS`
    and :data:`BATTERY_STATE_FIELDS` rather than restated, so a field added to
    the driver is graded here without this file being touched.
    """

    @pytest.mark.parametrize("key,field", sorted(MOTOR_STATE_FIELDS.items()))
    @pytest.mark.parametrize("value", _SCALAR_NON_READINGS)
    def test_a_motor_vector_of_non_readings_is_left_out(self, key: str, field: str, value: Any) -> None:
        """All or nothing: one slot that reports no value costs that vector."""
        snapshot = parse_low_state(_low_state(motor={field: value}), "motor_state_parallel")
        assert snapshot[key] is None

    @pytest.mark.parametrize("field", IMU_STATE_FIELDS)
    @pytest.mark.parametrize("value", _VECTOR_NON_READINGS)
    def test_an_imu_vector_of_non_readings_is_left_out(self, field: str, value: Any) -> None:
        """A buffer that iterates as integers is not a two-element attitude."""
        snapshot = parse_low_state(_low_state(imu={field: value}), "motor_state_parallel")
        assert snapshot["imu"][field] is None

    def test_an_imu_carrying_nothing_readable_reports_every_vector_as_absent(self) -> None:
        """The section stays, so a consumer indexing it still gets an answer."""
        blank = dict.fromkeys(IMU_STATE_FIELDS, _ABSENT)
        snapshot = parse_low_state(_low_state(imu=blank), "motor_state_parallel")
        assert snapshot["imu"] == dict.fromkeys(IMU_STATE_FIELDS, None)

    @pytest.mark.parametrize("key,field", sorted(BATTERY_STATE_FIELDS.items()))
    @pytest.mark.parametrize("value", _SCALAR_NON_READINGS)
    def test_a_battery_field_that_is_no_reading_is_left_out(
        self, sdk: _FakeSdk, key: str, field: str, value: Any
    ) -> None:
        """A defaulted ``0.0`` on ``soc`` is an empty pack; on the rest, a lie."""
        driver = _live_driver(sdk)
        driver._on_battery(_battery(**{field: value}))
        assert (driver._battery or {})[key] is None

    def test_a_reading_still_reaches_the_record_unchanged(self, sdk: _FakeSdk) -> None:
        """Converging on the shared coercer did not widen the refusal."""
        driver = _live_driver(sdk)
        driver._on_battery(_battery())
        assert driver._battery == {"pct": 88.0, "voltage": 48.2, "current": -3.1}
        snapshot = parse_low_state(_low_state(), "motor_state_parallel")
        assert snapshot["joints"] == [0.5] * _WIDTH
        assert snapshot["imu"] == {"rpy": [0.01, 0.02, 0.03], "gyro": [0.1, 0.2, 0.3], "acc": [0.0, 0.0, 9.81]}


class TestOneUnreadableFieldDoesNotCostTheFieldsBesideIt:
    """A per-field read, so a rename costs its field and not the frame.

    Both decoders wrapped the whole record in one ``try``, so ``float()`` raising
    on a single field discarded every field the same message carried - and on the
    ``LowState`` path that meant discarding the positions too, leaving
    ``_last_state`` on the previous frame. The staleness then reads as a dropped
    wire rather than as one renamed field.
    """

    def test_a_bad_imu_field_keeps_the_other_vectors_and_the_positions(self, sdk: _FakeSdk) -> None:
        """The positions are what ``send_action`` holds from, so losing them writes."""
        driver = _live_driver(sdk)
        assert sdk.subscriber is not None
        sdk.subscriber.handler(_low_state(imu={"rpy": "0.05"}))
        state = driver.read_state()
        assert state["joints"] == [0.5] * _WIDTH
        assert state["imu"]["rpy"] is None
        assert state["imu"]["acc"] == [0.0, 0.0, 9.81]

    def test_a_bad_charge_field_keeps_the_pack_voltage_and_current(self, sdk: _FakeSdk) -> None:
        """Voltage and current read fine and are the fallback health signal."""
        driver = _live_driver(sdk)
        driver._on_battery(_battery(soc="n/a"))
        assert driver._battery == {"pct": None, "voltage": 48.2, "current": -3.1}

    def test_a_frame_carrying_no_readable_field_leaves_the_record_alone(self, sdk: _FakeSdk) -> None:
        """An empty record would erase a good reading with an absence."""
        driver = _live_driver(sdk)
        driver._on_battery(_battery())
        driver._on_battery(_battery(**dict.fromkeys(BATTERY_STATE_FIELDS.values(), _ABSENT)))
        assert driver._battery == {"pct": 88.0, "voltage": 48.2, "current": -3.1}


class TestTheHoldPositionIsNeverFabricated:
    """The one row that moves the robot rather than only misreporting it.

    ``joints`` is ``send_action``'s ``held_q``, and ``build_frame`` writes
    ``held_q[slot]`` as the position target of every uncommanded upper-body slot.
    """

    def test_a_frame_whose_positions_do_not_read_is_not_cached(self, sdk: _FakeSdk) -> None:
        """A full-width vector of defaulted zeros passed both existing guards."""
        driver = _unobserved_driver(sdk, enable=False)
        assert sdk.subscriber is not None
        sdk.subscriber.handler(_low_state(motor={"q": _ABSENT}))
        assert driver.read_state() == {}
        assert driver.get_observation() == {}

    @pytest.mark.parametrize("key,field", sorted(MOTOR_STATE_FIELDS.items()))
    def test_one_slot_that_does_not_read_costs_the_whole_vector(self, key: str, field: str) -> None:
        """A vector short one element renumbers every slot after the gap.

        ``held_q`` is indexed *by slot*, and ``send_action`` bounds a caller's
        targets with ``len(held_q)``, so a 22-element vector would both hold the
        wrong joint at each shifted slot and silently narrow the addressable set.
        Dropping the vector reaches the "no frame has arrived" refusal instead.
        """
        snapshot = parse_low_state(_low_state(one_slot=(7, {field: _ABSENT})), "motor_state_parallel")
        assert snapshot[key] is None

    def test_send_action_refuses_rather_than_commanding_a_pose_no_frame_reported(self, sdk: _FakeSdk) -> None:
        """The refusal the driver already spells was unreachable for this frame."""
        driver = _unobserved_driver(sdk)
        assert sdk.subscriber is not None
        sdk.subscriber.handler(_low_state(motor={"q": _ABSENT}))
        result = driver.send_action({"left_shoulder_pitch": 0.1})
        assert result["status"] == "error"
        assert "no LowState frame has arrived yet" in result["content"][0]["text"]
        assert sdk.publisher is not None
        assert sdk.publisher.written == []

    def test_an_observed_frame_is_still_what_the_arms_hold(self, sdk: _FakeSdk) -> None:
        """The reading path is unchanged: a real position is still held."""
        driver = _live_driver(sdk)
        assert driver.send_action({"left_shoulder_pitch": 0.1})["status"] == "success"
        assert sdk.publisher is not None
        held = sdk.publisher.written[-1]
        slot = BOOSTER_JOINT_INDEX["left_elbow_yaw"]
        assert slot in UPPER_BODY_SLOTS
        assert held.motor_cmd_at(slot).q == pytest.approx(round(0.1 * slot, 3))


class TestTheStatusVerbReportsNoChargeAsAbsent:
    """``get_status`` reads ``battery.get("pct")``, which the writer made unreachable."""

    @pytest.mark.parametrize("value", [_ABSENT, True, "n/a"])
    def test_a_charge_field_that_is_no_reading_publishes_no_percentage(self, sdk: _FakeSdk, value: Any) -> None:
        """``0.0`` reads as an empty pack and ``1.0`` as a one-percent one."""
        driver = _live_driver(sdk)
        driver._on_battery(_battery(soc=value))
        envelope = asyncio.run(driver.get_status())["content"][0]["json"]
        assert envelope["battery_pct"] is None

    def test_a_real_charge_still_publishes(self, sdk: _FakeSdk) -> None:
        driver = _live_driver(sdk)
        driver._on_battery(_battery(soc=88.0))
        assert asyncio.run(driver.get_status())["content"][0]["json"]["battery_pct"] == 88.0


class TestTheModuleReadsThroughTheOneOwnerOfThisRule:
    """A derivation, so the typed defaults cannot come back one field at a time."""

    def test_the_driver_resolves_the_shared_coercers(self) -> None:
        assert booster.telemetry_float is telemetry_float
        assert booster.telemetry_float_list is telemetry_float_list

    def test_no_telemetry_read_carries_a_typed_default(self) -> None:
        """``getattr(msg, field, 0.0)`` is the defect; a sentinel name is not."""
        tree = ast.parse(Path(inspect.getfile(booster)).read_text(encoding="utf-8"))
        defaulted = [
            f"line {node.lineno}: {ast.unparse(node)}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 3
            and isinstance(node.args[2], ast.Constant)
            and node.args[2].value is not None
        ]
        assert defaulted == [], (
            "a telemetry field read with a typed default publishes a constant as a reading; "
            f"pass None and coerce it: {defaulted}"
        )
