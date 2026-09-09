"""A G1 ``rt/lowstate`` field this driver cannot read costs that field alone.

``LowState_`` carries two independent things the G1 driver needs: the IMU it
publishes on the mesh, and the ``mode_machine`` layout id every motion write
echoes.  :meth:`~strands_robots.drivers.g1.G1Driver._on_lowstate` decoded both
in one ``try``, reading each IMU field with a *typed default* and a truncating
slice::

    "rpy": [float(x) for x in getattr(imu, "rpy", [0.0, 0.0, 0.0])[:3]],
    "quaternion": [float(x) for x in getattr(imu, "quaternion", [1.0, 0.0, 0.0, 0.0])[:4]],

Two things follow, and the second is the one that reaches the actuators:

* A defaulted attitude is not a missing value, it is *level and upright*.
  ``rpy=[0, 0, 0]`` and ``quaternion=[1, 0, 0, 0]`` are well-formed readings,
  published on the mesh IMU topic for as long as the robot runs, for a humanoid
  that may be lying down.  ``accelerometer=[0, 0, 0]`` is free fall, which no
  resting IMU reports.
* ``getattr`` returns ``None`` for a field the message *declares and leaves
  unset*, because the attribute exists and the default never applies.  ``None``
  is not subscriptable, so ``[:3]`` raised, the shared ``except Exception``
  swallowed it, and the ``mode_machine`` decoded **after** the IMU was never
  written.  Its gate then refuses every motion write with "mode_machine unknown
  - lowstate has not delivered yet" for the life of the process, on a robot
  whose lowstate is arriving.

The Go2's ``_on_lowstate`` decodes the same ``imu_state`` from the same vendor
IDL family and states the rule this suite grades: "a firmware revision that
drops one must cost that field, not the whole callback and with it the IMU the
mesh publishes."  It reads every field through the owner in
:mod:`strands_robots.drivers.base` and needs no ``try`` at all.  So the property
is graded over *both* Unitree drivers, with the Go2 as a passing control - a
cell that fails only on the G1 would not show that this is the house rule.

Vector arity is deliberately left ungraded: neither the shared owner nor the Go2
decoder checks it, and grading it on one of the two drivers is how they came to
disagree in the first place (#3376).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types
from collections.abc import Callable
from typing import Any

import pytest

from strands_robots.drivers import g1, go2

#: Every IMU field the G1 decoder publishes, with a healthy reading for it.
_HEALTHY_IMU: dict[str, list[float]] = {
    "rpy": [0.1, -0.2, 0.3],
    "gyroscope": [1.0, 2.0, 3.0],
    "accelerometer": [0.0, 0.0, 9.81],
    "quaternion": [0.99, 0.01, 0.02, 0.03],
}

#: Shapes a field can hold that are not a reading. The first is the reachable
#: one - a field the IDL declares and the firmware leaves unset - and it is the
#: shape that used to take the whole callback with it.
_NOT_A_READING: list[tuple[str, Any]] = [
    ("declared but unset", None),
    ("a scalar where a vector belongs", 0.5),
    ("an element that is not a number", [1.0, "nope", 3.0]),
    ("a raw buffer", memoryview(b"\x01\x02\x03\x04")),
    ("a flag among the elements", [True, 0.0, 0.0]),
]


#: A field a firmware revision renamed or removed, as opposed to one left unset.
_ABSENT = object()


def _imu(**overrides: Any) -> types.SimpleNamespace:
    """A healthy ``imu_state``, with ``overrides`` applied.

    A key mapped to :data:`_ABSENT` is dropped, which is how a firmware
    revision that renames or removes a field arrives.
    """
    fields = dict(_HEALTHY_IMU)
    fields.update(overrides)
    return types.SimpleNamespace(**{k: v for k, v in fields.items() if v is not _ABSENT})


def _g1_lowstate(imu: types.SimpleNamespace | None, mode_machine: Any = 9) -> types.SimpleNamespace:
    return types.SimpleNamespace(imu_state=imu, mode_machine=mode_machine)


def _go2_lowstate(imu: types.SimpleNamespace | None) -> types.SimpleNamespace:
    """The Go2 decodes battery after the IMU, so that is its later cache."""
    return types.SimpleNamespace(
        imu_state=imu,
        bms_state=types.SimpleNamespace(soc=88.0, current=-1.5, cycle=42),
        motor_state=None,
    )


#: Both Unitree ``_on_lowstate`` decoders, as the unbound functions whose own
#: source the structural cells read.
_LOWSTATE_DECODERS: list[Callable[..., None]] = [g1.G1Driver._on_lowstate, go2.Go2Driver._on_lowstate]


def _decoder_tree(decoder: Callable[..., None]) -> ast.Module:
    """The decoder's own source as a tree, dedented so it parses standalone."""
    return ast.parse(textwrap.dedent(inspect.getsource(decoder)))


def _record(driver: Any, attr: str) -> dict[str, Any]:
    """The record the decoder wrote, refusing the not-yet-decoded ``None``.

    An absent record would otherwise read as a passing refusal cell: every
    field of a record that was never written is trivially "not a reading".
    """
    record = getattr(driver, attr)
    assert record is not None, f"the decoder wrote no {attr}"
    return record


class TestTheAttitudeIsNotDefaulted:
    """A field the message does not carry is ``None``, not level and upright."""

    @pytest.mark.parametrize("field", sorted(_HEALTHY_IMU), ids=sorted(_HEALTHY_IMU))
    def test_an_absent_field_is_no_reading(self, field: str) -> None:
        driver = g1.G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lowstate(_g1_lowstate(_imu(**{field: _ABSENT})))
        assert _record(driver, "_imu")[field] is None

    @pytest.mark.parametrize(("label", "value"), _NOT_A_READING, ids=[label for label, _ in _NOT_A_READING])
    @pytest.mark.parametrize("field", sorted(_HEALTHY_IMU), ids=sorted(_HEALTHY_IMU))
    def test_an_unusable_field_is_no_reading(self, field: str, label: str, value: Any) -> None:
        driver = g1.G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lowstate(_g1_lowstate(_imu(**{field: value})))
        assert _record(driver, "_imu")[field] is None, label

    @pytest.mark.parametrize(("label", "value"), _NOT_A_READING, ids=[label for label, _ in _NOT_A_READING])
    @pytest.mark.parametrize("field", sorted(_HEALTHY_IMU), ids=sorted(_HEALTHY_IMU))
    def test_the_other_fields_of_the_same_record_survive(self, field: str, label: str, value: Any) -> None:
        """One unusable field does not discard the attitude that did arrive."""
        driver = g1.G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lowstate(_g1_lowstate(_imu(**{field: value})))
        record = _record(driver, "_imu")
        survivors = {k: v for k, v in _HEALTHY_IMU.items() if k != field}
        assert {k: record[k] for k in survivors} == survivors, label

    def test_a_healthy_message_still_reads_every_field(self) -> None:
        """The control: converging on the owner did not narrow a real reading."""
        driver = g1.G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lowstate(_g1_lowstate(_imu()))
        record = _record(driver, "_imu")
        assert {k: record[k] for k in _HEALTHY_IMU} == _HEALTHY_IMU
        assert isinstance(record["t"], float)


class TestOneUnusableFieldDoesNotCostTheFieldsAfterIt:
    """Graded over both Unitree drivers - the Go2 is the passing control."""

    @pytest.mark.parametrize(("label", "value"), _NOT_A_READING, ids=[label for label, _ in _NOT_A_READING])
    def test_the_g1_still_learns_the_layout_id(self, label: str, value: Any) -> None:
        """``mode_machine`` is decoded after the IMU and gates every write."""
        driver = g1.G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lowstate(_g1_lowstate(_imu(rpy=value), mode_machine=9))
        assert driver._mode_machine == 9, label

    @pytest.mark.parametrize(("label", "value"), _NOT_A_READING, ids=[label for label, _ in _NOT_A_READING])
    def test_the_go2_still_learns_the_battery(self, label: str, value: Any) -> None:
        driver = go2.Go2Driver()
        driver._on_lowstate(_go2_lowstate(_imu(rpy=value)))
        assert _record(driver, "_battery")["pct"] == 88.0, label

    def test_the_motion_gate_no_longer_blames_a_missing_lowstate(self) -> None:
        """The refusal a caller reads is not owed to an unreadable IMU field.

        The next gate (``_fsm_id``, issue #2765) still refuses, which is the
        point: the write path is blocked on the FSM source it documents, not on
        a lowstate that already delivered.
        """
        driver = g1.G1Driver(tool_name="g1", port="1.2.3.4")
        driver._connected = True
        driver._on_lowstate(_g1_lowstate(_imu(rpy=None), mode_machine=9))
        outcome = driver.send_action({"left_shoulder_pitch": 0.1})
        text = " ".join(part.get("text", "") for part in outcome["content"])
        assert "mode_machine unknown" not in text, text
        assert "lowstate has not delivered" not in text, text


class TestTheLayoutIdIsAReadingToo:
    """``mode_machine`` is echoed on the wire, so a flag is not one."""

    @pytest.mark.parametrize(
        ("label", "value"),
        [("a flag", True), ("a raw buffer", b"\x09"), ("not a number", "n/a")],
        ids=["a flag", "a raw buffer", "not a number"],
    )
    def test_an_unusable_layout_id_keeps_the_last_one_that_parsed(self, label: str, value: Any) -> None:
        """Matching ``_refresh_fsm_id``: a refused reading keeps the previous.

        The layout id does not change while the robot is powered, so the last
        reading that parsed is a better answer than ``None`` - which the gate
        reads as "lowstate has not delivered yet".
        """
        driver = g1.G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lowstate(_g1_lowstate(_imu(), mode_machine=9))
        driver._on_lowstate(_g1_lowstate(_imu(), mode_machine=value))
        assert driver._mode_machine == 9, label

    def test_a_layout_id_that_parses_is_still_taken(self) -> None:
        """The control: the coercion did not stop reading a real layout id."""
        driver = g1.G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lowstate(_g1_lowstate(_imu(), mode_machine=9))
        driver._on_lowstate(_g1_lowstate(_imu(), mode_machine=10))
        assert driver._mode_machine == 10


class TestNeitherUnitreeIMUDecoderCarriesATypedDefault:
    """Structural, because a behaviour table cannot see the *next* field.

    Every value read in either ``_on_lowstate`` reaches its coercer through
    ``getattr(obj, name, None)``. A non-``None`` default is what makes an absent
    field indistinguishable from a reading, and it is what a fifth field added
    later would most easily reintroduce.
    """

    @pytest.mark.parametrize("decoder", _LOWSTATE_DECODERS, ids=["g1", "go2"])
    def test_every_field_read_defaults_to_none(self, decoder: Callable[..., None]) -> None:
        tree = _decoder_tree(decoder)
        defaulted = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 3
            and not (isinstance(node.args[2], ast.Constant) and node.args[2].value is None)
        ]
        assert defaulted == [], f"typed defaults on a telemetry read: {defaulted}"

    @pytest.mark.parametrize("decoder", _LOWSTATE_DECODERS, ids=["g1", "go2"])
    def test_the_scan_reaches_the_reads(self, decoder: Callable[..., None]) -> None:
        """Non-vacuity: "no typed defaults" must not mean "no reads found"."""
        tree = _decoder_tree(decoder)
        reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
        ]
        assert len(reads) >= 4, f"only {len(reads)} field reads found - the scan is looking in the wrong place"
