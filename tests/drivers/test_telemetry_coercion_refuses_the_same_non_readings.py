"""The Unitree drivers agree on which telemetry fields are readings.

``_on_bms``, ``_on_lowstate``, ``_on_mainboard`` and their siblings decode a DDS
message by reading ``getattr(msg, <field>, None)`` and coercing the result, so
the coercion is what decides whether a field reaches the mesh as a number or as
``None``. That rule was written twice - four functions with identical names in
:mod:`strands_robots.drivers.g1` and :mod:`strands_robots.drivers.go2` - and the
two copies did not agree:

===========================  =======================  =======================
value                        ``g1`` (before)          ``go2`` (before)
===========================  =======================  =======================
``True`` on a scalar field   ``1.0`` / ``1``          ``None``
``bytearray(b"\\x01\\x02")``   ``None``                 ``[1.0, 2.0]``
``memoryview(b"\\x01\\x02")``  ``[1.0, 2.0]``           ``[1.0, 2.0]``
``[True, 2]``                ``[1.0, 2.0]``           ``None``
===========================  =======================  =======================

Each copy guarded a class of value the other let through, and the third row is
the one neither guarded: ``memoryview`` is bytes-like, iterates as integers, and
so decoded a raw buffer into a two-element "quaternion" on both drivers. Every
one of those cells publishes a plausible reading where the field carried none,
which is the exact outcome both decoders' docstrings say the ``None`` default
exists to prevent - a typed default "looks like a reading", and a number derived
from a flag or a buffer looks like one just as well.

The refusal branches were uncovered in both modules, which is how two copies of
one rule came to disagree unnoticed. So the rule now has one owner in
:mod:`strands_robots.drivers.base`, and this suite grades it three ways: the
rule itself as a table, the two decoders end to end, and a derivation over the
driver package that refuses a third private copy.
"""

from __future__ import annotations

import ast
import types
from pathlib import Path
from typing import Any

import pytest

from strands_robots.drivers import base, g1, go2
from strands_robots.drivers.base import (
    telemetry_float,
    telemetry_float_list,
    telemetry_int,
    telemetry_int_list,
)

#: ``(value, float, int, float_list, int_list)`` - the whole rule as one table.
#: The bytes-like rows and the ``bool`` rows are the drift this replaced; the
#: rest pin that converging on the union of the two copies' guards did not
#: widen the refusal into values that *are* readings.
_RULE: list[tuple[Any, float | None, int | None, list[float] | None, list[int] | None]] = [
    (None, None, None, None, None),
    (True, None, None, None, None),
    (False, None, None, None, None),
    (b"\x01\x02", None, None, None, None),
    (bytearray(b"\x01\x02"), None, None, None, None),
    (memoryview(b"\x01\x02"), None, None, None, None),
    ("12", 12.0, 12, None, None),
    ("not a number", None, None, None, None),
    (3.7, 3.7, 3, None, None),
    (object(), None, None, None, None),
    ([1, 2], None, None, [1.0, 2.0], [1, 2]),
    ((1.5, 2.5), None, None, [1.5, 2.5], [1, 2]),
    ([], None, None, [], []),
    ([1, None], None, None, None, None),
    ([True, 2], None, None, None, None),
    ([1, "x"], None, None, None, None),
]


@pytest.mark.parametrize(("value", "as_float", "as_int", "as_floats", "as_ints"), _RULE, ids=repr)
def test_one_rule_decides_what_counts_as_a_reading(
    value: Any,
    as_float: float | None,
    as_int: int | None,
    as_floats: list[float] | None,
    as_ints: list[int] | None,
) -> None:
    """The four coercers answer as the table says, for every class of value."""
    assert telemetry_float(value) == as_float
    assert telemetry_int(value) == as_int
    assert telemetry_float_list(value) == as_floats
    assert telemetry_int_list(value) == as_ints


def test_a_vector_reading_is_a_fresh_list() -> None:
    """The caller may mutate the envelope without racing the DDS thread's write."""
    source = [1.0, 2.0]
    coerced = telemetry_float_list(source)
    assert coerced == source
    assert coerced is not source


def _cache(driver: Any, attr: str) -> dict[str, Any]:
    """The record the decoder wrote, refusing the not-yet-decoded ``None``.

    The caches are declared ``dict | None`` because a driver that has received
    no message on a topic has no record, so reading one without saying that is
    a type error - and an absent record would otherwise read as a passing
    refusal cell.
    """
    record = getattr(driver, attr)
    assert record is not None, f"the decoder wrote no {attr}"
    return record


class TestBothDriversReadThroughTheOneOwner:
    """Identity, not equivalence - a second copy could pass a behaviour table."""

    @pytest.mark.parametrize("driver_module", [g1, go2], ids=["g1", "go2"])
    @pytest.mark.parametrize(
        "name",
        ["telemetry_float", "telemetry_int", "telemetry_float_list", "telemetry_int_list"],
    )
    def test_the_driver_resolves_the_shared_function(self, driver_module: types.ModuleType, name: str) -> None:
        assert getattr(driver_module, name) is getattr(base, name)

    def test_no_driver_keeps_a_private_coercer_of_this_shape(self) -> None:
        """A third driver cannot land a third copy under the old names.

        Derived over the package rather than listed, so a driver added later is
        held to the rule without anyone remembering to extend this file.
        """
        retired = {"_to_float", "_to_int", "_to_float_list", "_to_int_list"}
        found: list[str] = []
        for module_path in sorted(Path(base.__file__).parent.rglob("*.py")):
            tree = ast.parse(module_path.read_text(encoding="utf-8"))
            found += [
                f"{module_path.name}:{node.name}"
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name in retired
            ]
        assert found == [], f"private telemetry coercers still defined: {found}"


class TestTheDecodersRefuseANonReading:
    """The rule through the callback, on the fields each driver publishes."""

    def test_a_flag_on_the_g1_battery_field_is_no_percentage(self) -> None:
        driver = g1.G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_bms(types.SimpleNamespace(soc=True, current=1.5, cycle=False))
        assert _cache(driver, "_battery")["pct"] is None
        assert _cache(driver, "_battery")["cycle"] is None
        assert _cache(driver, "_battery")["current"] == 1.5

    @pytest.mark.parametrize(
        "buffer",
        [bytearray(b"\x01\x02"), memoryview(b"\x01\x02")],
        ids=["bytearray", "memoryview"],
    )
    def test_a_buffer_on_the_g1_mainboard_vectors_is_no_vector(self, buffer: Any) -> None:
        driver = g1.G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_mainboard(types.SimpleNamespace(fan_state=buffer, temperature=buffer, value=[1.0], state=[2]))
        assert _cache(driver, "_mainboard")["fan_state"] is None
        assert _cache(driver, "_mainboard")["temperature"] is None
        assert _cache(driver, "_mainboard")["value"] == [1.0]

    @pytest.mark.parametrize(
        "buffer",
        [bytearray(b"\x01\x02"), memoryview(b"\x01\x02")],
        ids=["bytearray", "memoryview"],
    )
    def test_a_buffer_on_the_go2_imu_is_no_quaternion(self, buffer: Any) -> None:
        driver = go2.Go2Driver()
        imu = types.SimpleNamespace(
            quaternion=buffer, gyroscope=buffer, accelerometer=[0.0, 0.0, 9.8], rpy=[0.0, 0.0, 0.0]
        )
        driver._on_lowstate(types.SimpleNamespace(imu_state=imu, bms_state=None, motor_state=None))
        assert _cache(driver, "_imu")["quaternion"] is None
        assert _cache(driver, "_imu")["gyroscope"] is None
        assert _cache(driver, "_imu")["accelerometer"] == [0.0, 0.0, 9.8]

    def test_a_buffer_on_the_go2_foot_force_is_no_contact_reading(self) -> None:
        driver = go2.Go2Driver()
        driver._on_sportmode(
            types.SimpleNamespace(
                mode=1,
                gait_type=2,
                body_height=0.3,
                position=[0.0, 0.0, 0.3],
                velocity=[0.0, 0.0, 0.0],
                yaw_speed=0.0,
                foot_force=memoryview(b"\x01\x02\x03\x04"),
            )
        )
        assert _cache(driver, "_sport")["foot_force"] is None
        assert _cache(driver, "_sport")["body_height"] == 0.3
