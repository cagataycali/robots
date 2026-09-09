"""An ``IMUState_`` vector the message does not carry caches ``None``, not level.

:meth:`~strands_robots.drivers.g1.G1Driver._on_lowstate` read its four IMU
vectors with typed defaults -- ``[0.0, 0.0, 0.0]`` for ``rpy``, ``gyroscope``
and ``accelerometer``, ``[1.0, 0.0, 0.0, 0.0]`` for ``quaternion``.  A
``getattr`` carrying a default cannot fail, and every one of those constants is
a well-formed reading of a robot that is upright and still: zero rpy is
perfectly level, the identity quaternion is upright, and a zero accelerometer is
free fall, which a standing robot never reports because gravity always lands on
one axis.  A firmware that renames or drops a field therefore published a
constant shaped exactly like telemetry, to ``strands/{peer_id}/imu`` via
:class:`~strands_robots.mesh.sensors.SensorLoopsMixin`, for as long as the robot
ran.

The rule these tests pin was already kept everywhere else. The twin driver
reads the same four names through the shared coercer
(:meth:`~strands_robots.drivers.go2.Go2Driver._on_lowstate`), the same class
reads its battery pack that way one method down (:meth:`G1Driver._on_bms`, so a
renamed field lands ``None`` rather than a plausible zero), the sibling humanoid
states it outright in
:func:`~strands_robots.drivers.booster.parse_low_state` ("a snapshot that
reports a zeroed IMU the robot never sent is worse than one that reports none"),
and the consuming verb ``g1_imu`` documented ``None`` for every field while the
writer made that outcome unreachable.
"""

from __future__ import annotations

import inspect
import types
from typing import Any

import pytest

from strands_robots.drivers import g1, go2
from strands_robots.drivers.g1 import G1Driver
from strands_robots.tools.g1.g1_imu import g1_imu

# The four IMU vectors both Unitree drivers cache, read off their own writers.
IMU_FIELDS = ("rpy", "gyroscope", "accelerometer", "quaternion")


def _class_name(module: Any) -> str:
    """The driver class a Unitree module defines, so the pin names neither twice."""
    return "G1Driver" if module.__name__.endswith("g1") else "Go2Driver"


# One well-formed frame: the robot is rolled and yawed, gravity is on Z. Every
# case below starts from this and removes or corrupts exactly one field, so a
# fabricated value is visible against three real neighbours.
WIRED = {
    "rpy": [0.05, -0.02, 1.3],
    "gyroscope": [0.01, 0.02, 0.03],
    "accelerometer": [0.1, 0.0, 9.79],
    "quaternion": [0.79, 0.0, 0.0, 0.61],
}


def _driver() -> G1Driver:
    return G1Driver(tool_name="g1", port="192.168.1.172")


def _decode(**fields: Any) -> dict[str, Any]:
    """Run one ``rt/lowstate`` frame through the driver and return ``_imu``.

    Args:
        **fields: The ``imu_state`` attributes to publish. A name that is
            absent from this mapping is absent from the message, which is what
            a renamed or dropped IDL field looks like on the wire.

    Returns:
        The cached record, ``t`` removed so cases compare by value.
    """
    driver = _driver()
    driver._on_lowstate(types.SimpleNamespace(imu_state=types.SimpleNamespace(**fields), mode_machine=4))
    assert driver._imu is not None, "a LowState carrying imu_state must cache a record"
    record = dict(driver._imu)
    record.pop("t")
    return record


def test_a_fully_populated_frame_is_cached_verbatim() -> None:
    """The good path is untouched: every declared vector round-trips by value."""
    assert _decode(**WIRED) == WIRED


@pytest.mark.parametrize("missing", sorted(IMU_FIELDS))
def test_an_absent_vector_caches_none_and_leaves_its_siblings_alone(missing: str) -> None:
    """A field the message does not carry is ``None``, not a level reading.

    The three fields that *were* on the wire keep their values, so the record
    reports precisely what arrived rather than being discarded wholesale.
    """
    record = _decode(**{k: v for k, v in WIRED.items() if k != missing})
    assert record[missing] is None
    assert {k: v for k, v in record.items() if k != missing} == {k: v for k, v in WIRED.items() if k != missing}


def test_an_imu_state_carrying_no_vectors_reports_no_readings() -> None:
    """An ``imu_state`` with none of the four fields fabricates nothing.

    Pre-fix this was the worst cell: a complete, plausible record of a robot
    standing perfectly level and upright while in free fall.
    """
    assert _decode() == dict.fromkeys(IMU_FIELDS)


@pytest.mark.parametrize("value", ["0.05", b"\x01\x02\x03", 0.05, None], ids=["str", "bytes", "scalar", "none"])
def test_a_non_vector_field_costs_only_itself(value: Any) -> None:
    """One unreadable vector reports ``None``; the other three survive.

    Pre-fix a string ``rpy`` raised inside the comprehension, the driver's
    ``except`` swallowed it, and the whole assignment was abandoned -- the
    three good readings in the same frame were lost with it.
    """
    record = _decode(**{**WIRED, "rpy": value})
    assert record["rpy"] is None
    assert {k: record[k] for k in ("gyroscope", "accelerometer", "quaternion")} == {
        k: WIRED[k] for k in ("gyroscope", "accelerometer", "quaternion")
    }


def test_the_imu_verb_reports_the_unread_field_as_none() -> None:
    """``g1_imu`` renders the ``None`` its own docstring documents.

    The verb documented every field as "or ``None``" while the writer made
    that outcome unreachable for a driver that had received a frame.
    """
    driver = _driver()
    driver._on_lowstate(
        types.SimpleNamespace(
            imu_state=types.SimpleNamespace(**{k: v for k, v in WIRED.items() if k != "rpy"}),
            mode_machine=4,
        )
    )
    envelope = g1_imu(driver)
    assert envelope["present"] is True
    assert envelope["rpy"] is None
    assert envelope["accelerometer"] == WIRED["accelerometer"]


def test_both_unitree_drivers_read_their_imu_through_the_same_rule() -> None:
    """The G1 and the Go2 coerce the same four names with the same function.

    :meth:`Go2Driver._on_lowstate` already read its IMU through
    :func:`~strands_robots.drivers.base.telemetry_float_list`; the G1 was the
    one Unitree telemetry read still carrying typed defaults. Deriving the call
    from both sources means a future driver that hand-rolls the coercion again
    fails here rather than drifting quietly.
    """
    for module in (g1, go2):
        body = inspect.getsource(module.__dict__[_class_name(module)]._on_lowstate)
        for field in IMU_FIELDS:
            assert f'telemetry_float_list(getattr(imu, "{field}", None))' in body, (
                f"{module.__name__}._on_lowstate must read {field} through the shared coercer"
            )


def test_no_imu_read_carries_a_typed_default() -> None:
    """A default reintroduced on any of the four reads fails here.

    The values are the defect's own constants, so this cell is a derivation
    over the writer rather than a restatement of it.
    """
    source = inspect.getsource(G1Driver._on_lowstate)
    body = source.split('"""', 2)[2]  # skip the docstring, which quotes them
    for default in ("[0.0, 0.0, 0.0]", "[1.0, 0.0, 0.0, 0.0]"):
        assert default not in body, f"{default} is a reading of a robot that is fine"
