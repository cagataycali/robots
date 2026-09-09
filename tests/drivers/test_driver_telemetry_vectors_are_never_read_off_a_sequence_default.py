"""A driver reads a telemetry vector off ``None``, never off a sequence default.

``getattr(msg, field, <sequence literal>)`` cannot fail, and the literal it
yields has the exact shape of a reading - so a firmware layout that does not
declare that name publishes a constant shaped like telemetry for as long as the
robot runs.  :func:`~strands_robots.drivers.base.telemetry_float_list` and its
integer sibling are the one owner of that decision: a field the message does not
carry, or carries as something that is not a vector of numbers, lands ``None``.

This pins the Booster T1's ``LowState`` snapshot - the last read in the package
still taken off a sequence default - and then derives the rule over the whole
``drivers/`` package, so the next vector decode cannot reintroduce one.
"""

from __future__ import annotations

import ast
import pathlib
import types
from typing import Any

import pytest

import strands_robots.drivers as drivers_package
from strands_robots.drivers.booster import parse_low_state

#: A humanoid mid-fall.  Every value is distinguishable from an empty vector and
#: from what a bytes-like field would iterate into.
_IMU: dict[str, Any] = {
    "rpy": [0.30, -0.10, 1.57],
    "gyro": [0.01, 0.02, 0.03],
    "acc": [0.10, 0.20, 9.81],
}

#: Values that are not a vector reading, whatever they iterate as.  The
#: bytes-like members matter most: a raw SDK buffer iterates into small plausible
#: numbers, so it used to decode into a short vector of the right dtype.
_NON_READINGS: tuple[Any, ...] = (
    memoryview(b"\x01\x02"),
    b"\x01\x02",
    "0.1,0.2,0.3",
    [0.30, None, 1.57],
    4.0,
)


def _snapshot(imu: dict[str, Any]) -> dict[str, Any]:
    """Read one ``LowState`` whose ``imu_state`` declares exactly ``imu``'s keys.

    Args:
        imu: The IMU fields the message carries.  A key absent here is a field
            the firmware never declared.

    Returns:
        The ``"imu"`` sub-record of the parsed snapshot.
    """
    state = types.SimpleNamespace(
        motor_state_parallel=[types.SimpleNamespace(q=0.0, dq=0.0, tau_est=0.0, temperature=0.0)],
        imu_state=types.SimpleNamespace(**imu),
    )
    record: dict[str, Any] = parse_low_state(state, "motor_state_parallel")["imu"]
    return record


class TestAVectorTheLowStateDoesNotCarryIsAbsentFromTheSnapshot:
    """The docstring's own promise: it reports none rather than a zeroed IMU."""

    @pytest.mark.parametrize("field", sorted(_IMU))
    def test_a_dropped_field_is_none(self, field: str) -> None:
        """Not ``[]``, which is a present key a consumer indexes into."""
        record = _snapshot({k: v for k, v in _IMU.items() if k != field})
        assert record[field] is None
        assert {k: record[k] for k in _IMU if k != field} == {k: v for k, v in _IMU.items() if k != field}

    def test_a_well_formed_message_is_unchanged(self) -> None:
        """The fix costs the good path nothing."""
        assert _snapshot(dict(_IMU)) == _IMU


class TestAValueThatIsNoReadingIsNotDecodedIntoAPlausibleVector:
    """Bytes-like and part-numeric values iterate; that does not make them readings."""

    @pytest.mark.parametrize("value", _NON_READINGS, ids=lambda v: type(v).__name__)
    def test_the_field_reads_as_absent(self, value: Any) -> None:
        assert _snapshot({**_IMU, "acc": value})["acc"] is None

    @pytest.mark.parametrize("value", _NON_READINGS, ids=lambda v: type(v).__name__)
    def test_it_costs_only_its_own_field(self, value: Any) -> None:
        """A raise from inside one comprehension used to cost the whole snapshot.

        ``float(None)`` raised out of ``parse_low_state`` and past its caller, so
        one malformed element discarded the joint, velocity, torque and
        temperature arrays read off the same message.
        """
        record = _snapshot({**_IMU, "acc": value})
        assert record["rpy"] == _IMU["rpy"]
        assert record["gyro"] == _IMU["gyro"]


def test_no_driver_reads_a_telemetry_field_from_a_sequence_default() -> None:
    """Derived over the package rather than named per driver.

    A driver that adds a vector decode inherits the rule without anyone
    remembering to extend a list of module names.
    """
    sequences = (ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp)
    root = pathlib.Path(str(drivers_package.__file__)).parent
    scanned = sorted(root.rglob("*.py"))
    assert len(scanned) > 10, f"read the installed package, not a stale path: {root}"
    offenders = [
        f"{path.relative_to(root)}:{node.lineno}: {ast.unparse(node)}"
        for path in scanned
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 3
            and isinstance(node.args[2], sequences)
        )
    ]
    assert offenders == []
