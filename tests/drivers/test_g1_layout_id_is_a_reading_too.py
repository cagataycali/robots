"""``mode_machine`` is a reading, and it is the one ``_on_lowstate`` sends back.

The four IMU vectors in ``rt/lowstate`` are read through the shared telemetry
coercers, so a field the message does not carry lands ``None`` rather than a
typed default. ``mode_machine`` is decoded in the same method and was still
coerced with a bare ``int()``, which differs from that rule in both directions a
uint8 can go wrong:

* ``int(True)`` is ``1``, and ``1`` is a valid hardware-layout id. A flag on the
  field therefore produced a layout id no message declared, indistinguishable
  from a real one.
* ``int(2.7)`` is ``2``, so a float was truncated into a *different* valid id
  rather than refused.

That value is not only cached, it is echoed on every ``LowCmd_`` this driver
builds (``_build_lowcmd_from_action``, ``_build_zero_torque_lowcmd``), and the
firmware drops a frame whose layout id does not match the one it announced. So
an id built from something that was not a number is a write the robot silently
ignores - the failure mode a typed default has everywhere else in this decoder,
on the one field that leaves the process again.

``telemetry_int`` already refuses both, and a refused reading now leaves
:attr:`~strands_robots.drivers.g1.G1Driver._mode_machine` at its previous value.
That matches :meth:`~strands_robots.drivers.g1.G1Driver._refresh_fsm_id` two
ranges over: the layout id does not change while the robot is powered, so the
last reading that parsed is a better answer than ``None`` - which the motion
gate reads as "lowstate has not delivered yet".

The structural cell here is an AST scan rather than a search for the defect's
own constants, because the constants a *future* field would default to are not
knowable. It covers the whole method and both Unitree drivers, so the Go2 is a
passing control: this is the house rule, not a G1 rule.
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

#: Both Unitree ``_on_lowstate`` decoders, whose own source the scan reads.
_DECODERS: list[Callable[..., None]] = [g1.G1Driver._on_lowstate, go2.Go2Driver._on_lowstate]

#: A healthy ``imu_state``, so these cells vary only ``mode_machine``.
_IMU = types.SimpleNamespace(
    rpy=[0.1, -0.2, 0.3],
    gyroscope=[1.0, 2.0, 3.0],
    accelerometer=[0.0, 0.0, 9.81],
    quaternion=[0.99, 0.01, 0.02, 0.03],
)

#: Values that are not a uint8 layout id. The two ``bool`` spellings are the
#: reason this matters: ``int(True)`` is ``1`` and ``int(False)`` is ``0``, both
#: valid layout ids, so a flag on the field was indistinguishable from a reading.
_NOT_AN_ID: list[tuple[str, Any]] = [
    ("a flag that is true", True),
    ("a flag that is false", False),
    ("a raw buffer", b"\x09"),
    ("not a number", "n/a"),
    ("absent", None),
]

#: Values the shared owner reads as an id, so this change must not refuse them.
#: A float is truncated rather than refused - ``telemetry_int(2.7) == 2`` is the
#: owner's own documented answer, pinned in its rule table, and a decoder that
#: graded it where its sibling does not is the drift #3376 removed.
_STILL_AN_ID: list[tuple[Any, int]] = [(0, 0), (9, 9), (255, 255), ("9", 9), (2.7, 2), (2.0, 2)]


def _driver() -> Any:
    return g1.G1Driver(tool_name="g1", port="1.2.3.4")


def _deliver(driver: Any, mode_machine: Any) -> None:
    driver._on_lowstate(types.SimpleNamespace(imu_state=_IMU, mode_machine=mode_machine))


class TestTheLayoutIdIsARead:
    """A value that is not a number does not become an id the wire carries."""

    @pytest.mark.parametrize(("label", "value"), _NOT_AN_ID, ids=[label for label, _ in _NOT_AN_ID])
    def test_an_unreadable_id_keeps_the_last_one_that_parsed(self, label: str, value: Any) -> None:
        driver = _driver()
        _deliver(driver, 9)
        _deliver(driver, value)
        assert driver._mode_machine == 9, label

    @pytest.mark.parametrize(("label", "value"), _NOT_AN_ID, ids=[label for label, _ in _NOT_AN_ID])
    def test_an_unreadable_id_is_never_learned_in_the_first_place(self, label: str, value: Any) -> None:
        """With no previous reading the gate's refusal stays honest."""
        driver = _driver()
        _deliver(driver, value)
        assert driver._mode_machine is None, label

    @pytest.mark.parametrize(("value", "expected"), _STILL_AN_ID, ids=[repr(v) for v, _ in _STILL_AN_ID])
    def test_an_id_that_parses_is_still_taken(self, value: Any, expected: int) -> None:
        """The control: the coercion did not stop reading real layout ids.

        The float rows are the accepted boundary. ``int()`` truncated them and
        so does the owner, deliberately - refusing them here would make this
        decoder stricter than the Go2 on a value both used to accept.
        """
        driver = _driver()
        _deliver(driver, value)
        assert driver._mode_machine == expected

    def test_a_later_reading_replaces_an_earlier_one(self) -> None:
        """The control for keeping the previous value: it is not a latch."""
        driver = _driver()
        _deliver(driver, 9)
        _deliver(driver, 10)
        assert driver._mode_machine == 10

    def test_the_imu_is_unaffected_either_way(self) -> None:
        """The IMU is decoded independently, which this must not change."""
        driver = _driver()
        _deliver(driver, True)
        assert driver._imu is not None
        assert driver._imu["rpy"] == [0.1, -0.2, 0.3]
        assert driver._imu["quaternion"] == [0.99, 0.01, 0.02, 0.03]


class TestNoLowstateReadCarriesATypedDefault:
    """Derived over the method, so a field added later joins the rule."""

    @pytest.mark.parametrize("decoder", _DECODERS, ids=["g1", "go2"])
    def test_every_field_read_defaults_to_none(self, decoder: Callable[..., None]) -> None:
        defaulted = [
            ast.unparse(node)
            for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(decoder))))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 3
            and not (isinstance(node.args[2], ast.Constant) and node.args[2].value is None)
        ]
        assert defaulted == [], f"a typed default makes an absent field look like a reading: {defaulted}"

    @pytest.mark.parametrize("decoder", _DECODERS, ids=["g1", "go2"])
    def test_the_scan_reaches_the_reads(self, decoder: Callable[..., None]) -> None:
        """Non-vacuity: "no typed defaults" must not mean "no reads found"."""
        reads = [
            node
            for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(decoder))))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
        ]
        assert len(reads) >= 5, f"only {len(reads)} field reads found - the scan is looking in the wrong place"
