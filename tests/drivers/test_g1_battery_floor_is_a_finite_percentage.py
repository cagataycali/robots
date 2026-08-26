"""The G1's battery floor is held to a numeric domain before it is stored.

``battery_floor_pct`` is the only value :meth:`G1Driver._check_motion_gates`
compares a live reading against, and it was the one constructor parameter with
no domain: a bare ``float()`` coerced it and stored the result.  ``float("nan")``
survives that coercion, and ``battery_pct < nan`` is False for every reading, so
the driver stored a floor it advertised and enforced nowhere -- the gate opened
on a critically low pack while :meth:`get_status` still reported a floor.

The two sibling numbers this driver takes (``duration`` and ``n_steps``, on
:meth:`run_policy`) both go through a shared domain from
:mod:`strands_robots.utils`.  This file grades the floor against the same shared
domain, and grades the values that are *not* refused just as hard: a floor of
``0.0`` or ``-10.0`` never trips, but that is what its arithmetic says and a
caller can mean it, so the fix must leave those alone.

Deliberately out of scope: whether a percentage should be bounded to
``[0, 100]``.  ``500.0`` is accepted here and refuses every pack, and ``-10.0``
is accepted and refuses none; both behave exactly as the comparison reads, so
narrowing them is a policy choice about what a percentage may mean rather than a
correction, and it is left for whoever wants to make it.
"""

from __future__ import annotations

import ast
import inspect
import math
from typing import Any

import numpy as np
import pytest

import strands_robots.drivers.g1 as g1_module
from strands_robots.drivers.g1 import G1Driver
from strands_robots.utils import finite_number_error

# The reading the gate compares the floor against is a state-of-charge
# percentage decoded from ``rt/lf/bmsstate``; 3.0% is a pack that must not be
# asked to hold a 1.3 m biped upright.
_CRITICAL_PCT = 3.0

# Spellings a bare ``float()`` either accepted silently or turned into a bare
# builtin exception escaping the constructor's documented contract.
_UNUSABLE: list[tuple[str, Any]] = [
    ("nan", float("nan")),
    ("inf", float("inf")),
    ("negative-inf", float("-inf")),
    ("nan-as-a-string", "nan"),
    ("number-as-a-string", "20"),
    ("True", True),
    ("False", False),
    ("None", None),
    ("a-list", [20.0]),
    ("past-float64", 10**400),
]

# Every one of these behaves exactly as ``battery_pct < floor`` reads, so the
# fix must not touch them.  ``0.0``/``-10.0`` never trip; ``500.0`` always does.
_USABLE: list[tuple[str, Any]] = [
    ("the-default", 15.0),
    ("zero", 0.0),
    ("negative", -10.0),
    ("full", 100.0),
    ("above-a-hundred", 500.0),
    ("numpy-float32", np.float32(18.5)),
    ("numpy-float64", np.float64(18.5)),
    ("an-int", 20),
]


def _driver(**kwargs: Any) -> G1Driver:
    return G1Driver(tool_name="g1", port="1.2.3.4", **kwargs)


def _open_gate_on_a_critical_pack(driver: G1Driver) -> dict[str, Any] | None:
    """Put the driver one step from a write, with only the battery in doubt."""
    driver._connected = True
    driver._mode_machine = 5  # the uint8 layout id lowstate really delivers
    driver._fsm_id = 500  # inside HANDSHAKE_FSMS, so the FSM is not the reason
    driver._battery = {"pct": _CRITICAL_PCT}
    return driver._check_motion_gates("arm")


class TestANonFiniteFloorIsRefused:
    """The regression: a floor that cannot be enforced is refused up front."""

    @pytest.mark.parametrize("value", [v for _, v in _UNUSABLE], ids=[i for i, _ in _UNUSABLE])
    def test_construction_refuses_it(self, value: Any) -> None:
        with pytest.raises(ValueError, match="battery_floor_pct"):
            _driver(battery_floor_pct=value)

    def test_a_nan_floor_can_no_longer_open_the_gate(self) -> None:
        """The money case: ``nan`` used to disable the floor silently.

        Every ``battery_pct < nan`` is False, so the comparison passed a
        critically low pack while the stored floor still read ``nan``.
        """
        with pytest.raises(ValueError):
            _driver(battery_floor_pct=float("nan"))

    def test_the_refusal_names_the_parameter_and_the_value(self) -> None:
        with pytest.raises(ValueError) as caught:
            _driver(battery_floor_pct=float("nan"))
        text = str(caught.value)
        assert "G1Driver" in text  # the surface that received it
        assert "battery_floor_pct" in text  # the caller's own parameter name
        assert "nan" in text  # the value, so a log names what was passed


class TestAFloorThatBehavesAsItReadsIsUntouched:
    """Over-reach controls: every value the comparison can honour still works."""

    @pytest.mark.parametrize("value", [v for _, v in _USABLE], ids=[i for i, _ in _USABLE])
    def test_construction_accepts_it(self, value: Any) -> None:
        driver = _driver(battery_floor_pct=value)
        assert math.isclose(driver._battery_floor_pct, float(value))

    def test_the_default_floor_still_refuses_a_critical_pack(self) -> None:
        refusal = _open_gate_on_a_critical_pack(_driver())
        assert refusal is not None
        assert "under floor" in refusal["content"][0]["text"]

    def test_a_floor_of_zero_still_admits_a_critical_pack(self) -> None:
        """``0.0`` means "no floor" and is left meaning that.

        This is the value a stricter domain would have refused, so it is the
        control that keeps the fix from becoming a policy change.
        """
        assert _open_gate_on_a_critical_pack(_driver(battery_floor_pct=0.0)) is None

    def test_a_healthy_pack_is_admitted_under_the_default_floor(self) -> None:
        driver = _driver()
        driver._connected = True
        driver._mode_machine = 5
        driver._fsm_id = 500
        driver._battery = {"pct": 95.0}
        assert driver._check_motion_gates("arm") is None

    def test_a_driver_with_no_battery_reading_is_not_refused_for_the_battery(self) -> None:
        """An unread pack is not a low pack - the gate skips the comparison."""
        driver = _driver()
        driver._connected = True
        driver._mode_machine = 5
        driver._fsm_id = 500
        driver._battery = None
        assert driver._check_motion_gates("arm") is None


class TestThePremisesTheseCellsRestOn:
    """What must be true for the cells above to mean what they claim."""

    @pytest.mark.parametrize("value", [v for _, v in _UNUSABLE], ids=[i for i, _ in _UNUSABLE])
    def test_the_shared_domain_refuses_it(self, value: Any) -> None:
        assert finite_number_error(value, "battery_floor_pct", "G1Driver") is not None

    @pytest.mark.parametrize("value", [v for _, v in _USABLE], ids=[i for i, _ in _USABLE])
    def test_the_shared_domain_accepts_it(self, value: Any) -> None:
        assert finite_number_error(value, "battery_floor_pct", "G1Driver") is None

    def test_a_nan_floor_would_pass_the_comparison_the_gate_makes(self) -> None:
        """Why ``nan`` is the dangerous spelling rather than merely odd.

        The gate is ``battery_pct < floor``.  With a ``nan`` floor that is
        False for every reading, so the pack is never the reason - which is
        indistinguishable from a healthy pack.
        """
        assert (_CRITICAL_PCT < float("nan")) is False
        assert (0.0 < float("nan")) is False

    def test_the_battery_reading_is_a_percentage(self) -> None:
        """The floor is compared against a state-of-charge percentage."""
        source = inspect.getsource(G1Driver._on_bms)
        assert '"pct"' in source
        assert "soc" in source  # the BMS field it is decoded from


class TestTheGuardIsWiredWhereItHasToBe:
    """Structural: placement and single-sourcing, read off the module."""

    @staticmethod
    def _init_ast() -> ast.FunctionDef:
        tree = ast.parse(inspect.getsource(g1_module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "G1Driver":
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                        return child
        raise AssertionError("G1Driver.__init__ not found")

    def test_the_guard_precedes_the_coercion_that_would_accept_nan(self) -> None:
        """A guard after ``float()`` would judge a value already stored."""
        init = self._init_ast()
        guards = [
            node.lineno
            for node in ast.walk(init)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "finite_number_error"
        ]
        stores = [
            node.lineno
            for node in ast.walk(init)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "_battery_floor_pct" for t in node.targets)
        ]
        assert len(guards) == 1, f"expected one domain call, found {guards}"
        assert len(stores) == 1, f"expected one store, found {stores}"
        assert guards[0] < stores[0]

    def test_the_domain_is_the_shared_one_rather_than_a_local_copy(self) -> None:
        """A re-implemented check here could drift from ``duration``'s."""
        source = inspect.getsource(G1Driver.__init__)
        assert "finite_number_error(" in source
        assert "math.isnan" not in source
        assert "isfinite" not in source

    def test_the_constructor_documents_the_refusal_it_makes(self) -> None:
        """The tree-wide grader requires it; assert it here too."""
        doc = G1Driver.__init__.__doc__ or ""
        assert "Raises:" in doc
        assert "ValueError" in doc
        assert "battery_floor_pct" in doc
