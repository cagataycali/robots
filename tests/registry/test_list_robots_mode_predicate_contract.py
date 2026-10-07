"""``list_robots`` mode filters must match the public predicates they document.

``list_robots(mode=...)`` documents each filter by naming the query predicate a
caller would use to reproduce it: ``mode="sim"`` mirrors :func:`has_sim` and
``mode="real"`` is the union of :func:`has_hardware` (declared hardware block)
and :func:`~strands_robots.drivers.get_native_driver_class` (native driver
registered through :func:`~strands_robots.drivers.register_native_driver`). If
the docstring names a predicate that does not exist in the registry API (it
once referenced a phantom ``has_real`` that was never a function -- the real
predicate is ``has_hardware``), a reader who trusts the docstring reaches for
an attribute that raises ``AttributeError``.

These tests pin two things so the documentation cannot drift ahead of the
implementation again:

1. Every ``has_*`` predicate the ``list_robots`` docstring cites resolves to a
   real public callable in ``strands_robots.registry``.
2. The mode filters actually correspond to those predicates row-for-row:
   ``sim`` <-> :func:`has_sim`, ``real`` <-> union of ``has_hardware`` and
   native-driver registration (matching
   :func:`~strands_robots.drivers.list_driver_coverage`), ``both`` is their
   intersection with ``has_sim``, and ``all`` is every registered robot.
"""

from __future__ import annotations

import re

import pytest

import strands_robots.registry as registry_pkg
from strands_robots.drivers.registry import get_native_driver_class
from strands_robots.registry.robots import (
    LIST_ROBOTS_MODES,
    has_hardware,
    has_sim,
    list_robots,
)

# Backtick-quoted ``has_<name>`` tokens cited in the ``list_robots`` docstring.
_PREDICATE_TOKEN_RE = re.compile(r"``(has_[a-z_]+)``")


def _cited_predicates() -> set[str]:
    doc = list_robots.__doc__ or ""
    # Only inspect the Args block (the mode bullets); the Returns block lists
    # output-dict KEY names (has_sim, has_real), which are payload, not APIs.
    args_block = doc.split("Returns:", 1)[0]
    return set(_PREDICATE_TOKEN_RE.findall(args_block))


def _is_real(name: str) -> bool:
    """The union ``mode='real'`` selects on: a declared hardware block OR a
    registered native driver. Mirrors
    :func:`~strands_robots.drivers.list_driver_coverage`'s non-empty rows."""
    return has_hardware(name) or get_native_driver_class(name) is not None


def test_docstring_cites_at_least_the_two_predicates() -> None:
    """Guard the parser: the docstring must cite the sim + hardware predicates."""
    cited = _cited_predicates()
    assert "has_sim" in cited
    assert "has_hardware" in cited


def test_cited_predicates_are_real_registry_callables() -> None:
    """Every ``has_*`` predicate the docstring names must be a real public API.

    Fails on the pre-fix docstring, which cited a phantom ``has_real``.
    """
    for name in sorted(_cited_predicates()):
        obj = getattr(registry_pkg, name, None)
        assert callable(obj), f"list_robots docstring cites ``{name}`` but strands_robots.registry has no such callable"


def test_real_mode_matches_has_hardware_or_native_driver() -> None:
    """``mode='real'`` selects exactly the robots a user can drive for real:
    the union of ``has_hardware`` (declared) and ``get_native_driver_class``
    (registered). The union, not either half alone, is the row set
    :func:`~strands_robots.drivers.list_driver_coverage` surfaces as non-empty.
    """
    real_names = {r["name"] for r in list_robots(mode="real")}
    expected = {r["name"] for r in list_robots(mode="all") if _is_real(r["name"])}
    assert real_names == expected
    assert real_names, "expected at least one real-drivable robot in the registry"


def test_real_mode_includes_native_driver_only_robots() -> None:
    """Pin: a robot with no ``hardware`` block but a registered native driver
    (``URDriver``, ``FrankaDriver``, ``SpotDriver``, ``XArmDriver``,
    ``KukaDriver``, ``KinovaDriver``, ``StretchDriver``, ``RBY1Driver``,
    ``Go2Driver`` for H1) is listed in ``mode='real'``. ``has_hardware``
    alone would hide 25 of the 49 drivable robots the registry can reach --
    every Universal Robots arm, Franka Panda, Spot, Stretch, Unitree H1,
    xArm7 -- a silent-404 against ``Robot(name, mode='real')`` which builds
    the driver cleanly on each of them.
    """
    real_names = {r["name"] for r in list_robots(mode="real")}
    # Pick a representative subset: a UR arm, Franka, Spot, Stretch,
    # Unitree H1, xArm. All have ``has_hardware`` False yet build drivers.
    for native_only in ("ur5e", "panda", "spot", "stretch", "unitree_h1", "xarm7", "kuka_iiwa", "kinova_gen3"):
        assert has_hardware(native_only) is False, (
            f"{native_only} unexpectedly declares a hardware block; "
            "this test watches robots that have ONLY a native driver"
        )
        assert get_native_driver_class(native_only) is not None, (
            f"{native_only} should have a native driver registered"
        )
        assert native_only in real_names, (
            f"{native_only} has a native driver but ``mode='real'`` hid it "
            "-- regressing the ``has_hardware``-only pre-fix contract"
        )


def test_sim_mode_matches_has_sim() -> None:
    """``mode='sim'`` selects exactly the robots for which has_sim() is True."""
    sim_names = {r["name"] for r in list_robots(mode="sim")}
    expected = {r["name"] for r in list_robots(mode="all") if has_sim(r["name"])}
    assert sim_names == expected
    assert sim_names, "expected at least one sim-capable robot in the registry"


def test_both_mode_is_the_intersection() -> None:
    """``mode='both'`` selects robots that satisfy has_sim() AND _is_real()."""
    both_names = {r["name"] for r in list_robots(mode="both")}
    expected = {r["name"] for r in list_robots(mode="all") if has_sim(r["name"]) and _is_real(r["name"])}
    assert both_names == expected


def test_all_mode_is_every_registered_robot() -> None:
    """``mode='all'`` is the unfiltered set (a superset of sim, real, and both)."""
    all_names = {r["name"] for r in list_robots(mode="all")}
    for sub in ("sim", "real", "both"):
        assert {r["name"] for r in list_robots(mode=sub)} <= all_names


@pytest.mark.parametrize("bad_mode", ["hardware", "SIM", "", "any", "simulation"])
def test_unknown_mode_raises_valueerror(bad_mode: str) -> None:
    """An unrecognized mode fails loudly rather than returning an unfiltered list."""
    assert bad_mode not in LIST_ROBOTS_MODES
    with pytest.raises(ValueError, match="Unknown list_robots mode"):
        list_robots(mode=bad_mode)
