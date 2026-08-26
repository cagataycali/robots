# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: a backend must not permute a parameter it shares with a sibling.

``create_simulation(backend=...)`` is the backend-agnostic entry point, so the
same ``sim`` variable is a different class depending on one string. A method two
backends both implement therefore has two signatures, and nothing in the type
system relates them: ``add_camera`` is not on the :class:`SimEngine` ABC at all
(``__abstractmethods__`` does not list it), and ``randomize`` is on it only as a
``**kwargs`` sink whose docstring says "Concrete backends define their own
parameter signatures".

Two of them had permuted a parameter they share with both siblings, so one
positional call meant two things. Measured on this tree before the fix::

    add_camera("wrist", [0.1, 0.0, 0.05], [0.3, 0.0, 0.0], 100, 200, 90)

    mujoco   fov=100  width=200  height=90
    newton   fov=100  width=200  height=90
    isaac    fov=90   width=100  height=200      <- a 100x200 view at fov 90

Every one of those six values is inside its own domain on both readings, so
nothing refuses the call and nothing warns: the caller gets a different camera,
and finds out from the pixels. ``randomize`` had the same shape, with the three
range parameters in reverse order between MuJoCo (``color_range``,
``friction_range``, ``mass_range``) and Newton (``mass_range``,
``friction_range``, ``color_range``).

Both divergences contradicted a parity claim the code and docs already made.
Newton's ``randomize`` docstring said "Keyword names and defaults mirror the
MuJoCo backend so randomization code transfers across backends unchanged" - the
premise (names and defaults) is narrower than the conclusion (code transfers),
and the missing third term was the order. ``docs/simulation/newton.md`` states
the camera order as ``add_camera(name, position, target, fov=60, width, height,
parent_body=None)`` and calls it "matching the MuJoCo signature", and
``docs/simulation/domain-randomization.md`` lists the three ranges in MuJoCo's
order. Those two documented orders are what this module grades the backends
against, so the pages and the signatures cannot drift apart.

The rule is that a backend may *add* parameters but must not *permute* the ones
it shares. That is deliberately weaker than "a shared parameter sits at the same
positional index" - see :class:`TestWhatThisDoesNotDecide`, which measures the
methods where the weaker rule holds and the stronger one does not, and pins the
distinction rather than leaving it to be rediscovered.

The universe is derived from
:data:`~strands_robots.simulation.factory._BUILTIN_BACKENDS`, the table
``create_simulation`` resolves, so a fourth backend is held to the rule the hour
it lands rather than when someone remembers to extend a list. Nothing here needs
Isaac Sim, Newton, a GPU or GL: only the signatures are read, and all three
engine modules import on a host with neither optional runtime installed.
"""

from __future__ import annotations

import importlib
import inspect
import itertools
import pathlib
import re
from collections.abc import Callable
from typing import Any

import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.factory import _BUILTIN_BACKENDS

#: One positional camera call whose every argument is valid under either reading.
#: ``fov=100`` and ``fov=90`` are both inside ``(0, 180)`` and ``100``/``200``/``90``
#: are all positive integers, which is why the permutation was silent rather than
#: refused.
_CAMERA_CALL: tuple[Any, ...] = ("wrist", [0.1, 0.0, 0.05], [0.3, 0.0, 0.0], 100, 200, 90)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _engines() -> dict[str, Any]:
    """Resolve every shipped backend name to its engine class.

    Returns:
        A mapping of backend name to engine class, derived from the table
        ``create_simulation`` resolves rather than from a list in this file.
    The values are typed ``Any`` rather than ``type`` for the reason this module
    exists. ``add_camera`` is declared on no base class, so mypy reports
    ``"type" has no attribute "add_camera"`` against a bare ``type`` and
    ``"type[SimEngine]" has no attribute "add_camera"`` once ``issubclass``
    narrows it - the checker cannot see the surface these tests grade either.
    """
    out: dict[str, Any] = {}
    for name, (module_path, class_name) in sorted(_BUILTIN_BACKENDS.items()):
        out[name] = getattr(importlib.import_module(module_path), class_name)
    return out


def _positional_parameters(method: Callable[..., Any]) -> list[str]:
    """The parameter names a caller can supply positionally, in order.

    ``self`` is dropped, and keyword-only parameters are excluded because a
    keyword-only name cannot be reached positionally and so cannot be permuted
    in the sense this module grades.

    Args:
        method: The function to read.

    Returns:
        The positional parameter names, in signature order.
    """
    return [
        p.name
        for p in inspect.signature(method).parameters.values()
        if p.name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]


def permuted_shared_parameters(left: Callable[..., Any], right: Callable[..., Any]) -> tuple[str, ...]:
    """The parameters two signatures share but order differently.

    A parameter only one side declares is ignored: a backend adding its own
    loader or option is allowed, and shifts nothing about the *relative* order
    of the names both sides carry.

    Args:
        left: One implementation.
        right: The other implementation.

    Returns:
        The shared parameter names whose position within the shared sequence
        differs between the two, sorted. Empty when the shared parameters are
        in the same relative order.
    """
    left_params = _positional_parameters(left)
    right_params = _positional_parameters(right)
    shared = set(left_params) & set(right_params)
    left_order = [p for p in left_params if p in shared]
    right_order = [p for p in right_params if p in shared]
    return tuple(sorted(name for name in shared if left_order.index(name) != right_order.index(name)))


def _shared_methods() -> list[tuple[str, str, str]]:
    """Every ``(method, backend_a, backend_b)`` a rule can be asked about.

    Returns:
        One row per public method implemented by a pair of backends.
    """
    engines = _engines()
    owners: dict[str, list[str]] = {}
    for backend, engine in engines.items():
        for name, _ in inspect.getmembers(engine, inspect.isfunction):
            if not name.startswith("_"):
                owners.setdefault(name, []).append(backend)
    rows: list[tuple[str, str, str]] = []
    for method, backends in sorted(owners.items()):
        for left, right in itertools.combinations(sorted(backends), 2):
            rows.append((method, left, right))
    return rows


def _order_in(window: str, names: tuple[str, ...], pattern: str) -> list[str]:
    """``names`` sorted by where ``pattern`` first matches each of them in ``window``.

    Args:
        window: The documentation text to read.
        names: The parameter names to locate.
        pattern: A ``str.format`` template producing the regex for one name.

    Returns:
        ``names`` in the order the window mentions them.
    """
    found: list[tuple[int, str]] = []
    for name in names:
        match = re.search(pattern.format(re.escape(name)), window)
        assert match is not None, f"{name} not matched by {pattern!r} in:\n{window}"
        found.append((match.start(), name))
    return [name for _, name in sorted(found)]


def _documented_signature_order(page: str, marker: str, names: tuple[str, ...]) -> list[str]:
    """The order ``names`` appear in, in the documented signature line holding ``marker``.

    Reads a single line so a later paragraph naming the same parameters in prose
    cannot decide the answer.

    Args:
        page: Path relative to the repository root.
        marker: A substring selecting the signature line.
        names: The parameter names to locate.

    Returns:
        ``names`` in documented order.
    """
    for line in (_REPO_ROOT / page).read_text(encoding="utf-8").splitlines():
        if marker in line:
            return _order_in(line, names, r"\b{}\b")
    raise AssertionError(f"{marker!r} not found in {page}")


def _documented_call_order(page: str, marker: str, names: tuple[str, ...]) -> list[str]:
    """The order ``names`` are *assigned* in, in the first documented call after ``marker``.

    Matches ``name=`` rather than a bare mention: the randomization page
    annotates ``randomize_physics`` with a comment naming ``mass_range`` and
    ``friction_range`` before either is passed, so a bare-mention scan reads that
    comment as the documented order.

    Args:
        page: Path relative to the repository root.
        marker: A substring selecting the call.
        names: The parameter names to locate.

    Returns:
        ``names`` in documented order.
    """
    text = (_REPO_ROOT / page).read_text(encoding="utf-8")
    start = text.index(marker)
    return _order_in(text[start : start + 900], names, r"\b{}\s*=")


_SHARED_METHODS = _shared_methods()


class TestThePopulationIsReal:
    """The premises the derived rule rests on, so a silent empty scan cannot pass."""

    def test_every_shipped_backend_resolves_without_its_runtime(self) -> None:
        engines = _engines()
        assert len(engines) >= 3, sorted(engines)
        for backend, engine in engines.items():
            assert issubclass(engine, SimEngine), f"{backend}: {engine!r}"

    def test_there_are_methods_two_backends_both_implement(self) -> None:
        assert len(_SHARED_METHODS) > 20, len(_SHARED_METHODS)

    def test_the_two_methods_this_module_names_are_in_the_population(self) -> None:
        pairs = {(method, left, right) for method, left, right in _SHARED_METHODS}
        assert ("add_camera", "isaac", "mujoco") in pairs
        assert ("randomize", "mujoco", "newton") in pairs

    def test_add_camera_is_related_by_no_declaration_at_all(self) -> None:
        """Nothing but this rule relates the three ``add_camera`` signatures."""
        assert not hasattr(SimEngine, "add_camera")
        assert "add_camera" not in getattr(SimEngine, "__abstractmethods__", frozenset())

    def test_randomize_is_related_only_by_a_kwargs_sink(self) -> None:
        assert _positional_parameters(SimEngine.randomize) == []


class TestNoBackendPermutesASharedParameter:
    """The rule, over every method a pair of shipped backends both implement."""

    @pytest.mark.parametrize(("method", "left", "right"), _SHARED_METHODS)
    def test_the_shared_parameters_are_in_one_order(self, method: str, left: str, right: str) -> None:
        engines = _engines()
        permuted = permuted_shared_parameters(getattr(engines[left], method), getattr(engines[right], method))
        assert permuted == (), (
            f"{method}: {left} and {right} order {list(permuted)} differently.\n"
            f"  {left}: {_positional_parameters(getattr(engines[left], method))}\n"
            f"  {right}: {_positional_parameters(getattr(engines[right], method))}"
        )


class TestOnePositionalCallMeansOneThingEverywhere:
    """The consequence, driven through the signatures rather than asserted about them."""

    def test_the_camera_call_binds_the_same_parameters_on_every_backend(self) -> None:
        engines = _engines()
        bound = {}
        for backend, engine in engines.items():
            arguments = inspect.signature(engine.add_camera).bind(None, *_CAMERA_CALL).arguments
            bound[backend] = {k: v for k, v in arguments.items() if k != "self"}
        reference = bound["mujoco"]
        for backend, got in bound.items():
            assert got == reference, f"{backend} read the same call as {got}, mujoco as {reference}"

    def test_the_camera_call_is_the_resolution_and_angle_the_caller_wrote(self) -> None:
        """Pins which reading is the shared one, not merely that they agree."""
        engines = _engines()
        for backend, engine in engines.items():
            arguments = inspect.signature(engine.add_camera).bind(None, *_CAMERA_CALL).arguments
            assert arguments["fov"] == 100, backend
            assert arguments["width"] == 200, backend
            assert arguments["height"] == 90, backend


class TestTheDocumentedOrderIsEveryBackendsOrder:
    """The two pages that write an order down grade the signatures."""

    def test_the_camera_order_matches_the_newton_page(self) -> None:
        names = ("fov", "width", "height")
        documented = _documented_signature_order("docs/simulation/newton.md", "`add_camera(name,", names)
        assert documented == ["fov", "width", "height"], documented
        engines = _engines()
        for backend, engine in engines.items():
            params = _positional_parameters(engine.add_camera)
            assert [p for p in params if p in names] == documented, backend

    def test_the_randomization_range_order_matches_the_randomization_page(self) -> None:
        names = ("color_range", "friction_range", "mass_range")
        documented = _documented_call_order("docs/simulation/domain-randomization.md", "sim.randomize(", names)
        assert documented == ["color_range", "friction_range", "mass_range"], documented
        engines = _engines()
        graded = 0
        for backend, engine in engines.items():
            params = _positional_parameters(engine.randomize)
            if not set(names) <= set(params):
                continue
            graded += 1
            assert [p for p in params if p in names] == documented, backend
        assert graded >= 2, "no backend declares all three ranges, so nothing was graded"


class TestTheRuleIsNotVacuous:
    """The shipped set is clean, so the predicate is graded on constructed exemplars.

    Without these the whole module would pass on a tree whose rule had been
    weakened to accept anything, because after the fix there is no violation
    left for the derived scan to find.
    """

    @staticmethod
    def _permuting() -> tuple[Callable[..., Any], Callable[..., Any]]:
        def left(self: Any, name: str, fov: float = 60.0, width: int = 640) -> None: ...
        def right(self: Any, name: str, width: int = 640, fov: float = 60.0) -> None: ...

        return left, right

    @staticmethod
    def _inserting() -> tuple[Callable[..., Any], Callable[..., Any]]:
        def left(self: Any, name: str, fov: float = 60.0, width: int = 640) -> None: ...
        def right(self: Any, name: str, extra: int = 0, fov: float = 60.0, width: int = 640) -> None: ...

        return left, right

    def test_a_permutation_is_reported(self) -> None:
        left, right = self._permuting()
        assert permuted_shared_parameters(left, right) == ("fov", "width")

    def test_an_insertion_is_not_reported(self) -> None:
        left, right = self._inserting()
        assert permuted_shared_parameters(left, right) == ()

    def test_both_outcomes_are_reachable(self) -> None:
        outcomes = {
            permuted_shared_parameters(*self._permuting()) != (),
            permuted_shared_parameters(*self._inserting()) != (),
        }
        assert outcomes == {True, False}, outcomes

    def test_a_keyword_only_parameter_is_out_of_scope(self) -> None:
        """A name no caller can pass positionally cannot be permuted positionally."""

        def left(self: Any, name: str, *, fov: float = 60.0, width: int = 640) -> None: ...
        def right(self: Any, name: str, *, width: int = 640, fov: float = 60.0) -> None: ...

        assert permuted_shared_parameters(left, right) == ()


class TestABackendMayStillAddItsOwnParameters:
    """The rule must not have been satisfied by flattening the backends together."""

    def test_isaac_keeps_its_own_loaders(self) -> None:
        params = _positional_parameters(_engines()["isaac"].add_robot)
        assert {"mjcf_path", "usd_path"} <= set(params), params

    def test_newton_keeps_its_own_source(self) -> None:
        assert "source" in _positional_parameters(_engines()["newton"].add_robot)

    def test_mujoco_keeps_the_position_axis_newton_lacks(self) -> None:
        mujoco = set(_positional_parameters(_engines()["mujoco"].randomize))
        newton = set(_positional_parameters(_engines()["newton"].randomize))
        assert {"randomize_positions", "position_noise"} <= mujoco - newton

    def test_each_backend_keeps_its_own_camera_defaults(self) -> None:
        """Reordering must not have changed what omitting a parameter means."""
        engines = _engines()
        for backend, expected in (("mujoco", (640, 480)), ("newton", (640, 480)), ("isaac", (None, None))):
            params = inspect.signature(engines[backend].add_camera).parameters
            assert (params["width"].default, params["height"].default) == expected, backend

    def test_the_camera_mount_contract_is_unchanged(self) -> None:
        """``parent_body`` stays last and world-fixed by default on every backend."""
        for backend, engine in _engines().items():
            params = _positional_parameters(engine.add_camera)
            assert params[-1] == "parent_body", f"{backend}: {params}"
            assert inspect.signature(engine.add_camera).parameters["parent_body"].default is None


class TestWhatThisDoesNotDecide:
    """The boundary: a shared parameter may still sit at a different *index*.

    A backend that inserts its own parameter mid-signature shifts every shared
    parameter after it, so a positional call still means two things even with no
    permutation anywhere. Closing that would move ``IsaacSimulation.add_robot``'s
    ``mjcf_path`` / ``usd_path`` off the position they have shipped at, which is
    an API decision about the primary entry point rather than a correction, so
    it is measured here and left to its own change.
    """

    @staticmethod
    def _index_divergences() -> list[tuple[str, str, str, tuple[str, ...]]]:
        engines = _engines()
        rows = []
        for method, left, right in _SHARED_METHODS:
            left_params = _positional_parameters(getattr(engines[left], method))
            right_params = _positional_parameters(getattr(engines[right], method))
            shared = set(left_params) & set(right_params)
            bad = tuple(sorted(p for p in shared if left_params.index(p) != right_params.index(p)))
            if bad:
                rows.append((method, left, right, bad))
        return rows

    def test_index_parity_is_not_the_rule_and_is_measurably_weaker(self) -> None:
        rows = self._index_divergences()
        assert rows, "index parity now holds everywhere - promote it to the rule above"
        assert {method for method, _, _, _ in rows} >= {"add_robot"}, rows

    def test_every_index_divergence_is_an_insertion_not_a_permutation(self) -> None:
        """Whatever index parity still allows, the order rule above is satisfied."""
        engines = _engines()
        for method, left, right, _ in self._index_divergences():
            assert permuted_shared_parameters(getattr(engines[left], method), getattr(engines[right], method)) == (), (
                method,
                left,
                right,
            )
