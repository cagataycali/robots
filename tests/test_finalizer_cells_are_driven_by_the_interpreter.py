"""Repo hygiene: a test reaches a finalizer by releasing an object, not by calling ``__del__``.

``__del__`` is invoked by the interpreter when the last reference to an object
goes away. A test that calls it as an ordinary method grades the method *body*
while saying nothing about the finalization a caller actually gets: the object
is still alive and still referenced, so a guard that depends on the object being
collected -- or a teardown whose cost is the point, like a socket that must not
block on close -- is exercised in a state the runtime never produces. Dropping
the last reference and collecting reaches the same code through the path that
ships.

The idiom this gate points at is already the one used for the finalizers that
have the most careful coverage in this tree.
``tests/test_hardware_cleanup_survives_a_failed_robot_init.py`` forces the
collection in its ``_failed_bring_up()`` helper and pins *why* in
``test_the_finalizer_runs_only_once_the_traceback_is_released``: a half-built
instance stays alive while the traceback that references it does, so an
assertion taken straight after the raise reads a log that has not been written.
That reasoning is a property of finalization, not of one file.

There is a second, practical reason to keep the call out of the tree. CodeQL's
``py/explicit-call-to-delete`` fires on it, ``github-advanced-security`` opens a
pull-request review thread per new alert, and the ``default`` branch ruleset
sets ``required_review_thread_resolution: true`` -- so the idiom is a merge
blocker for whoever writes the next finalizer test, whatever its severity.
``.github/codeql/codeql-config.yml`` deliberately excludes exactly two rule ids
and this is not one of them, which leaves the tree as the place to settle it.

Scope: every ``obj.__del__()`` call in a test area. Defining ``__del__``, and
``super().__del__()`` inside such a definition, are untouched -- this is about
*invoking* another object's finalizer by hand.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Test areas swept. Derived below rather than trusted, so an area that ships
#: tests and is missing here is reported instead of silently skipped.
_REQUIRED_TEST_AREAS = frozenset({"tests", "tests_integ"})

_SKIP_DIRS = frozenset({"__pycache__", ".git", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache"})


def _test_areas() -> list[Path]:
    """Top-level directories of the repository that ship tests.

    Returns:
        Existing test-area directories, sorted by name.
    """
    areas = [
        entry
        for entry in REPO_ROOT.iterdir()
        if entry.is_dir() and entry.name in _REQUIRED_TEST_AREAS and not entry.name.startswith(".")
    ]
    return sorted(areas, key=lambda p: p.name)


def _python_files(area: Path) -> list[Path]:
    """Every Python file under ``area``, skipping caches and virtualenvs.

    Args:
        area: Directory to walk.

    Returns:
        The Python files found, sorted.
    """
    return sorted(p for p in area.rglob("*.py") if not _SKIP_DIRS & set(p.parts))


def explicit_finalizer_calls(source: str) -> list[int]:
    """Line numbers of calls that invoke another object's ``__del__``.

    A call qualifies when it is an attribute call whose attribute is
    ``__del__``. ``super().__del__()`` written *inside* a ``__del__``
    definition is a cooperative chain-up rather than a hand-invoked finalizer,
    so it is not reported.

    Args:
        source: Python source text.

    Returns:
        Line numbers, ascending. Unparseable source yields no findings; the
        syntax gates own that failure.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    chained: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__del__":
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "__del__"
                    and isinstance(inner.func.value, ast.Call)
                    and isinstance(inner.func.value.func, ast.Name)
                    and inner.func.value.func.id == "super"
                ):
                    chained.add(id(inner))

    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__del__"
        and id(node) not in chained
    )


def _offenders() -> dict[str, list[int]]:
    """Every explicit finalizer call in the swept test areas.

    Returns:
        Repository-relative path -> line numbers.
    """
    found: dict[str, list[int]] = {}
    for area in _test_areas():
        for path in _python_files(area):
            if path.resolve() == Path(__file__).resolve():
                continue
            lines = explicit_finalizer_calls(path.read_text(encoding="utf-8"))
            if lines:
                found[str(path.relative_to(REPO_ROOT))] = lines
    return found


class TestNoTestCallsAFinalizerByHand:
    """The rule, and the sweep that carries it."""

    def test_no_test_invokes_another_objects_del(self) -> None:
        """Release the last reference and collect instead."""
        offenders = _offenders()
        assert offenders == {}, (
            "these tests call __del__ as a method, which grades the method body "
            "rather than the finalization a caller gets: "
            f"{offenders}. Drop the last reference and collect instead "
            "(`del obj` then `gc.collect()`), as "
            "tests/test_hardware_cleanup_survives_a_failed_robot_init.py does. "
            "Where the finalizer must run on a particular thread, hand that "
            "thread the only remaining reference and release it there."
        )

    def test_every_area_that_ships_tests_is_swept(self) -> None:
        """Non-vacuity: the rule above cannot pass by sweeping nothing."""
        swept = {area.name for area in _test_areas()}
        assert swept == set(_REQUIRED_TEST_AREAS), (
            f"test areas present but not swept: {sorted(_REQUIRED_TEST_AREAS - swept)}"
        )
        counts = {area.name: len(_python_files(area)) for area in _test_areas()}
        assert all(n > 0 for n in counts.values()), f"an area swept no files: {counts}"


class TestTheScannerDetectsWhatItClaimsTo:
    """The detector, graded against sources whose verdict is known."""

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("obj.__del__()", [1]),
            ("engine.__del__()\nclient.__del__()", [1, 2]),
            ("self._client.__del__()", [1]),
            # The shape this gate asks for.
            ("del obj\ngc.collect()", []),
            # Defining a finalizer, and chaining up inside one, are untouched.
            ("class A:\n    def __del__(self):\n        pass", []),
            ("class A:\n    def __del__(self):\n        super().__del__()", []),
            # A cooperative chain-up outside a __del__ definition is still a
            # hand-invoked finalizer.
            ("class A:\n    def close(self):\n        super().__del__()", [3]),
            # Not a call.
            ("fn = obj.__del__", []),
            ("obj.__delattr__('x')", []),
        ],
        ids=[
            "bare_call",
            "two_calls",
            "nested_attribute",
            "release_and_collect",
            "defining_del",
            "chain_up_inside_del",
            "chain_up_outside_del",
            "reference_without_call",
            "similarly_named_dunder",
        ],
    )
    def test_the_scanner_agrees_with_a_known_verdict(self, source: str, expected: list[int]) -> None:
        assert explicit_finalizer_calls(source) == expected

    def test_unparseable_source_yields_no_findings(self) -> None:
        """The syntax gates own a broken file; this one must not crash on it."""
        assert explicit_finalizer_calls("def (") == []
