"""A lazily exported tool name is never read off the ``strands_robots.tools`` package.

The tools package maps each exported name to the ``@tool`` object inside the
submodule of the *same* name and caches the result in the package ``__dict__``.
Most of those names are also submodules, so the package attribute has two
possible values, and which one a read gets is decided by what the process
imported first:

===========================================  ======  ==================  ===============
spelling                                     cold    tool cached first   submodule first
===========================================  ======  ==================  ===============
``from strands_robots.tools import X``        tool    tool                module
``import strands_robots.tools.X as x``        module  tool                module
``importlib.import_module("...tools.X")``     module  module              module
``from strands_robots.tools.X import X``      tool    tool                tool
===========================================  ======  ==================  ===============

``from strands_robots.tools import X`` is the only spelling that writes the
*tool* into that slot: a submodule import writes the module, and the two bottom
rows never read or write it. So one such read anywhere in the process is what
makes the module-alias form order-dependent, and that form is what this tree
uses widely as a monkeypatch target - it then yields the tool, and the patch
target read raises ``AttributeError`` rather than naming an import-order
problem.

Both failure directions were live: two tests read the name and used the result
as a module (``AttributeError: 'DecoratedFunctionTool' object has no attribute
'__file__'``), and two examples read it and used the result as a tool
(``AttributeError: module ... has no attribute '__wrapped__'``). Each passed in
the selection it was written against and failed in another.

The remedy is per intent, and both remedies are immune by construction: read the
submodule when the module object is wanted, and read the tool off the submodule
when the tool is wanted. This guard bans the ambiguous read instead of grading
what each site does with the result, because the read is ambiguous whatever the
site intends.

The complementary spelling - a plain ``tools.X`` attribute read - is graded,
within its own module, by
``tests/tools/test_tool_actions_block_names_the_dispatched_vocabulary.py``.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import strands_robots
import strands_robots.tools as tools_pkg

_PACKAGE_ROOT = Path(strands_robots.__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent
_TOOLS_DIR = Path(tools_pkg.__file__).resolve().parent

#: The one spelling this module bans.
_PACKAGE_MODULE = "strands_robots.tools"

#: Areas that must contribute at least one scanned module, so a mis-rooted scan
#: cannot report a clean tree. ``examples`` is among them because an example is
#: run by a reader, and both example sites read the ambiguous spelling.
_REQUIRED_AREAS = frozenset({"strands_robots", "tests", "tests_integ", "scripts", "examples"})


def _shadowable_names() -> frozenset[str]:
    """Lazily exported names that a same-named submodule also provides.

    A name with no matching submodule (the ``episode_judge`` helpers, which are
    several names out of one module) has a single possible value, so reading it
    off the package is unambiguous and stays in scope for nobody.
    """
    return frozenset(name for name in tools_pkg._LAZY_IMPORTS if (_TOOLS_DIR / f"{name}.py").exists())


def _ambiguous_reads(tree: ast.AST, shadowable: frozenset[str]) -> list[tuple[int, str]]:
    """Every ``from strands_robots.tools import <shadowable>`` under ``tree``.

    Args:
        tree: A parsed module.
        shadowable: Names for which the package attribute has two values.

    Returns:
        ``(line, name)`` for each imported name that is shadowable, in source
        order.
    """
    return sorted(
        (node.lineno, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == _PACKAGE_MODULE
        for alias in node.names
        if alias.name in shadowable
    )


def _scanned_areas() -> tuple[str, ...]:
    """Every top-level directory of the repository that ships Python.

    Derived rather than listed, so a new first-party area is graded the moment it
    lands and the scan cannot be narrowed by dropping a name. Dot-prefixed
    directories are skipped.
    """
    return tuple(
        sorted(
            entry.name
            for entry in _REPO_ROOT.iterdir()
            if entry.is_dir() and not entry.name.startswith(".") and any(entry.rglob("*.py"))
        )
    )


def _first_party_modules() -> list[Path]:
    """Every ``.py`` file under the scanned areas."""
    return [path for area in _scanned_areas() for path in sorted((_REPO_ROOT / area).rglob("*.py"))]


def _resolve_in_a_fresh_interpreter(preamble: str, statement: str) -> str:
    """Type name that ``statement`` binds to ``X`` in a clean process.

    The resolution under test is decided by what the process imported first, so
    it cannot be observed inside a suite that has already imported both. Each
    row runs in its own interpreter.

    Args:
        preamble: Imports to run before ``statement``, or the empty string.
        statement: A statement binding the name ``X``.

    Returns:
        ``type(X).__name__``.
    """
    code = f"{preamble}\n{statement}\nprint(type(X).__name__)\n"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


_COLD = ""
_TOOL_CACHED_FIRST = "from strands_robots.tools import use_rosbridge as _seed"
_SUBMODULE_FIRST = "import strands_robots.tools.use_rosbridge as _seed"

_AMBIGUOUS = "from strands_robots.tools import use_rosbridge as X"
_MODULE_ALIAS = "import strands_robots.tools.use_rosbridge as X"
_IMMUNE_MODULE = "import importlib; X = importlib.import_module('strands_robots.tools.use_rosbridge')"
_IMMUNE_TOOL = "from strands_robots.tools.use_rosbridge import use_rosbridge as X"


class TestTheAmbiguityIsReal:
    """Why the ban exists. If the package stops caching tools, these fail first."""

    def test_the_banned_spelling_resolves_two_ways(self) -> None:
        """One statement, two objects, decided by what the process imported first."""
        cold = _resolve_in_a_fresh_interpreter(_COLD, _AMBIGUOUS)
        after_submodule = _resolve_in_a_fresh_interpreter(_SUBMODULE_FIRST, _AMBIGUOUS)

        assert cold == "DecoratedFunctionTool"
        assert after_submodule == "module"
        assert cold != after_submodule

    def test_the_banned_spelling_is_what_caches_the_tool(self) -> None:
        """The module-alias form is order-dependent only because of that cache.

        Cold, it resolves to the module. Preceded by the banned read it resolves
        to the tool, which is how a monkeypatch target silently stops being one.
        """
        cold = _resolve_in_a_fresh_interpreter(_COLD, _MODULE_ALIAS)
        after_ban = _resolve_in_a_fresh_interpreter(_TOOL_CACHED_FIRST, _MODULE_ALIAS)

        assert cold == "module"
        assert after_ban == "DecoratedFunctionTool"

    @pytest.mark.parametrize(
        ("statement", "expected"),
        [(_IMMUNE_MODULE, "module"), (_IMMUNE_TOOL, "DecoratedFunctionTool")],
        ids=["module-wanted", "tool-wanted"],
    )
    def test_each_remedy_answers_the_same_way_in_every_order(self, statement: str, expected: str) -> None:
        """Both remedies read the submodule, so no import order changes them."""
        resolved = {
            preamble_id: _resolve_in_a_fresh_interpreter(preamble, statement)
            for preamble_id, preamble in (
                ("cold", _COLD),
                ("tool-cached-first", _TOOL_CACHED_FIRST),
                ("submodule-first", _SUBMODULE_FIRST),
            )
        }

        assert set(resolved.values()) == {expected}, resolved


class TestNoAmbiguousReadShipsInTheTree:
    """The ban itself, over every first-party module."""

    def test_no_module_reads_a_shadowable_tool_name_off_the_package(self) -> None:
        """A shadowable name is read off its submodule, never off the package."""
        shadowable = _shadowable_names()
        offenders = sorted(
            f"{path.relative_to(_REPO_ROOT)}:{line} imports {name!r}"
            for path in _first_party_modules()
            for line, name in _ambiguous_reads(ast.parse(path.read_text(encoding="utf-8")), shadowable)
        )

        assert not offenders, (
            "a tool name that is also a submodule is read off the tools package, so which object "
            "the name binds to is decided by what the process imported first - and the read caches "
            "the tool, which turns the module-alias form used elsewhere into a read of the tool: "
            f"{offenders}"
        )


class TestTheScanIsNonVacuous:
    """A scan that reached nothing, or flagged everything, would pass silently."""

    def test_a_constructed_ambiguous_read_is_flagged(self) -> None:
        """The rule detects the spelling it bans, wherever the alias points."""
        source = "from strands_robots.tools import pose_tool\nfrom strands_robots.tools import use_ros as r\n"

        assert _ambiguous_reads(ast.parse(source), _shadowable_names()) == [(1, "pose_tool"), (2, "use_ros")]

    @pytest.mark.parametrize(
        "source",
        [
            "import strands_robots.tools.pose_tool as pose_mod\n",
            "from strands_robots.tools.pose_tool import pose_tool\n",
            "import importlib\n\npose_mod = importlib.import_module('strands_robots.tools.pose_tool')\n",
            "from strands_robots.tools import load_episode\n",
            "from strands_robots import Robot\n",
        ],
        ids=["module-alias", "tool-off-submodule", "import-module", "not-a-submodule", "unrelated"],
    )
    def test_an_unambiguous_read_is_not_flagged(self, source: str) -> None:
        """Both remedies, the plain-module form, and a name with no submodule.

        ``load_episode`` is one of several names exported from a single module,
        so no submodule shadows it and the package attribute has one value.
        """
        assert _ambiguous_reads(ast.parse(source), _shadowable_names()) == []

    def test_the_shadowable_set_is_derived_and_not_empty(self) -> None:
        """Every shadowable name is a real export backed by a real submodule."""
        shadowable = _shadowable_names()

        assert shadowable, "premise: no lazily exported tool name is also a submodule"
        assert shadowable < frozenset(tools_pkg._LAZY_IMPORTS), (
            "premise: every name is shadowable, so the split is idle"
        )
        for name in shadowable:
            assert tools_pkg._LAZY_IMPORTS[name] == (f".{name}", name)

    def test_the_scan_reaches_every_first_party_area(self) -> None:
        """A mis-rooted or narrowed scan would report a clean tree over nothing."""
        modules = _first_party_modules()
        reached = {path.relative_to(_REPO_ROOT).parts[0] for path in modules}

        assert _REQUIRED_AREAS <= reached, sorted(reached)
        assert all(area in _scanned_areas() for area in _REQUIRED_AREAS), _scanned_areas()
