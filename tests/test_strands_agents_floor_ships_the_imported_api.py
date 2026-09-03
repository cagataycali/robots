"""The declared strands-agents floor must ship the strands API the package imports.

``strands_robots`` imports three strands symbols that did not exist in
strands-agents 1.0:

* ``strands.types.tools.ToolContext`` -- first shipped in **1.5.0**, imported at
  module scope by ``strands_robots/tools/robot_mesh.py`` and
  ``strands_robots/tools/lerobot_train.py``.
* ``strands.types._events.ToolResultEvent`` -- first shipped in **1.7.0**,
  imported at module scope by ``strands_robots/hardware_robot.py`` and
  ``strands_robots/simulation/mujoco/simulation.py``.
* ``strands.hooks.BeforeToolCallEvent`` -- first shipped in **1.10.0**, imported
  at module scope by ``strands_robots/dashboard/agent_hitl.py``. ``strands.hooks``
  exports no tool-call event at all below that release, so the name is absent
  rather than moved.

Both are module-level imports, so on an older release the whole module raises at
import. The floor previously read ``>=1.0.0``, seven releases below the
capability, and the consequence was not a clean refusal:

* ``import strands_robots`` still succeeds, because ``Simulation`` and the
  hardware drivers are lazy (PEP 562) exports;
* ``strands_robots.Simulation`` then raises a bare ``AttributeError`` -- the
  advertised attribute simply is not there;
* ``Robot(..., mode="sim")`` reports ``Simulation backend 'mujoco' is declared
  in the built-in registry but its implementation module ...``, blaming the
  backend rather than the dependency that is too old; and
* the only mention of the real cause is a ``UserWarning``.

So a resolve anywhere in the declared range could produce an install that
satisfies packaging, imports, lists robots -- and cannot build a simulation.

``strands_robots.dashboard`` is not a lazy export, so the third symbol fails
differently and more plainly: on strands-agents 1.9.1 -- a release the floor
admitted -- ``import strands_robots.dashboard.agent_hitl`` raises ``ImportError:
cannot import name 'BeforeToolCallEvent' from 'strands.hooks'``. That module is
the human-in-the-loop gate a dashboard installs to pause an agent tool call
before real hardware moves, so the release range decided whether the gate was
there to be wired at all.

:data:`_STRANDS_SYMBOL_FLOORS` is the single owner of the measurement. The
tests below derive the required floor from it rather than restating a number, and
:meth:`TestTheFloorIsSelfMaintaining.test_every_strands_import_has_a_recorded_floor`
fails the moment the package imports a strands symbol that is not in the table --
so a future import of a newer strands API cannot silently leave the floor behind.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import strands_robots

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = Path(strands_robots.__file__).resolve().parent

# (module, symbol) -> first strands-agents release that ships it, measured
# against the released wheels. Everything not listed at 1.5.0 or later was
# already present in 1.0.0.
_STRANDS_SYMBOL_FLOORS: dict[tuple[str, str], str] = {
    ("strands", "Agent"): "1.0.0",
    ("strands", "tool"): "1.0.0",
    ("strands.tools.decorator", "tool"): "1.0.0",
    ("strands.tools.tools", "AgentTool"): "1.0.0",
    ("strands.types.tools", "AgentTool"): "1.0.0",
    ("strands.types.tools", "ToolResult"): "1.0.0",
    ("strands.types.tools", "ToolSpec"): "1.0.0",
    ("strands.types.tools", "ToolUse"): "1.0.0",
    ("strands.types.tools", "ToolContext"): "1.5.0",
    ("strands.types._events", "ToolResultEvent"): "1.7.0",
    ("strands.hooks", "HookProvider"): "1.0.0",
    ("strands.hooks", "HookRegistry"): "1.0.0",
    ("strands.hooks", "BeforeToolCallEvent"): "1.10.0",
}


def _required_floor() -> Version:
    """The highest first-shipped release among the symbols the package imports."""
    return max(Version(v) for v in _STRANDS_SYMBOL_FLOORS.values())


def _imported_strands_symbols() -> dict[tuple[str, str], list[str]]:
    """Map every ``(module, symbol)`` the package imports from strands to its files.

    Parses the shipped sources rather than reading ``sys.modules``, so a module
    that is optional at runtime (an unavailable backend) is still audited.
    """
    found: dict[tuple[str, str], list[str]] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = str(path.relative_to(_PACKAGE_ROOT.parent))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level:
                continue
            module = node.module or ""
            if module != "strands" and not module.startswith("strands."):
                continue
            for alias in node.names:
                found.setdefault((module, alias.name), []).append(rel)
    return found


def _capability_comment() -> str:
    """The comment block directly above the core ``strands-agents`` specifier.

    That block is the only place a maintainer reading ``pyproject.toml`` learns
    *why* the floor is the number it is, so it is read here as content rather
    than left as decoration.
    """
    lines = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    idx = next(i for i, line in enumerate(lines) if line.strip().startswith('"strands-agents>='))
    block = []
    for line in reversed(lines[:idx]):
        stripped = line.strip()
        if not stripped.startswith("#"):
            break
        block.append(stripped.lstrip("#").strip())
    return " ".join(reversed(block))


def _declared_strands_specifiers() -> dict[str, Requirement]:
    """Every declared ``strands-agents`` requirement, keyed by where it lives."""
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    out: dict[str, Requirement] = {}
    for raw in project["dependencies"]:
        req = Requirement(raw)
        if req.name == "strands-agents":
            out["project.dependencies"] = req
    for extra, entries in project.get("optional-dependencies", {}).items():
        for raw in entries:
            req = Requirement(raw)
            if req.name == "strands-agents":
                out[f"optional-dependencies.{extra}"] = req
    return out


class TestTheDeclaredFloorCoversEveryImportedSymbol:
    """Packaging must not admit a strands too old for the code it installs."""

    def test_every_declared_specifier_floors_at_the_capability(self) -> None:
        required = _required_floor()
        specifiers = _declared_strands_specifiers()
        assert specifiers, "expected at least one declared strands-agents requirement"

        too_low = {}
        for where, req in specifiers.items():
            lower = [s for s in req.specifier if s.operator in (">=", "==", "~=")]
            assert lower, f"{where}: {req} declares no lower bound"
            floor = min(Version(s.version) for s in lower)
            if floor < required:
                too_low[where] = str(req)
        assert not too_low, (
            f"strands-agents floor must be >= {required} because the package imports "
            f"{sorted(s for s, v in _STRANDS_SYMBOL_FLOORS.items() if Version(v) == required)}; "
            f"these specifiers admit older releases: {too_low}"
        )

    def test_the_upper_bound_stays_inside_the_audited_major(self) -> None:
        # The table was measured against 1.x wheels only; a range reaching into
        # 2.0 would claim a major nobody probed.
        for where, req in _declared_strands_specifiers().items():
            assert any(s.operator == "<" and Version(s.version) <= Version("2.0.0") for s in req.specifier), (
                f"{where}: {req} should cap below 2.0.0 until the 2.x API is audited"
            )


class TestTheFloorIsSelfMaintaining:
    """The table must stay in step with what the sources actually import."""

    def test_every_strands_import_has_a_recorded_floor(self) -> None:
        imported = _imported_strands_symbols()
        unrecorded = {sym: files for sym, files in imported.items() if sym not in _STRANDS_SYMBOL_FLOORS}
        assert not unrecorded, (
            "these strands symbols are imported with no recorded first-shipped release, "
            "so nothing checks the packaging floor against them: "
            f"{ {f'{m}.{s}': files for (m, s), files in unrecorded.items()} }. "
            "Add each to _STRANDS_SYMBOL_FLOORS with the release that first ships it, "
            "and raise the pyproject floor if it is higher."
        )

    def test_the_table_records_nothing_the_package_stopped_importing(self) -> None:
        # A stale entry could hold the floor above what the code needs.
        imported = set(_imported_strands_symbols())
        stale = sorted(f"{m}.{s}" for (m, s) in _STRANDS_SYMBOL_FLOORS if (m, s) not in imported)
        assert not stale, f"_STRANDS_SYMBOL_FLOORS records symbols the package no longer imports: {stale}"


class TestTheRecordedSymbolsExistInTheInstalledStrands:
    """Guard the table against strands removing or moving a symbol."""

    @pytest.mark.parametrize(("module", "symbol"), sorted(_STRANDS_SYMBOL_FLOORS))
    def test_symbol_is_importable(self, module: str, symbol: str) -> None:
        imported = pytest.importorskip(module)
        assert hasattr(imported, symbol), (
            f"{module}.{symbol} is recorded in _STRANDS_SYMBOL_FLOORS and imported by the "
            "package, but the installed strands-agents does not provide it"
        )


class TestTheManifestSaysWhichSymbolSetsTheFloor:
    """A floor nobody can attribute is a floor the next reader may lower.

    The two classes above keep the table and the specifier in step with the
    sources, and both are satisfied by a bare number. Neither reads the comment
    that carries the reason, so raising the floor while leaving the prose behind
    is silent -- and that is what happened when ``strands.hooks`` arrived: the
    block still credited a symbol two releases below the new bound.
    """

    def test_the_comment_names_the_symbol_and_release_that_set_the_floor(self) -> None:
        required = _required_floor()
        setters = sorted(
            f"{module}.{symbol}"
            for (module, symbol), first in _STRANDS_SYMBOL_FLOORS.items()
            if Version(first) == required
        )
        assert setters, "no recorded symbol matches the required floor, so the table cannot explain it"

        prose = _capability_comment()
        assert prose, "the strands-agents specifier carries no comment block to read"
        assert str(required) in prose, (
            f"the comment above the strands-agents specifier does not mention {required}, "
            f"the floor the table requires for {setters}. A reader cannot tell what the "
            f"bound is protecting. Comment reads: {prose!r}"
        )
        unnamed = [name for name in setters if name not in prose]
        assert not unnamed, (
            f"these symbols set the {required} floor but the comment above the specifier does "
            f"not name them: {unnamed}. Name each one there, so lowering the bound has to "
            f"argue with the capability it would drop."
        )
