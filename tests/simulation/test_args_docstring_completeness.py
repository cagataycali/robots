"""A documented parameter is the only discovery surface a caller has.

Several simulation parameters carry a real, enforced domain: ``ros2_domain``
must be an ``int`` in ``[0, 232]`` or construction raises, ``control_substeps``
must be a positive integer or the runner raises, and ``add_object``'s
``material`` mapping is refused for any key outside
:data:`~strands_robots.simulation.mujoco.spec_builder.MATERIAL_KEYS`. When such
a parameter has no ``Args:`` entry, a caller can be *refused for a parameter the
docstring never mentions* - the guard reports a name the reader cannot look up,
so the remedy is to read the source. That is the same shape as a knob whose
accepted domain is undiscoverable, and it went unnoticed on six surfaces at once
because nothing compared a signature against its own ``Args:`` block.

These tests pin the comparison for every public method of every public class in
:mod:`strands_robots.simulation` (backend subpackages included) whose docstring
*already* has an ``Args:`` section: adding a parameter without documenting it -
or documenting one the signature no longer takes - fails here. A docstring with
no ``Args:`` section at all is out of scope: this guard checks a block that
exists for completeness rather than demanding one, which is
``test_public_member_docstrings.py``'s job for the docstring itself.

Combined entry labels are honoured deliberately. Google style is used loosely in
this package for parameters that share one description, so
``start_cameras_recording`` documents ``width/height:`` (slash-joined) and
``start_cameras_recording_synchronous`` documents ``width, height:``
(comma-joined). Both are genuinely documented, and a parser that compared the
raw label would demand entries that already exist - so labels are split on
``/``, ``,`` and ``or`` before comparing. :class:`TestCombinedEntryLabelsCount`
pins that calibration against the two real surfaces that rely on it.

The scan walks modules by AST, so it needs none of the optional simulation
backends installed, and :class:`TestTheScanIsNonVacuous` fails if a mis-rooted
scan reports a clean sweep over nothing.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import strands_robots.simulation as simulation_pkg

# Derived from an imported symbol rather than a path literal, so a moved package
# cannot leave this scanning an empty tree while reporting success.
_PACKAGE_ROOT = Path(inspect.getfile(simulation_pkg)).parent

_SECTION_HEADERS = ("Args:", "Arguments:", "Parameters:")

# ``name (type):`` / ``name:`` / ``**kwargs:`` / ``width, height:`` / ``a/b:``
_ENTRY = re.compile(r"^\s*(\*{0,2}[A-Za-z_][\w ,/]*?)\s*(\([^)]*\))?\s*:")

# One entry label may name several parameters that share a description.
_LABEL_SPLIT = re.compile(r"\s*(?:/|,|\bor\b)\s*")


def documented_parameters(doc: str) -> tuple[frozenset[str], bool]:
    """Names documented in ``doc``'s ``Args:`` section, and whether it has one.

    Args:
        doc: A dedented docstring (as :func:`ast.get_docstring` returns).

    Returns:
        ``(names, has_section)``. ``names`` is empty when there is no ``Args:``
        section, so callers must consult ``has_section`` to tell "documents
        nothing" from "has no block to check". Combined labels are split, and a
        leading ``*`` / ``**`` is stripped so ``**kwargs:`` documents ``kwargs``.
    """
    lines = doc.splitlines()
    header_indent = None
    body: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() in _SECTION_HEADERS:
            header_indent = len(line) - len(line.lstrip())
            body = lines[index + 1 :]
            break
    if header_indent is None:
        return frozenset(), False

    names: set[str] = set()
    entry_indent = None
    for line in body:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= header_indent:
            break  # section ended (next header or a dedented paragraph)
        if entry_indent is None:
            entry_indent = indent
        if indent != entry_indent:
            continue  # continuation line of the entry above
        match = _ENTRY.match(line)
        if not match:
            continue
        for part in _LABEL_SPLIT.split(match.group(1)):
            cleaned = part.strip().lstrip("*")
            if cleaned:
                names.add(cleaned)
    return frozenset(names), True


def signature_parameters(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Parameter names of ``fn`` in declaration order, minus ``self`` / ``cls``."""
    args = fn.args
    names = [p.arg for p in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return [name for name in names if name not in ("self", "cls")]


def documented_surfaces(root: Path) -> list[tuple[str, list[str], frozenset[str]]]:
    """Every public method of a public class under ``root`` that has an ``Args:`` block.

    Args:
        root: Package directory to walk. Every ``*.py`` beneath it is parsed by
            AST, so no optional backend needs to be importable.

    Returns:
        ``(surface_id, signature_parameters, documented_parameters)`` triples,
        where ``surface_id`` is ``"<relative path>::<Class>.<method>"``.
    """
    surfaces = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            if cls.name.startswith("_"):
                continue
            for fn in cls.body:
                if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if fn.name.startswith("_") and fn.name != "__init__":
                    continue
                doc = ast.get_docstring(fn)
                if not doc:
                    continue
                documented, has_section = documented_parameters(doc)
                if not has_section:
                    continue
                rel = path.relative_to(root.parent)
                surfaces.append((f"{rel}::{cls.name}.{fn.name}", signature_parameters(fn), documented))
    return surfaces


_SURFACES = documented_surfaces(_PACKAGE_ROOT)


@pytest.mark.parametrize(
    ("surface", "params", "documented"),
    _SURFACES,
    ids=[surface for surface, _, _ in _SURFACES],
)
def test_every_parameter_has_an_args_entry(surface: str, params: list[str], documented: frozenset[str]) -> None:
    """A parameter with no entry is undiscoverable, even where a guard enforces it."""
    missing = [name for name in params if name not in documented]
    assert not missing, f"{surface} accepts {missing} with no Args: entry"


@pytest.mark.parametrize(
    ("surface", "params", "documented"),
    _SURFACES,
    ids=[surface for surface, _, _ in _SURFACES],
)
def test_no_args_entry_names_a_parameter_the_signature_lacks(
    surface: str, params: list[str], documented: frozenset[str]
) -> None:
    """The other direction: an entry for a removed parameter is a false promise."""
    phantom = sorted(name for name in documented if name not in params)
    assert not phantom, f"{surface} documents {phantom}, which it does not accept"


class TestTheScanIsNonVacuous:
    """A mis-rooted or over-filtered scan must fail rather than sweep nothing."""

    def test_the_scan_root_is_the_simulation_package(self) -> None:
        assert _PACKAGE_ROOT.name == "simulation"
        assert (_PACKAGE_ROOT / "base.py").is_file()

    def test_enough_surfaces_are_scanned_to_be_meaningful(self) -> None:
        assert len(_SURFACES) >= 70, f"only {len(_SURFACES)} surfaces scanned"

    @pytest.mark.parametrize(
        "expected",
        [
            "simulation/base.py::SimEngine.get_observation",
            "simulation/base.py::SimEngine.run_policy",
            "simulation/policy_runner.py::PolicyRunner.run",
            "simulation/policy_runner.py::PolicyRunner.evaluate",
            "simulation/mujoco/simulation.py::MuJoCoSimEngine.__init__",
            "simulation/mujoco/simulation.py::MuJoCoSimEngine.add_object",
        ],
        ids=lambda expected: expected.rsplit("::", 1)[-1],
    )
    def test_a_known_surface_is_in_scope(self, expected: str) -> None:
        """The six surfaces this guard was written for must actually be walked."""
        assert any(surface == expected for surface, _, _ in _SURFACES), expected


class TestCombinedEntryLabelsCount:
    """One entry may name several parameters, and both spellings are in use here."""

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("width/height", {"width", "height"}),
            ("width, height", {"width", "height"}),
            ("host or port", {"host", "port"}),
            ("fps (int)", {"fps"}),
            ("**kwargs", {"kwargs"}),
        ],
        ids=["slash", "comma", "or", "typed", "var_keyword"],
    )
    def test_a_combined_label_documents_every_name_it_lists(self, label: str, expected: set[str]) -> None:
        doc = f"Summary.\n\nArgs:\n    {label}: Shared description.\n"
        documented, has_section = documented_parameters(doc)
        assert has_section
        assert documented == frozenset(expected)

    def test_the_real_surfaces_relying_on_combined_labels_are_clean(self) -> None:
        """Both recorder entry points document width/height as one combined entry.

        They are the reason the split exists: comparing the raw label would
        report them as missing entries that are plainly present.
        """
        relying = [
            (surface, params, documented)
            for surface, params, documented in _SURFACES
            if surface.endswith(("start_cameras_recording", "start_cameras_recording_synchronous"))
        ]
        assert len(relying) >= 2, [surface for surface, _, _ in _SURFACES if "cameras_recording" in surface]
        for surface, params, documented in relying:
            assert {"width", "height"} <= documented, surface
            assert not [name for name in params if name not in documented], surface


class TestTheScannerDetectsAPlantedDefect:
    """An empty result must mean clean sources, not a scanner that matches nothing."""

    def _scan(self, tmp_path: Path, source: str) -> list[tuple[str, list[str], frozenset[str]]]:
        pkg = tmp_path / "simulation"
        pkg.mkdir()
        (pkg / "planted.py").write_text(source, encoding="utf-8")
        return documented_surfaces(pkg)

    def test_a_missing_entry_is_reported(self, tmp_path: Path) -> None:
        surfaces = self._scan(
            tmp_path,
            "class Engine:\n"
            "    def go(self, kept, dropped):\n"
            '        """Summary.\n\n        Args:\n            kept: Documented.\n        """\n',
        )
        assert len(surfaces) == 1
        _, params, documented = surfaces[0]
        assert [name for name in params if name not in documented] == ["dropped"]

    def test_a_phantom_entry_is_reported(self, tmp_path: Path) -> None:
        surfaces = self._scan(
            tmp_path,
            "class Engine:\n"
            "    def go(self, kept):\n"
            '        """Summary.\n\n        Args:\n'
            '            kept: Documented.\n            removed: Gone from the signature.\n        """\n',
        )
        assert len(surfaces) == 1
        _, params, documented = surfaces[0]
        assert sorted(name for name in documented if name not in params) == ["removed"]

    def test_a_docstring_without_an_args_section_is_out_of_scope(self, tmp_path: Path) -> None:
        surfaces = self._scan(
            tmp_path,
            'class Engine:\n    def go(self, undocumented):\n        """Summary only."""\n',
        )
        assert surfaces == []

    def test_a_continuation_line_is_not_read_as_an_entry(self, tmp_path: Path) -> None:
        """A wrapped description often contains a colon; it must not invent a name."""
        surfaces = self._scan(
            tmp_path,
            "class Engine:\n"
            "    def go(self, only):\n"
            '        """Summary.\n\n        Args:\n'
            '            only: First line.\n                note: this wraps and has a colon.\n        """\n',
        )
        assert len(surfaces) == 1
        _, params, documented = surfaces[0]
        assert documented == frozenset({"only"})
        assert not [name for name in params if name not in documented]

    def test_a_private_class_is_out_of_scope(self, tmp_path: Path) -> None:
        surfaces = self._scan(
            tmp_path,
            "class _Internal:\n"
            "    def go(self, dropped):\n"
            '        """Summary.\n\n        Args:\n            other: Documented.\n        """\n',
        )
        assert surfaces == []
