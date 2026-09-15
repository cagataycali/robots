"""Repo hygiene: a ``path::Name`` citation in ``AGENTS.md`` names a definition that exists.

``AGENTS.md`` closes most of its rules with a ``Pinned by`` line naming the test
that grades the rule, and several rules name the production function they are
about, in the ``<path>::<Name>`` coordinate pytest itself uses. That coordinate
is what a reader copies into ``pytest`` to run the pin, or greps for to find the
handler a rule describes, so a coordinate whose file exists but whose name does
not is a dead pointer of the same kind as a Sphinx role that does not resolve -
the rule looks enforced, and the pin it names cannot be run.

Measured on ``main`` at ``0b93bc2a``: 16 such coordinates, 15 resolving and one
naming ``test_no_module_scope_windowed_gl_default``, a test that had been
renamed to ``test_no_module_scope_platform_bound_gl_default`` when its rule was
widened from the windowed pair to every platform-bound backend. The same stale
name sat in that test module's own docstring as a bare ``:func:`` role, which
:mod:`tests.test_docstring_xref_roles_resolve` deliberately leaves out of scope
because a bare role has no decidable target in general. A ``path::Name`` pair
does: the path says which file to read.

The graded set is derived from ``AGENTS.md`` itself, so a coordinate added later
is held to the rule without anyone editing this file, and the resolution reads
the cited file's AST rather than importing it - a pin's module may need an
optional extra, and the question here is whether the name is defined, not
whether it runs.

A dotted name (``Class.method``) resolves segment by segment through nested
class and function definitions. A name with no dot is accepted wherever the
file defines it, because pytest's ``::`` coordinate reaches a method through its
class but ``AGENTS.md`` also cites a nested function (``simulation.py::_job``)
that pytest could not address at all - the citation is a *reading* coordinate,
and the reader's grep finds a nested ``def`` as readily as a top-level one.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_MD = REPO_ROOT / "AGENTS.md"

#: ``<path>.py::<Name>``, where ``<Name>`` may be dotted. The path is bounded on
#: the left by a non-path character so a prose word running into it is not read
#: as a directory, and the pair is read wherever it appears, fenced or not - a
#: coordinate quoted in a code block is still one a reader copies.
_CITATION_RE = re.compile(r"(?<![\w./-])([\w./-]+\.py)::([A-Za-z_][\w.]*)")

_DEFINITION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _citations(text: str) -> list[tuple[int, str, str]]:
    """Every ``(line, path, name)`` coordinate in ``text``, in file order."""
    found: list[tuple[int, str, str]] = []
    for match in _CITATION_RE.finditer(text):
        line = text[: match.start()].count("\n") + 1
        found.append((line, match.group(1), match.group(2)))
    return found


def _defines(tree: ast.AST, dotted: str) -> bool:
    """Whether ``tree`` defines ``dotted``.

    The first segment matches a class or function definition at any depth. Each
    further segment must be defined *directly* in the body of the definition the
    previous segment selected, so ``Outer.method`` does not resolve through a
    ``method`` that belongs to a class nested inside ``Outer``.
    """
    head, _, rest = dotted.partition(".")
    for node in ast.walk(tree):
        if isinstance(node, _DEFINITION_TYPES) and node.name == head:
            if not rest or _defines_directly(node, rest):
                return True
    return False


def _defines_directly(owner: ast.AST, dotted: str) -> bool:
    """Whether ``dotted`` is defined segment by segment in ``owner``'s own body."""
    head, _, rest = dotted.partition(".")
    for child in getattr(owner, "body", ()):
        if isinstance(child, _DEFINITION_TYPES) and child.name == head:
            if not rest or _defines_directly(child, rest):
                return True
    return False


def _unresolved(text: str, root: Path) -> list[str]:
    """Each citation in ``text`` that names no definition under ``root``."""
    problems: list[str] = []
    for line, path, name in _citations(text):
        target = root / path
        if not target.is_file():
            problems.append(f"line {line}: {path}::{name} - {path} does not exist")
            continue
        try:
            tree = ast.parse(target.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            problems.append(f"line {line}: {path}::{name} - {path} does not parse ({exc.msg})")
            continue
        if not _defines(tree, name):
            problems.append(f"line {line}: {path}::{name} - {path} defines no {name!r}")
    return problems


class TestAnAgentsMdCoordinateNamesADefinitionThatExists:
    def test_the_graded_set_is_not_empty(self) -> None:
        """A regex that matched nothing would pass the sweep by reaching no citation."""
        citations = _citations(AGENTS_MD.read_text(encoding="utf-8"))
        assert len(citations) >= 10, f"expected AGENTS.md to carry path::Name citations, found {len(citations)}"

    def test_every_citation_resolves(self) -> None:
        problems = _unresolved(AGENTS_MD.read_text(encoding="utf-8"), REPO_ROOT)
        assert not problems, (
            "AGENTS.md cites a path::Name coordinate that names nothing. Either the definition was "
            "renamed or moved and the citation was not, or the citation was mistyped; update it to the "
            "spelling the file defines:\n  " + "\n  ".join(problems)
        )


class TestTheResolverGradesWhatItClaims:
    """The rule is exercised on a planted tree so a regression in the resolver is visible."""

    @pytest.fixture
    def planted(self, tmp_path: Path) -> Path:
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "mod.py").write_text(
            "def top():\n"
            "    def inner():\n"
            "        pass\n"
            "class Outer:\n"
            "    class Nested:\n"
            "        def method(self):\n"
            "            pass\n"
            "    def other(self):\n"
            "        pass\n",
            encoding="utf-8",
        )
        (tmp_path / "broken.py").write_text("def (:\n", encoding="utf-8")
        return tmp_path

    @pytest.mark.parametrize(
        "name",
        ["top", "inner", "Outer", "Nested", "method", "Outer.Nested", "Outer.Nested.method", "Outer.other"],
    )
    def test_a_defined_name_resolves(self, planted: Path, name: str) -> None:
        assert _unresolved(f"Pinned by `pkg/mod.py::{name}`.", planted) == []

    @pytest.mark.parametrize(
        ("name", "reason"),
        [
            ("absent", "defines no 'absent'"),
            ("Outer.method", "defines no 'Outer.method'"),
            ("Nested.other", "defines no 'Nested.other'"),
        ],
    )
    def test_a_name_the_file_does_not_define_is_reported(self, planted: Path, name: str, reason: str) -> None:
        problems = _unresolved(f"Pinned by `pkg/mod.py::{name}`.", planted)
        assert len(problems) == 1 and reason in problems[0], problems

    def test_a_missing_file_is_reported(self, planted: Path) -> None:
        problems = _unresolved("see `pkg/gone.py::top`", planted)
        assert problems == ["line 1: pkg/gone.py::top - pkg/gone.py does not exist"]

    def test_a_file_that_does_not_parse_is_reported_rather_than_skipped(self, planted: Path) -> None:
        problems = _unresolved("see `broken.py::anything`", planted)
        assert len(problems) == 1 and "does not parse" in problems[0], problems

    def test_the_line_number_names_where_the_citation_sits(self, planted: Path) -> None:
        text = "first line\n\nthird: `pkg/mod.py::absent`\n"
        problems = _unresolved(text, planted)
        assert problems and problems[0].startswith("line 3:"), problems

    def test_a_coordinate_is_read_inside_a_code_fence_too(self, planted: Path) -> None:
        text = "```\npytest pkg/mod.py::absent\n```\n"
        assert len(_unresolved(text, planted)) == 1

    def test_a_bare_path_without_a_name_is_not_a_coordinate(self, planted: Path) -> None:
        assert _citations("see `pkg/gone.py` and `tests/`") == []
