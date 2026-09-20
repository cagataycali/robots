"""Repo hygiene: no test module is a copy of another module's suite.

History: ``tests/test_registry.py`` held 63 tests whose bodies were, statement
for statement, 63 of the 66 in ``tests/registry/test_public_api.py``, and
``tests/test_registry_integrity.py`` shared 8 of its bodies with
``tests/registry/test_integrity.py``. Both pairs came from a module being
copied into the ``tests/registry`` package and the original being left behind,
after which the two copies drifted: the copy under ``tests/registry`` grew the
three ``list_robots`` mode cells, the flat copy grew seven registry-metadata
cells, and a reader could not tell which file to extend.

A duplicated suite costs more than its lines. Every fix has to be applied
twice or the two copies disagree, the duplicated cells run twice on every CI
job for one behaviour, and - the reason this is graded rather than tidied - a
copy does not inherit the ``conftest.py`` of the package it was copied FROM.
``tests/registry/conftest.py`` repoints ``STRANDS_BASE_DIR`` and
``STRANDS_ASSETS_DIR`` at temp dirs for every module in that package, so the
flat copies ran the same registry assertions against the developer's real
``~/.strands_robots/user_robots.json`` - exactly the "passes on clean CI, fails
locally" hole that conftest exists to close.

The measurement is the AST of each test function with positions and docstring
formatting normalised away, so reindenting or renaming a copy does not hide it.
Two modules sharing a cell or two is ordinary - a clock-step pin reads the same
way for the mesh and for the dashboard - so the gate is on the SHARE: a module
may not have most of its bodies living in one other module.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The areas that ship pytest modules, walked one at a time.
_TREES = ("tests", "tests_integ")

#: The largest share of one module's test bodies that may also live in a single
#: other module. Half, so a pair that happens to state one behaviour the same
#: way is ordinary and a module that is mostly another module's suite is not.
MAX_SHARED_SHARE = 0.5


def _body_hash(node: ast.AST) -> str:
    """Return a positionless digest of one test function, decorators included.

    Positions and attributes are dropped so reindenting or moving a copy does
    not change its digest; the decorator list is kept, so two functions that
    differ only in their ``parametrize`` arguments are different bodies.

    Args:
        node: The ``def`` to digest.

    Returns:
        A hex digest of the normalised dump.
    """
    return hashlib.sha256(ast.dump(node, annotate_fields=False, include_attributes=False).encode()).hexdigest()


def _test_bodies(path: Path) -> dict[str, str]:
    """Return ``{qualified test name: body digest}`` for one pytest module.

    Args:
        path: Module to read. A file that does not parse contributes nothing -
            a syntax error is another gate's finding, not this one's.

    Returns:
        One entry per ``test*`` function, at module scope or inside a class.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}
    found: dict[str, str] = {}

    def walk(node: ast.AST, prefix: str = "") -> None:
        for child in getattr(node, "body", []):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and child.name.startswith("test"):
                found[prefix + child.name] = _body_hash(child)
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")

    walk(tree)
    return found


def _test_modules(root: Path = REPO_ROOT) -> list[Path]:
    """Return every ``test_*.py`` under every area that ships one.

    Args:
        root: Repository root to walk. Defaults to this repository.

    Returns:
        Paths, in area order, excluding bytecode caches.
    """
    found: list[Path] = []
    for tree in _TREES:
        for path in sorted((root / tree).rglob("test_*.py")):
            if "__pycache__" not in path.parts:
                found.append(path)
    return found


def duplicated_modules(root: Path = REPO_ROOT) -> list[str]:
    """Return one line per module that is mostly a copy of one other module.

    Args:
        root: Repository root to sweep. Defaults to this repository.

    Returns:
        Human-readable findings, sorted, worst share first. Empty when no
        module shares more than :data:`MAX_SHARED_SHARE` of its test bodies
        with any single other module.
    """
    bodies = {path: _test_bodies(path) for path in _test_modules(root)}
    bodies = {path: names for path, names in bodies.items() if names}
    holders: dict[str, set[Path]] = {}
    for path, names in bodies.items():
        for digest in names.values():
            holders.setdefault(digest, set()).add(path)

    findings: list[tuple[float, str]] = []
    for path, names in bodies.items():
        digests = set(names.values())
        shared: dict[Path, int] = {}
        for digest in digests:
            for other in holders[digest]:
                if other is not path:
                    shared[other] = shared.get(other, 0) + 1
        for other, count in shared.items():
            share = count / len(digests)
            if share > MAX_SHARED_SHARE:
                verdict = "every body" if count == len(digests) else f"{count} of {len(digests)} bodies"
                findings.append(
                    (
                        share,
                        f"{path.relative_to(root)}: {verdict} also in {other.relative_to(root)} "
                        f"({share:.0%}) - keep one module, and carry over any cell the survivor lacks",
                    )
                )
    findings.sort(key=lambda row: (-row[0], row[1]))
    return [line for _share, line in findings]


def test_no_test_module_is_mostly_a_copy_of_another() -> None:
    """No module may share more than half of its test bodies with one other."""
    findings = duplicated_modules()
    assert not findings, "Duplicated test suites:\n" + "\n".join(findings)


def test_a_planted_copy_is_reported(tmp_path: Path) -> None:
    """The sweep names a copied module, so a clean sweep means something."""
    original = tmp_path / "tests"
    original.mkdir()
    (tmp_path / "tests_integ").mkdir()
    body = "def test_one():\n    assert 1 == 1\n\n\ndef test_two():\n    assert 2 == 2\n"
    (original / "test_subject.py").write_text(body, encoding="utf-8")
    assert duplicated_modules(tmp_path) == []

    (original / "test_subject_copy.py").write_text(body, encoding="utf-8")
    findings = duplicated_modules(tmp_path)
    assert len(findings) == 2, findings
    assert any("test_subject_copy.py: every body also in" in line for line in findings)


@pytest.mark.parametrize(
    ("cells", "shared", "expected"),
    [
        pytest.param(2, 1, 0, id="half-of-a-pair-is-ordinary"),
        pytest.param(3, 2, 1, id="two-thirds-is-most-of-it"),
        pytest.param(2, 2, 1, id="both-cells-copied"),
    ],
)
def test_the_share_is_read_against_each_module_s_own_size(
    tmp_path: Path, cells: int, shared: int, expected: int
) -> None:
    """A shared cell is a finding for the module it is most of, not for both."""
    area = tmp_path / "tests"
    area.mkdir()
    (tmp_path / "tests_integ").mkdir()

    def cell(index: int) -> str:
        return f"def test_{index}():\n    assert {index} == {index}\n"

    small = "\n\n".join(cell(i) for i in range(cells))
    # The large module holds the shared cells plus enough of its own that the
    # copy is a minority of ITS bodies - only the small module is a copy.
    large = "\n\n".join([cell(i) for i in range(shared)] + [cell(100 + i) for i in range(6)])
    (area / "test_small.py").write_text(small, encoding="utf-8")
    (area / "test_large.py").write_text(large, encoding="utf-8")

    findings = duplicated_modules(tmp_path)
    assert len(findings) == expected, findings
    if expected:
        assert findings[0].startswith("tests/test_small.py: ")
