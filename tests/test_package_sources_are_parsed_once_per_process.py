"""A package source file is parsed once per test process, not once per grader.

About 170 graders walk ``strands_robots`` and parse every file. Each parsed on
its own, many at module scope, which pytest-xdist repeats on every worker at
collection. Measured on the whole-tree roster at 14 workers, that was 134,919
parses of package files costing 775 worker-seconds, against a floor of 31
seconds for parsing each file once per worker. :func:`tests._package_ast.parse_file`
is that floor; this module keeps the tree from drifting back.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import strands_robots
from tests._package_ast import parse_file, parse_source

_TESTS = Path(__file__).resolve().parent

#: ``ast.parse(<expression>.read_text(...))`` on one line, whatever builds the
#: path (``Path(mod.__file__)``, ``(ROOT / rel)``): the spelling that parses a
#: file afresh wherever it is.
_FRESH_FILE_PARSE = re.compile(r"ast\.parse\([^\n]*\.read_text\(")


def test_no_test_module_parses_a_source_file_afresh() -> None:
    offenders = sorted(
        path.relative_to(_TESTS).as_posix()
        for path in _TESTS.rglob("*.py")
        if path.name != "_package_ast.py"
        and path != Path(__file__)
        and _FRESH_FILE_PARSE.search(path.read_text(encoding="utf-8"))
    )
    assert offenders == [], f"use tests._package_ast.parse_file(path) instead: {offenders}"


def test_a_package_file_is_parsed_once_and_any_other_file_every_time(tmp_path: Path) -> None:
    package_file = Path(strands_robots.__file__)
    assert parse_file(package_file) is parse_file(package_file)

    fixture = tmp_path / "fixture.py"
    fixture.write_text("x = 1\n", encoding="utf-8")
    first = parse_file(fixture)
    fixture.write_text("y = 2\n", encoding="utf-8")
    assert [t.id for t in first.body[0].targets] == ["x"]  # type: ignore[attr-defined]
    assert [t.id for t in parse_file(fixture).body[0].targets] == ["y"]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("line", "fresh"),
    [
        ('tree = ast.parse(path.read_text(encoding="utf-8"))', True),
        ("tree = ast.parse(Path(mod.__file__).read_text())", True),
        ("tree = ast.parse((ROOT / rel).read_text(), filename=rel)", True),
        ("tree = parse_file(ROOT / rel)", False),
        ("tree = ast.parse(source)", False),
    ],
)
def test_the_fresh_parse_spelling_is_recognised_however_the_path_is_built(line: str, fresh: bool) -> None:
    assert bool(_FRESH_FILE_PARSE.search(line)) is fresh


def test_a_package_files_text_is_parsed_once_and_any_other_text_every_time() -> None:
    package_file = Path(strands_robots.__file__)
    source = package_file.read_text(encoding="utf-8")
    assert parse_source(source) is parse_file(package_file)
    assert parse_source("x = 1\n") is not parse_source("x = 1\n")
