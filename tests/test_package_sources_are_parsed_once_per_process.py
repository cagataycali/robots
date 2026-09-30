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

import strands_robots
from tests._package_ast import parse_file

_TESTS = Path(__file__).resolve().parent

#: ``ast.parse(<path>.read_text(...))`` with or without ``filename=``: the
#: spelling that parses a file afresh wherever it is.
_FRESH_FILE_PARSE = re.compile(r"ast\.parse\(\s*[A-Za-z_][\w.]*\.read_text\(")


def test_no_test_module_parses_a_source_file_afresh() -> None:
    offenders = sorted(
        path.relative_to(_TESTS).as_posix()
        for path in _TESTS.rglob("*.py")
        if path.name != "_package_ast.py" and _FRESH_FILE_PARSE.search(path.read_text(encoding="utf-8"))
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
