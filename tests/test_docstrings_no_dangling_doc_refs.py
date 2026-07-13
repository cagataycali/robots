"""Guard: published ``strands_robots`` source must be self-contained.

Docstrings and comments in the shipped package must not point readers at
documents that are not part of the repository, nor at line-number
self-references that rot the moment a file is edited.

Three dangling-reference classes are pinned here:

1. References to an internal design memo
   (``reports/STREAMING_DATA_LOOP_DEEP_DIVE.md``) that is not shipped in the
   distribution - every such pointer is a dead end for a reader.
2. ``~L<line>`` self-references, which silently drift out of date as soon as
   the surrounding file changes.
3. Citations of a test file by name (``test_foo.py``). The published wheel
   ships no test tree, so the pointer is a dead end for a package reader, and
   it rots the moment that test is renamed.

All three fail loudly here so they cannot creep back into the package.
"""

from __future__ import annotations

import re
from pathlib import Path

import strands_robots

_PKG_ROOT = Path(strands_robots.__file__).resolve().parent

# The unpublished internal memo earlier docstrings pointed at.
_UNPUBLISHED_MEMO = "STREAMING_DATA_LOOP_DEEP_DIVE"
# "~L1234"-style pointers into a file break the instant lines shift.
_ROTTING_LINE_REF = re.compile(r"~L\d+")


def _package_sources() -> list[Path]:
    return sorted(_PKG_ROOT.rglob("*.py"))


def test_no_reference_to_unpublished_deep_dive_memo() -> None:
    offenders = [
        str(path.relative_to(_PKG_ROOT))
        for path in _package_sources()
        if _UNPUBLISHED_MEMO in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"source references the unpublished '{_UNPUBLISHED_MEMO}' memo: {offenders}. "
        "Inline the rationale instead of pointing at a doc that is not shipped."
    )


def test_no_rotting_line_number_self_references() -> None:
    offenders = [
        str(path.relative_to(_PKG_ROOT))
        for path in _package_sources()
        if _ROTTING_LINE_REF.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        f"source uses rotting '~L<line>' self-references: {offenders}. "
        "Describe the location by symbol or behavior, not a line number."
    )


# A shipped-source citation of a test file (``test_foo.py``) is a third
# dangling-reference class: the published wheel/sdist ships no test tree, so
# the pointer is a dead end for a package reader, and it silently rots the
# moment that test is renamed. The invariant a test pins belongs described
# inline, next to the code that upholds it - never behind a test filename.
_TEST_FILE_REF = re.compile(r"\btest_[A-Za-z0-9_]+\.py\b")


def test_no_reference_to_test_files_by_name() -> None:
    offenders = {
        str(path.relative_to(_PKG_ROOT)): sorted(set(matches))
        for path in _package_sources()
        if (matches := _TEST_FILE_REF.findall(path.read_text(encoding="utf-8")))
    }
    assert not offenders, (
        f"shipped source cites test files by name: {offenders}. A package "
        "consumer installs without the test tree, so each pointer is a dead "
        "end that also rots when the test is renamed. Describe the invariant "
        "inline next to the code instead of citing a test filename."
    )
