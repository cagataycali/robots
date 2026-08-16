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
3. Citations of the test tree, in either spelling: a test file by name under
   either pytest naming convention - the ``test_foo.py`` prefix and the
   ``foo_test.py`` suffix - and a dotted module path into the test package,
   such as a ``:mod:``/``:data:`` cross-reference role. The published wheel
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


# A shipped-source citation of a test file is a third dangling-reference
# class: the published wheel/sdist ships no test tree, so the pointer is a
# dead end for a package reader, and it silently rots the moment that test is
# renamed. The invariant a test pins belongs described inline, next to the
# code that upholds it - never behind a test filename. Both pytest naming
# conventions are dead ends and are matched: the ``test_foo.py`` prefix and
# the ``foo_test.py`` suffix (e.g. an end-to-end ``*_agent_test.py`` harness).
_TEST_FILE_REF = re.compile(r"\b(?:test_[A-Za-z0-9_]+|[A-Za-z0-9_]+_test)\.py\b")


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


def test_test_file_ref_matches_both_pytest_naming_conventions() -> None:
    """Pin that the guard catches both pytest test-file naming conventions.

    The package scan above passes trivially once the tree is clean, so it would
    not notice a narrowing of the pattern back to prefix-only. This pins the
    contract directly: a ``test_foo.py`` prefix and a ``foo_test.py`` suffix are
    both dead-end citations and must match, while an ordinary sibling module
    (``factory.py``) must not.
    """
    assert _TEST_FILE_REF.findall("see test_run_policy.py for the pin") == ["test_run_policy.py"]
    assert _TEST_FILE_REF.findall("the e2e_agent_test.py fix history") == ["e2e_agent_test.py"]
    assert _TEST_FILE_REF.findall("the smolvla_test.py harness") == ["smolvla_test.py"]
    assert _TEST_FILE_REF.findall("delegates to factory.py at import") == []


# The same third class has a second spelling that the filename pattern above
# cannot see: a dotted module path into this repository's test package, written
# as a Sphinx role (``:mod:`` / ``:data:``) or in prose. It reads as a checkable
# cross-reference and is not one - ``import tests.simulation`` raises
# ModuleNotFoundError for anyone who installed the distribution, because the
# wheel ships ``strands_robots`` alone, and renaming the test moves the target
# just as silently as the filename form does. Only the dotted form is matched: a
# slash path such as an upstream project's ``tests/authentication.rs`` names a
# file in someone else's repository and is ordinary prose.
_TEST_MODULE_REF = re.compile(r"\btests\.[A-Za-z_][A-Za-z0-9_]*")


def test_no_reference_to_the_test_package_as_a_module() -> None:
    offenders = {
        str(path.relative_to(_PKG_ROOT)): sorted(set(matches))
        for path in _package_sources()
        if (matches := _TEST_MODULE_REF.findall(path.read_text(encoding="utf-8")))
    }
    assert not offenders, (
        f"shipped source cross-references this repository's test package: {offenders}. "
        "The distribution ships no test tree, so the target cannot be imported by a "
        "package consumer and the reference still rots on rename - a Sphinx role does "
        "not make it checkable. Describe the invariant inline next to the code instead."
    )


def test_test_module_ref_matches_the_dotted_form_only() -> None:
    """Pin the dotted spelling, and pin the narrowing that keeps this guard clean.

    The package scan above passes trivially on a clean tree, so it would notice
    neither this pattern being dropped nor it being widened to any mention of the
    word. A role into the test package must match under either role name, while a
    slash path into another project's repository must not - those name a file
    outside this distribution and are ordinary prose.
    """
    assert _TEST_MODULE_REF.findall(":data:`~tests.simulation.mujoco.test_tool_spec.X`") == ["tests.simulation"]
    assert _TEST_MODULE_REF.findall("see :mod:`tests.mesh.test_acl` for the pin") == ["tests.mesh"]
    assert _TEST_MODULE_REF.findall("mirrors zenoh's tests/authentication.rs config") == []
    assert _TEST_MODULE_REF.findall("mirrors the upstream tests/notebooks layout") == []
    assert _TEST_MODULE_REF.findall("covered by the tests. See below.") == []
