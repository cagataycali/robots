"""A publish refused for one release is re-run against its tag.

A ``release`` event runs the workflow file from the commit the tag names, so a
publish whose only trigger is ``release`` cannot be repaired for a release that
is already out: re-publishing it re-runs the same broken file.  The only ways
left are to move the tag or to cut another version, and both rewrite history
that has already been announced.  That is the position ``v0.5.2`` was in - the
release was published, the run was refused before any job existed, and PyPI kept
serving the previous version.

So the workflow also accepts a ``workflow_dispatch`` carrying the tag, read from
``main``: the two places that decide which tree is graded and built - the
reusable call's ``ref`` and the build job's checkout - both prefer that input,
which is what makes ``hatch version`` derive the released version rather than the
branch's.  Each falls back to the release's own ref, so the ``release`` path is
unchanged.

Parsing is line-based rather than via ``yaml``, for the reason
``tests/test_workflow_jobs_are_bounded.py`` gives: ``tests/`` is type-checked
under ``ignore_missing_imports = false`` and ``types-PyYAML`` is not a dev
dependency.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_PUBLISH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pypi-publish-on-release.yml"

#: The dispatch input every assertion below is about.
_INPUT = "tag"

#: How a job or step names that input, whatever it falls back to.
_READS_THE_INPUT = re.compile(r"\$\{\{\s*inputs\." + _INPUT + r"\s*\|\|")


def _text() -> str:
    return _PUBLISH.read_text(encoding="utf-8")


def _block(header: re.Pattern[str], indent: int) -> list[str]:
    """The lines under the first line matching ``header``, while they stay indented.

    Args:
        header: Pattern the opening line must match.
        indent: Column the block's own keys sit at; a line at or left of the
            opening line's own indentation ends the block.

    Returns:
        The block's lines, empty when the header is absent.
    """
    lines = _text().splitlines()
    for index, line in enumerate(lines):
        if header.match(line):
            body = []
            for rest in lines[index + 1 :]:
                if rest.strip() and len(rest) - len(rest.lstrip()) < indent:
                    break
                body.append(rest)
            return body
    return []


def test_the_publish_accepts_a_dispatch_carrying_the_release_tag() -> None:
    """Without this trigger a refused publish can only be re-run from the tag."""
    declared = _block(re.compile(r"^  workflow_dispatch:\s*$"), indent=4)
    assert any(line.strip() == f"{_INPUT}:" for line in declared), (
        f"pypi-publish-on-release.yml declares no workflow_dispatch `{_INPUT}` input, so a "
        "release whose run was refused can only be re-run by moving its tag"
    )
    assert any(line.strip() == "required: true" for line in declared), (
        f"the `{_INPUT}` input is optional, so a dispatch with it forgotten silently "
        "builds the default branch and publishes a version nobody released"
    )


@pytest.mark.parametrize(
    ("what", "line_pattern"),
    [
        ("the tree the reusable test job grades", re.compile(r"^ {6}ref:")),
        ("the tree the build job checks out", re.compile(r"^ {10}ref:")),
    ],
    ids=("call-test-lint", "build"),
)
def test_a_dispatched_re_run_grades_and_builds_the_tag(what: str, line_pattern: re.Pattern[str]) -> None:
    """Both ref decisions prefer the dispatch input, or the re-run builds a branch."""
    reads = [line for line in _text().splitlines() if line_pattern.match(line) and _READS_THE_INPUT.search(line)]
    assert reads, (
        f"{what} does not read the workflow_dispatch `{_INPUT}` input, so a dispatched "
        "re-run builds the default branch and hatch-vcs derives a development version "
        "the Validate step rejects"
    )
