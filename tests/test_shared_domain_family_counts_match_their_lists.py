"""A shared numeric domain that counts its families enumerates that many.

Four helpers in :mod:`strands_robots.utils` open their docstring by stating how
many families of quantity share the domain, then list them. The count is load
bearing: it is how a reader arriving from one caller learns the guard is spent
elsewhere too, and how the next author learns that adding a caller in a new
family means widening the list rather than just calling the function.

That count is also the one part of the docstring nothing rechecks. It goes
stale in the quietest possible way - a fifth caller in a new family gets a
bullet and the word above it keeps saying "four", so the sentence that exists
to tell a reader how far the domain reaches now understates it, and the
docstring contradicts the list directly beneath it. It has gone stale twice:
``positive_count_error`` said "three" while listing four, and
``non_negative_count_error`` said "two" while listing three.

So the claim is graded against the list it sits above. A stated count must have
a bullet per family - which also forbids the shape that made the drift
invisible, a count enumerated in prose with nothing countable beneath it.

Deliberately not a check that the bullets match the call sites: a family is a
kind of quantity, not a call site, and several of these families have more than
one caller (the seed family alone is spent by ``TrainSpec.seed`` and
``StreamingDatasetReader.open``). The count of families is an editorial claim
about the list, and the list is what it is checked against.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import strands_robots

_PACKAGE_DIR = Path(strands_robots.__file__).parent

#: Spelled-out counts a docstring may open with. Only the words the house style
#: actually uses - a digit would read as a value in these sentences, not a count.
_SPELLED_COUNTS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

#: The claim being graded: "Shared domain for <count> families of ...".
_STATED_FAMILY_COUNT = re.compile(rf"\bfor ({'|'.join(_SPELLED_COUNTS)}) families\b", re.IGNORECASE)


def _family_bullets(doc: str) -> list[str]:
    """Top-level ``*`` bullets of *doc*, which is one per family.

    Read off the CLEANED docstring :func:`ast.get_docstring` returns, whose
    common indentation has been stripped - so a family bullet starts at column
    ``0`` and a sub-bullet elaborating one family is still indented and is not
    miscounted as another family.
    """
    return [line for line in doc.splitlines() if line.startswith("* ")]


def _documented_family_counts() -> list[tuple[str, str, int, int]]:
    """Every definition whose docstring states a family count, with both numbers.

    Returns:
        ``(location, name, stated, enumerated)`` per definition, where ``stated``
        is the count the prose claims and ``enumerated`` the number of top-level
        bullets under it.
    """
    found: list[tuple[str, str, int, int]] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            match = _STATED_FAMILY_COUNT.search(doc)
            if match is None:
                continue
            stated = _SPELLED_COUNTS[match.group(1).lower()]
            location = f"{path.relative_to(_PACKAGE_DIR.parent)}:{node.lineno}"
            found.append((location, node.name, stated, len(_family_bullets(doc))))
    return found


def test_the_scan_finds_the_domains_that_state_a_family_count() -> None:
    """The population is non-empty and is the shared-domain helpers.

    Without this the assertions below pass vacuously the moment the phrasing
    drifts and the scan matches nothing.
    """
    found = _documented_family_counts()
    assert [name for _, name, _, _ in found] == [
        "positive_whole_number_error",
        "non_negative_whole_number_error",
        "positive_count_error",
        "non_negative_count_error",
    ], f"unexpected population: {found}"


@pytest.mark.parametrize("location, name, stated, enumerated", _documented_family_counts())
def test_a_stated_family_count_enumerates_that_many_families(
    location: str, name: str, stated: int, enumerated: int
) -> None:
    """The count in the prose equals the number of families listed beneath it."""
    assert enumerated == stated, (
        f"{location}: {name} documents a shared domain 'for {stated} families' but lists "
        f"{enumerated} of them. The count tells a reader how far the domain reaches and tells "
        f"the next author that a caller in a new family widens the list; a count that "
        f"disagrees with the list beneath it misstates both. Add the missing family as a "
        f"top-level '* ' bullet, or correct the count."
    )
