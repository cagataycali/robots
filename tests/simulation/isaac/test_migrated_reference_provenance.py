"""Every cross-repo reference in the absorbed Isaac package names its repository.

``strands_robots/simulation/isaac/`` did not originate here. #1156
(``feat(sim): absorb Isaac Sim backend from robots-sim into this repo``) moved it
from ``strands-labs/robots-sim``, and it arrived carrying that repository's issue
and pull-request numbering written in the bare ``PR #N`` / ``issue #N`` form. In
this repository that form resolves against *this* repository, so each such
reference silently renamed itself on the way in.

Two of them were checkable, and they failed in the two different ways that
matter (#1967):

* ``PR #117`` -- this repository has no #117 at all, so the reference resolves to
  nothing and a reader knows to dig.
* ``PR #31`` -- this repository's #31 is ``chore: code hygiene, logging cleanup,
  and f-string logging migration``, merged, real, and *unrelated*. The docstring
  citing it calls it "the exception-hygiene pin", so a reader who follows the
  reference to check the claim finds a plausible-sounding hygiene PR and
  concludes it verifies. A false pass is worse than a dead link, because nothing
  prompts a second look. ``robots-sim#69`` / ``robots-sim#88`` behave the same
  way: this repository's #69 and #88 are both real, both about CI, and neither is
  what the comment means.

So the rule this module pins is that a reference in this package must name the
repository it belongs to. It is scoped and thresholded rather than absolute, and
both choices are load-bearing:

**Scoped to the Isaac package.** It is the only part of the tree holding two
numbering namespaces in one syntax. ``strands_robots/`` elsewhere cites this
repository's own pull requests bare -- ``PR #85``, ``PR #92``, ``PR #86``,
``PR #101`` -- and AGENTS.md documents #85, #92 and #86 by name under "Review
Learnings", so a tree-wide ban would demand rewriting correct references.

**Thresholded at 1000, not absolute.** ``robots-sim`` never issued a number above
**173** (its newest issue is #171 and its newest pull request #173, and it is
being archived under its own #171), while this repository was already at #1156
on the day it absorbed the backend. The two ranges therefore cannot overlap, and
a four-digit reference in this package is unambiguously local. Three such
references exist and are deliberately left bare, because they are correct:
``issue #1537`` twice in ``simulation.py`` and ``issue #1812`` in
``delta_eef.py`` -- all three added by later commits to *this* repository, which
``git blame`` attributes to a different commit than #1156.

``TestTheThresholdIsJustified`` pins that reasoning against drift, so the day
this repository's numbering could collide with the retired one, this module fails
rather than quietly narrowing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Numbers at or above this cannot be robots-sim's: its highest ever is 173.
_LOCAL_NUMBER_FLOOR = 1000

# The bare forms that silently re-target on import into this repository.
_BARE_REFERENCE = re.compile(r"(?:PR|issue)\s+#(\d+)", re.IGNORECASE)

_ISAAC_PACKAGE = Path(__file__).resolve().parents[3] / "strands_robots" / "simulation" / "isaac"


def _isaac_sources() -> list[Path]:
    return sorted(_ISAAC_PACKAGE.glob("*.py"))


def _unqualified_foreign_references(text: str) -> list[str]:
    """Return bare references whose number falls in the robots-sim range."""
    return [match.group(0) for match in _BARE_REFERENCE.finditer(text) if int(match.group(1)) < _LOCAL_NUMBER_FLOOR]


class TestTheIsaacPackageIsClean:
    """The absorbed package carries no bare reference to its origin repository."""

    def test_the_package_has_sources_to_scan(self) -> None:
        """An empty scan must mean a clean package, not a broken path."""
        sources = _isaac_sources()
        assert sources, f"no Python sources found under {_ISAAC_PACKAGE}"
        assert (_ISAAC_PACKAGE / "simulation.py") in sources

    @pytest.mark.parametrize("source", _isaac_sources(), ids=lambda p: p.name)
    def test_no_source_cites_a_foreign_number_bare(self, source: Path) -> None:
        offenders = _unqualified_foreign_references(source.read_text(encoding="utf-8"))
        assert not offenders, (
            f"{source.name} cites {offenders} in a bare form. Numbers below "
            f"{_LOCAL_NUMBER_FLOOR} in this package are robots-sim's (see #1967); "
            f"write them as 'robots-sim#N' so they name the repository they belong to."
        )


class TestTheScannerReportsARealProperty:
    """A planted defect is caught and a planted correct reference is not."""

    def test_a_planted_bare_foreign_reference_is_caught(self) -> None:
        planted = "See PR #31 for the exception-hygiene pin."
        assert _unqualified_foreign_references(planted) == ["PR #31"]

    def test_a_planted_issue_form_is_caught(self) -> None:
        assert _unqualified_foreign_references("retired by issue #69") == ["issue #69"]

    def test_the_qualified_form_is_accepted(self) -> None:
        """'robots-sim#31' is the remedy, so it must not itself trip the scan."""
        assert _unqualified_foreign_references("the exception-hygiene pin (robots-sim#31)") == []

    def test_a_local_four_digit_reference_is_accepted_bare(self) -> None:
        """The three genuine local references in this package stay legal."""
        assert _unqualified_foreign_references("This is issue #1812's option 1") == []
        assert _unqualified_foreign_references("previously example-side, issue #1537") == []


class TestTheThresholdIsJustified:
    """The threshold separates two ranges that cannot overlap."""

    def test_the_floor_clears_the_highest_robots_sim_number(self) -> None:
        """robots-sim's newest issue is #171 and newest pull request #173."""
        assert _LOCAL_NUMBER_FLOOR > 173

    def test_the_floor_is_below_the_absorbing_pull_request(self) -> None:
        """#1156 absorbed the package, so every local number since exceeds the floor."""
        assert _LOCAL_NUMBER_FLOOR < 1156

    def test_the_genuine_local_references_sit_above_the_floor(self) -> None:
        for number in (1537, 1812):
            assert number >= _LOCAL_NUMBER_FLOOR


class TestNeighbouringReferenceFormsStayOutOfScope:
    """Stated boundaries, so an omission reads as a decision."""

    def test_a_full_url_is_not_flagged(self) -> None:
        """A URL already names the repository; only its link text needed fixing."""
        url = "`robots-sim#159 <https://github.com/strands-labs/robots-sim/issues/159>`_"
        assert _unqualified_foreign_references(url) == []

    def test_the_rest_of_the_tree_is_not_scanned(self) -> None:
        """Elsewhere a bare 'PR #85' is a correct local reference, so this scan is
        deliberately confined to the one package with two namespaces."""
        assert _ISAAC_PACKAGE.name == "isaac"
        assert _ISAAC_PACKAGE.parent.name == "simulation"
