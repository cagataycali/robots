"""Contract pin for the compare ref step 8 tells you to issue, and what a `404` means.

``AGENTS.md`` step 8 offers ``GET /compare/main...<head> -> .behind_by`` as the
cheap read that says whether a pre-merge composition is owed at all, and #2014
added the clause that a head which *cannot* be compared is not a ``0``. Both are
right about their own case. The template was not: an unqualified ref does not
resolve in the base repository, and step 1 mandates that the branch live on a
fork, so the documented form ``404``s on **every** pull request in this
repository.

That deletes the whole benefit the shortcut was introduced for. #2012's finding
was that ``behind_by: 0`` proves no composition exists and saves a clone plus two
suite runs; a reader following the template literally gets a ``404``, reaches the
"not a ``0``" clause, and runs the composition every time. The misreading is
silent and self-justifying - a ``404`` followed by a composition run looks like
diligence, and the composition almost always confirms nothing is wrong, so
nothing ever contradicts it.

The two causes of a ``404`` need opposite actions and are indistinguishable from
the status code:

==================================  ==========================  ===================
cause                               means                       correct action
==================================  ==========================  ===================
ref not qualified with fork owner   query construction error    re-issue qualified
head sha genuinely gone             uncomparable                run the composition
==================================  ==========================  ===================

So two classes are asserted here, and the first is what keeps the second honest.

``TestTheDocumentedCompareRefResolves`` *executes* the claim rather than
describing it. It lifts the ref template out of the prose, renders it against a
recorded pull request payload, and asserts the result is the ref the API answered
``200`` for - with the ref that ``404``d as the negative control. A pin that
merely asserted ``AGENTS.md`` *mentions* fork qualification would keep passing if
the template were tidied back to ``<head>``, which is exactly the edit that
reintroduces the defect.

``TestTheTreeEquivalenceIsScopedToBehindZero`` does the same for the second,
smaller correction in the same step. The tree-sha equivalence check - equal
``.commit.tree.sha`` between the head CI went green on and its squash on ``main``
- holds only when ``behind_by == 0``, which is the single case it was measured on.
When the branch is behind, the squash tree incorporates the intervening commits,
so the trees differ for a perfectly correct merge. Unscoped, it invites reading a
good merge as drift: the same expensive-in-the-diligent-direction shape as the
advisory-``CodeQL`` case step 8 already documents.

``TestTheGuidanceNamesTheForkConstraint`` pins the prose, because the prose is
the deliverable - a contributor reads ``AGENTS.md``, not this module. What is
asserted is *adjacency*: the qualified template, the reason it is obliged, and
the two-causes split have to stay in the same breath, since the template alone
reads as an incidental verbosity someone may shorten. Same structural reason
``tests/test_ref_creation_ruleset_scope.py`` and
``tests/test_graphql_node_id_targeting.py`` exist.

Negative control: with ``origin/main``'s ``AGENTS.md`` restored, the template
renders to ``main...<head>`` - not a ref this repository ever answered - and **13 of
the 17 tests fail**: 5 of the 7 in ``TestTheDocumentedCompareRefResolves``, both
prose assertions in ``TestTheTreeEquivalenceIsScopedToBehindZero``, and all 6 in
``TestTheGuidanceNamesTheForkConstraint``.

The 4 that pass are the ones that should, and saying which is the point of running
it. Two are recorded evidence rather than claims about the file - the ``404`` on the
unqualified ref, and the two tree pairs - so they are a property of the repository
and hold whatever ``AGENTS.md`` says. The third,
``test_the_rendered_ref_is_not_the_unqualified_one``, guards a *different* edit than
the one being reverted: ``main...<head>`` is already not the bare-branch form, so
only a template respelled to ``main...{head_branch}`` trips it. It is kept for that
case and is deliberately not load-bearing here.

See #2026.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"


class _CompareResponse(NamedTuple):
    """A ``GET /repos/{owner}/{repo}/compare/{ref}`` answer, as recorded."""

    http: int
    status: str
    behind_by: int | None


#: ``GET /repos/strands-labs/robots/compare/{ref}`` as answered on 2026-08-08,
#: for #1035's head (``Vivek0712/robots``, ``feat/ackermann-ros-robot``). The
#: first pair isolates one variable: same head, same instant, only the ref
#: spelling differs. ``behind_by`` is recorded because it is the field step 8
#: reads, and it is unavailable in the ``404`` row - which is the whole defect.
_RECORDED_COMPARE: dict[str, _CompareResponse] = {
    "main...feat/ackermann-ros-robot": _CompareResponse(404, "Not Found", None),
    "main...Vivek0712:robots:feat/ackermann-ros-robot": _CompareResponse(200, "diverged", 116),
    # A branch in the base repository, to show the qualified form is universal
    # and there is one spelling to remember rather than a choice to make.
    "main...strands-labs:robots:main": _CompareResponse(200, "identical", 0),
}

#: The answer for a ref this repository was never asked, so a template that
#: renders to one fails on the assertion below rather than on a ``KeyError``.
_NEVER_ASKED = _CompareResponse(0, "not a ref this repository ever answered", None)

#: #1035's head as GraphQL reports it, which is where step 8 now says to resolve
#: the three parts from.
_HEAD_REPOSITORY_NAME_WITH_OWNER = "Vivek0712/robots"
_HEAD_REF_NAME = "feat/ackermann-ros-robot"


class _TreePair(NamedTuple):
    """A head commit's tree and its squash's tree, against that branch's distance."""

    pr: int
    behind_by: int
    head_tree: str
    squash_tree: str


#: ``.commit.tree.sha`` for a head whose ``call-test-lint`` was ``SUCCESS`` and
#: for its squash on ``main``, against that pull request's ``behind_by``. #2012 is
#: the case the equivalence was measured on; #2024 is the counterexample, whose
#: ``main`` went green on all four checks afterwards, so its unequal trees are a
#: correct merge and not drift.
_RECORDED_TREES: tuple[_TreePair, ...] = (
    _TreePair(pr=2012, behind_by=0, head_tree="e174201b7ccf", squash_tree="e174201b7ccf"),
    _TreePair(pr=2024, behind_by=1, head_tree="4af91f210d09", squash_tree="8b3e7e8a3434"),
)


@pytest.fixture(scope="module")
def agents_text() -> str:
    """``AGENTS.md`` as shipped."""
    return _AGENTS_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def step_eight(agents_text: str) -> str:
    """The step-8 passage carrying the composition guidance.

    Sliced so an assertion cannot be satisfied by matching vocabulary that
    happens to appear elsewhere in a file this long.
    """
    start = agents_text.index("One field says whether a composition exists at all")
    end = agents_text.index("Fixing forward beats reverting here", start)
    passage = agents_text[start:end]
    assert len(passage) > 500, "step-8 slice collapsed; the anchors moved"
    return passage


def _documented_compare_ref_template(text: str) -> str:
    """The ref half of the ``compare`` template ``AGENTS.md`` tells you to issue.

    Reads the template out of the prose rather than restating it, so the file is
    the source of truth and a change to it is what this module measures. A
    trailing ``->  .behind_by`` on the same line is tolerated so that the older
    one-line spelling still parses: the point of failure should be the ref this
    renders to, not this function.
    """
    match = re.search(
        r"^\s*GET /repos/\{owner\}/\{repo\}/compare/(?P<ref>\S+?)(?:\s+->.*)?$",
        text,
        re.MULTILINE,
    )
    assert match is not None, "no compare template found in AGENTS.md; the passage moved or was removed"
    return match.group("ref")


def _render(template: str, *, head_name_with_owner: str, head_branch: str, base_branch: str = "main") -> str:
    """Substitute a pull request's head into the documented template."""
    head_owner, _, head_repo = head_name_with_owner.partition("/")
    return template.format(
        base_branch=base_branch,
        head_owner=head_owner,
        head_repo=head_repo,
        head_branch=head_branch,
    )


def _recorded(ref: str) -> _CompareResponse:
    """The recorded API answer for ``ref``, or the never-asked sentinel."""
    return _RECORDED_COMPARE.get(ref, _NEVER_ASKED)


def _rendered_for_the_fork_head(agents_text: str) -> str:
    """The documented template rendered against #1035's head."""
    return _render(
        _documented_compare_ref_template(agents_text),
        head_name_with_owner=_HEAD_REPOSITORY_NAME_WITH_OWNER,
        head_branch=_HEAD_REF_NAME,
    )


def _tree_equivalence_applies(behind_by: int) -> bool:
    """Whether the head/squash tree-sha equivalence is a valid check.

    The derivation step 8 now asks a reader to perform: equal trees are evidence
    only when the head already contained every commit on ``main``, because
    otherwise the squash tree incorporates the intervening commits.
    """
    return behind_by == 0


class TestTheDocumentedCompareRefResolves:
    """The template in the file, rendered, is a ref the API can answer."""

    def test_the_rendered_ref_is_the_one_that_answered_200(self, agents_text: str) -> None:
        assert _recorded(_rendered_for_the_fork_head(agents_text)).http == 200

    def test_the_rendered_ref_exposes_the_behind_by_field(self, agents_text: str) -> None:
        assert _recorded(_rendered_for_the_fork_head(agents_text)).behind_by is not None

    def test_the_rendered_ref_is_not_the_unqualified_one(self, agents_text: str) -> None:
        assert _rendered_for_the_fork_head(agents_text) != f"main...{_HEAD_REF_NAME}"

    def test_the_template_carries_the_head_owner_and_repository(self, agents_text: str) -> None:
        template = _documented_compare_ref_template(agents_text)
        assert "{head_owner}" in template
        assert "{head_repo}" in template

    def test_the_unqualified_ref_is_the_recorded_404(self) -> None:
        """The negative control, so the pair above is a measurement and not a preference."""
        assert _recorded(f"main...{_HEAD_REF_NAME}").http == 404

    def test_the_qualified_form_also_serves_a_base_repository_branch(self, agents_text: str) -> None:
        rendered = _render(
            _documented_compare_ref_template(agents_text),
            head_name_with_owner="strands-labs/robots",
            head_branch="main",
        )
        assert _recorded(rendered).http == 200

    def test_the_passage_says_where_to_resolve_the_head_from(self, step_eight: str) -> None:
        assert "headRepository" in step_eight
        assert "headRefName" in step_eight


class TestTheTreeEquivalenceIsScopedToBehindZero:
    """Equal trees are evidence at ``behind_by == 0`` and at no other value."""

    def test_equal_trees_where_the_check_applies(self) -> None:
        applicable = [row for row in _RECORDED_TREES if _tree_equivalence_applies(row.behind_by)]
        assert applicable, "no recorded row exercises the applicable branch"
        for row in applicable:
            assert row.head_tree == row.squash_tree, row

    def test_a_behind_branch_is_outside_the_check(self) -> None:
        behind = [row for row in _RECORDED_TREES if not _tree_equivalence_applies(row.behind_by)]
        assert behind, "no recorded row exercises the inapplicable branch"
        for row in behind:
            # Unequal trees for a merge that was correct - which is why reading
            # the check outside its scope invents drift.
            assert row.head_tree != row.squash_tree, row

    def test_the_passage_scopes_the_equivalence(self, step_eight: str) -> None:
        assert "scoped to `behind_by == 0`" in step_eight

    def test_the_passage_names_the_counterexample(self, step_eight: str) -> None:
        assert "8b3e7e8a3434" in step_eight, "the #2024 counterexample row is the evidence for the scoping"


class TestTheGuidanceNamesTheForkConstraint:
    """The template, its justification and the two-causes split stay adjacent."""

    def test_it_says_to_qualify_the_head(self, step_eight: str) -> None:
        assert "Qualify the head with its owner and repository" in step_eight

    def test_it_names_the_step_that_forces_a_fork(self, step_eight: str) -> None:
        assert "Step 1 mandates" in step_eight
        assert "fork" in step_eight

    def test_it_states_that_the_unqualified_form_always_fails_here(self, step_eight: str) -> None:
        assert "`404`s on every pull request here" in step_eight

    def test_it_separates_the_two_causes_of_a_404(self, step_eight: str) -> None:
        assert "two causes wanting opposite actions" in step_eight

    def test_it_keeps_the_uncomparable_case(self, step_eight: str) -> None:
        """#2014's clause is narrowed, not dropped: a gone sha still needs the run."""
        assert "is not a `0`: run the composition" in step_eight
        assert "force-push" in step_eight

    def test_it_says_which_form_the_404_must_be_read_on(self, step_eight: str) -> None:
        assert "*qualified* form as the uncomparable one" in step_eight
