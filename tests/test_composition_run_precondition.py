"""Contract pin for when a pre-merge composition run is owed, and what proves it.

``AGENTS.md`` > PR Workflow > step 8 tells a maintainer that a green ``CLEAN``
pull request is not sufficient, because none of the pre-merge checks can see a
*semantic* conflict with a pull request that landed on ``main`` after those
checks ran. #1766 and #1763 both edited ``_recompile_preserving_state``: the text
merged with no conflict, #1763 stayed ``MERGEABLE``/``CLEAN`` with ``SUCCESS``
checks after #1766 landed, and the squash still broke ``main``, because #1763
carried a premise test asserting the very defect #1766 had just fixed. The
prescribed remedy - merge ``main`` into the branch and run the affected tests -
is right, and it is the expensive branch of step 8: a clone, a merge, and two
suite runs that must be read as a delta because a hosted runner reports the same
376 pre-existing failures either way.

What it had no precondition for is whether a composition exists to be tested.
The trigger as written is file overlap, and file overlap answers a different
question: it says two pull requests touched the same file, not that either one
landed outside the other's ancestry. #2012 met the trigger verbatim - #1992 had
touched one of its two source files earlier the same day - and there was nothing
to compose, because #1992 sat 13 commits back in the branch's own ancestry.

One field separates the two, and it separates them exactly. Each row compares a
head against ``main`` as it stood when that pull request merged:

==================================  ===================================  ============
branch                              ``compare/<main then>...<head>``      composition
==================================  ===================================  ============
#1763, which broke ``main``          ``diverged  ahead_by=2  behind_by=1``  owed
#2012, which raised the same alarm   ``ahead  ahead_by=3  behind_by=0``     none exists
==================================  ===================================  ============

The direction matters, and it is the same asymmetry the node-ID decode carries in
the bullet below this passage: a ``behind_by`` of ``0`` *proves* nothing needs
composing, while a ``behind_by`` above zero does not prove a conflict exists. It
is a precondition that makes the overlap heuristic worth spending a run on, not a
replacement for it - a semantic conflict need not share a file at all, which is
why the overlap wording is left in place rather than swapped out.

Four classes are asserted here, and the first three are what keep the fourth
honest.

``TestTheCompositionPreconditionIsDecidableFromOneField`` *executes* the rule
over the recorded payloads, and asserts that each single-field variant drops the
outcome: flipping ``behind_by`` alone flips the verdict on both branches, and a
payload that cannot be read at all is owed a run rather than cleared. A pin that
merely asserted ``AGENTS.md`` *says* one field decides this would keep passing
against an API that had stopped reporting the field, leaving the guidance reading
plausibly while advising something impossible.

``TestTheCountsAreTotalsAndNotPageCounts`` *executes* the soundness property the
rule leans on. ``ahead_by``/``behind_by`` are totals while the ``commits`` array
is capped at 250, so distance does not weaken the check - and the reverse
direction of that same comparison reports ``behind_by=877`` beside an **empty**
``commits`` array, so a reader who derived the answer from ``commits`` would
conclude nothing was behind. That is the false safe this class exists to pin,
because it is the derivation a reader reaches for when the field is not named.

``TestTreeEqualityIsWholeTreeWhereThePathScopedDiffIsNot`` *executes* the
after-check. Equal tree shas say the bytes CI went green on are the bytes on
``main``; the path-scoped ``git diff --name-only ... -- strands_robots/ tests/``
says only that no path under two prefixes differs. #2001 is a real commit whose
entire diff - a changelog fragment, a notebook and its README - falls outside
both prefixes, so the path-scoped form reports empty across a tree change.

``TestTheGuidanceGatesTheCompositionRun`` pins the prose, because the prose is
the deliverable: a maintainer reads ``AGENTS.md``, not this module. What is
asserted is *adjacency* rather than vocabulary - the precondition, both
directions of its asymmetry, the totals property and the unreadable-is-not-zero
fallback have to stay in the same passage as the instruction, since the
precondition alone would read as licence to skip the run. A future edit
tightening it back to "check ``behind_by``" is exactly the regression, it looks
like an improvement, and nothing else in the tree would notice. That is the same
structural reason ``tests/test_graphql_node_id_targeting.py`` and
``tests/test_merge_gate_viewer_scope.py`` exist, and these text assertions follow
the shape those modules established.

Negative control: with ``origin/main``'s ``AGENTS.md`` restored, 8 of these 29
tests fail and they are exactly the eight qualifiers this change introduces. The
three executable classes pass unchanged, as do the context guard and the slice
non-vacuity - the instruction and its passage already exist, and the passage is
1810 characters collapsed on ``main`` against the 1500 that guard requires. That
split is the point: the compare payloads and the tree shas are properties of
GitHub's API and of commits already on ``main``, so only the guidance is new.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"

# Recorded `GET /repos/strands-labs/robots/compare/{base}...{head}` responses.
# Both ends of every comparison are pinned shas rather than `main`, so these stay
# the answers the API gave: a `main`-relative comparison drifts with every merge
# and could not be a test constant.

#: #1763's head against `main` as it stood when #1763 merged - which was #1766's
#: squash, three minutes earlier. This is the incident the composition rule was
#: written for, so the precondition has to fire here or it is useless.
#: `compare/3aabba091bc2...f5478b70e93c`.
_INCIDENT: dict[str, Any] = {"status": "diverged", "ahead_by": 2, "behind_by": 1}

#: #2012's head against `main` as it stood when #2012 merged (#2011's squash).
#: The branch that met the file-overlap trigger with nothing to compose.
#: `compare/94de3a589d3f...0fcdd015cb3f`.
_FALSE_ALARM: dict[str, Any] = {"status": "ahead", "ahead_by": 3, "behind_by": 0}

#: #2012's head against the squash of #1992 - the pull request whose overlap
#: raised the alarm. Thirteen commits back in the branch's own ancestry rather
#: than a concurrent landing. `compare/e38d966755ed...0fcdd015cb3f`.
_FALSE_ALARM_VS_OVERLAP_SOURCE: dict[str, Any] = {"status": "ahead", "ahead_by": 13, "behind_by": 0}

#: A head identical to its base. Included because `identical` is the fourth
#: status value and a reader switching on `status` has to handle it as "nothing
#: to compose" too. `compare/5757c1a2...5757c1a2`-shaped.
_IDENTICAL: dict[str, Any] = {"status": "identical", "ahead_by": 0, "behind_by": 0}

#: `compare/v0.4.1...5757c1a2`, far enough apart to hit the documented 250-commit
#: cap on the `commits` array while `ahead_by` reports the true total.
_TRUNCATED_AHEAD: dict[str, Any] = {
    "status": "ahead",
    "ahead_by": 877,
    "behind_by": 0,
    "total_commits": 877,
    "commits_length": 250,
}

#: The same pair compared the other way round. `behind_by` is the true total
#: while `commits` is *empty* - the shape that makes deriving the answer from
#: `commits` a false safe rather than merely imprecise.
_TRUNCATED_BEHIND: dict[str, Any] = {
    "status": "behind",
    "ahead_by": 0,
    "behind_by": 877,
    "total_commits": 0,
    "commits_length": 0,
}

# Recorded `GET /repos/strands-labs/robots/commits/{sha}` tree shas.

#: #2012's head, whose `call-test-lint / Test and Lint` run concluded `success`.
_PR_2012_HEAD = "0fcdd015cb3f623dc599b416caeedd0b289b88fa"
#: The squash of #2012 on `main`.
_PR_2012_SQUASH = "763305edf1d49bb2f2daec9bd0ed156ba62be0d0"
#: The tree each of the two commits above points at, recorded separately so that
#: the equality is the measurement rather than a restatement of one constant.
_PR_2012_HEAD_TREE = "e174201b7ccf7dcdf5f7d5c01f20a46e289b5ac3"
_PR_2012_SQUASH_TREE = "e174201b7ccf7dcdf5f7d5c01f20a46e289b5ac3"

#: #2001's squash and its parent. A real commit whose entire diff falls outside
#: `strands_robots/` and `tests/`, so the path-scoped after-check reports empty
#: across a genuine tree change.
_DOCS_ONLY_COMMIT = "d1c8527865d6633babea0e02b081be13ac87320e"
_DOCS_ONLY_TREE = "7d5a9a8526f35ee7b9493fec1c220da73cdac424"
_DOCS_ONLY_PARENT_TREE = "63a9ce1fa77714ef5c931827f0a36ab51e798d11"
_DOCS_ONLY_CHANGED_PATHS = (
    "changelog.d/1500-notebook-git-install-fallback-stale.md",
    "examples/notebooks/05_streaming_data_loop.ipynb",
    "examples/notebooks/README.md",
)

#: The prefixes step 8's `git diff --name-only` recipe scopes itself to.
_PATH_SCOPED_PREFIXES = ("strands_robots/", "tests/")


def _agents_text() -> str:
    return _AGENTS_PATH.read_text(encoding="utf-8")


def _composition_owed(compare: Mapping[str, Any] | None) -> bool:
    """Whether a local composition run is owed, from one compare response.

    Args:
        compare: A ``compare/{base}...{head}`` response body, or ``None`` when
            the comparison could not be made - a force-pushed fork sha, a
            ``404``, an API error.

    Returns:
        ``True`` when the head does not already contain every commit on the
        base, so a pair of changes exists that was never compiled together.
        An unreadable answer returns ``True`` as well: it is not a ``0``, and
        the conservative outcome is the one that runs the tests.
    """
    if compare is None:
        return True
    behind_by = compare.get("behind_by")
    if not isinstance(behind_by, int) or isinstance(behind_by, bool):
        # The field is the whole check; a response without a usable one cannot
        # clear the branch. Reported as owed rather than guessed at from
        # `status` or `commits`, both of which are weaker (see the counts class).
        return True
    return behind_by > 0


def _path_scoped_diff_is_empty(changed_paths: tuple[str, ...]) -> bool:
    """Step 8's ``git diff --name-only ... -- strands_robots/ tests/`` verdict."""
    return not [p for p in changed_paths if p.startswith(_PATH_SCOPED_PREFIXES)]


class TestTheCompositionPreconditionIsDecidableFromOneField:
    """One field decides it, and nothing else in the payload is needed."""

    def test_the_incident_the_rule_was_written_for_is_owed_a_run(self) -> None:
        assert _composition_owed(_INCIDENT) is True, (
            "#1763's head was behind the `main` it merged into, so the composition "
            "run step 8 prescribes was genuinely owed. A precondition that cleared "
            "this branch would be worse than no precondition: this is the merge that "
            "broke `main`. See #1766."
        )

    def test_the_branch_that_met_the_overlap_trigger_has_nothing_to_compose(self) -> None:
        assert _composition_owed(_FALSE_ALARM) is False, (
            "#2012's head already contained every commit on `main`, so the tree CI "
            "tested was the merge result and no pair of changes was uncompiled. "
            "Following the overlap trigger there cost a clone and two suite runs."
        )

    def test_the_overlapping_pull_request_was_in_the_branchs_own_ancestry(self) -> None:
        # The overlap was real - #1992 had touched one of #2012's two source files
        # the same day - so the trigger fired correctly and still meant nothing.
        assert _composition_owed(_FALSE_ALARM_VS_OVERLAP_SOURCE) is False
        assert _FALSE_ALARM_VS_OVERLAP_SOURCE["ahead_by"] == 13, (
            "the overlapping pull request sat 13 commits back in the branch's own "
            "ancestry, which is what distinguishes it from a concurrent landing."
        )

    def test_an_identical_head_has_nothing_to_compose(self) -> None:
        assert _composition_owed(_IDENTICAL) is False, (
            "`identical` is the fourth status value, and a head equal to its base "
            "trivially contains every commit on it."
        )

    @pytest.mark.parametrize(
        ("recorded", "expected"),
        [
            pytest.param(_INCIDENT, True, id="incident-1763"),
            pytest.param(_FALSE_ALARM, False, id="false-alarm-2012"),
        ],
    )
    def test_flipping_behind_by_alone_flips_the_verdict(self, recorded: Mapping[str, Any], expected: bool) -> None:
        # The single-field variant: every other field is left exactly as the API
        # returned it, so the verdict provably rests on `behind_by` and not on
        # `status`, `ahead_by`, or the pull request's identity.
        assert _composition_owed(recorded) is expected
        flipped = dict(recorded)
        flipped["behind_by"] = 0 if recorded["behind_by"] else 1
        assert _composition_owed(flipped) is (not expected), (
            f"changing only behind_by from {recorded['behind_by']} to "
            f"{flipped['behind_by']} did not change the verdict, so this rule is "
            "reading something else and the guidance names the wrong field."
        )

    @pytest.mark.parametrize(
        "unusable",
        [
            pytest.param(None, id="unreadable"),
            pytest.param({}, id="field-absent"),
            pytest.param({"status": "ahead", "ahead_by": 3}, id="counts-partial"),
            pytest.param({"behind_by": None}, id="field-null"),
            pytest.param({"behind_by": "0"}, id="field-string"),
            pytest.param({"behind_by": False}, id="field-bool"),
        ],
    )
    def test_an_unreadable_answer_is_not_a_zero(self, unusable: Mapping[str, Any] | None) -> None:
        assert _composition_owed(unusable) is True, (
            "a comparison that could not be read must be owed a composition run. "
            "Treating an absent, null or non-integer field as `0` turns the "
            "precondition into a way to skip the run, which is the opposite of what "
            "step 8 is for. A force-pushed fork sha is the common cause."
        )

    def test_a_bool_is_not_accepted_as_the_count(self) -> None:
        # `isinstance(True, int)` is True, so a bool would otherwise read as 1
        # and a `False` as 0 - the latter clearing a branch on a garbage payload.
        assert _composition_owed({"behind_by": False}) is True
        assert _composition_owed({"behind_by": True}) is True


class TestTheCountsAreTotalsAndNotPageCounts:
    """Distance does not weaken the check, and `commits` cannot substitute."""

    def test_the_count_survives_the_commits_array_being_truncated(self) -> None:
        assert _TRUNCATED_AHEAD["commits_length"] == 250, (
            "this comparison is meant to be far enough apart to hit the documented "
            "cap on the returned commits array; if it no longer is, pick a wider pair."
        )
        assert _TRUNCATED_AHEAD["ahead_by"] == _TRUNCATED_AHEAD["total_commits"] == 877
        assert _composition_owed(_TRUNCATED_AHEAD) is False, (
            "877 commits ahead and none behind still means nothing to compose. The "
            "counts are totals, so the check is sound at any distance."
        )

    def test_the_count_is_reported_beside_an_empty_commits_array(self) -> None:
        assert _TRUNCATED_BEHIND["behind_by"] == 877
        assert _TRUNCATED_BEHIND["commits_length"] == 0
        assert _composition_owed(_TRUNCATED_BEHIND) is True, (
            "a head 877 commits behind its base is owed a composition run."
        )

    def test_deriving_the_answer_from_the_commits_array_is_a_false_safe(self) -> None:
        # The derivation a reader reaches for when the field is not named. On the
        # recorded payload it reports the reassuring answer on the branch that is
        # 877 commits behind, which is why the guidance names `behind_by`.
        derived_from_commits = _TRUNCATED_BEHIND["commits_length"] > 0
        assert derived_from_commits is False
        assert _composition_owed(_TRUNCATED_BEHIND) is True
        assert derived_from_commits is not _composition_owed(_TRUNCATED_BEHIND), (
            "reading the commits array agreed with reading behind_by, so this test "
            "is no longer pinning a false safe. Re-measure the recorded payload."
        )


class TestTreeEqualityIsWholeTreeWhereThePathScopedDiffIsNot:
    """Equal trees say what the path-scoped diff cannot."""

    def test_the_verified_head_and_the_squash_have_the_same_tree(self) -> None:
        assert _PR_2012_HEAD != _PR_2012_SQUASH, (
            "squash rewrites the commit, so the shas differ - which is why nothing "
            "but a tree comparison ties a local run to `main`."
        )
        assert _PR_2012_HEAD_TREE == _PR_2012_SQUASH_TREE, (
            "the tree #2012's head was verified against and the tree its squash left "
            "on `main` are the same bytes, which is the claim the no-clone after-check "
            "makes. If a re-measurement disagrees, the recipe is wrong and not merely "
            "inconvenient."
        )

    def test_the_two_checks_disagree_on_a_change_outside_both_prefixes(self) -> None:
        # The whole point of offering the tree comparison: a real commit on which
        # the path-scoped recipe reports "verified" and the trees say otherwise.
        assert _DOCS_ONLY_TREE != _DOCS_ONLY_PARENT_TREE, (
            f"{_DOCS_ONLY_COMMIT} changed the tree; if these are equal the recorded "
            "shas are wrong and this test is vacuous."
        )
        tree_check_agrees = _DOCS_ONLY_TREE == _DOCS_ONLY_PARENT_TREE
        path_check_agrees = _path_scoped_diff_is_empty(_DOCS_ONLY_CHANGED_PATHS)
        assert path_check_agrees is True
        assert path_check_agrees is not tree_check_agrees, (
            "#2001's entire diff - a changelog fragment, a notebook and its README - "
            "falls outside strands_robots/ and tests/, so step 8's path-scoped recipe "
            "reports empty across a genuine tree change. The two checks disagreeing on "
            "a real commit is what makes the tree form the stronger one."
        )

    def test_the_path_scoped_diff_does_notice_a_change_under_a_prefix(self) -> None:
        # Non-vacuity for `_path_scoped_diff_is_empty`: a predicate that always
        # reported empty would pass the test above for the wrong reason.
        in_scope = _DOCS_ONLY_CHANGED_PATHS + ("strands_robots/utils.py",)
        assert _path_scoped_diff_is_empty(in_scope) is False
        assert _path_scoped_diff_is_empty(("tests/test_utils.py",)) is False


#: The end of step 8's composition passage. Bounding the slice on a sentence that
#: already exists makes "in the same passage" the literal assertion: a qualifier
#: reworded within the passage keeps passing however far it grows, and one moved
#: past this boundary fails. A fixed character window cannot express that - the
#: one this shape replaced elsewhere was measured against a passage it later
#: overran by 280 characters.
_PASSAGE_ANCHOR = "do not merge on the green alone"
_PASSAGE_END = "Fixing forward beats reverting here"


def _composition_passage() -> str | None:
    """Step 8's composition passage, whitespace collapsed.

    Collapsed because the assertion is about a qualifier being in the same
    breath as the instruction, not about where the line happens to wrap: a
    reflow must not fail a pin, and a phrase moved out of the passage must.
    """
    text = _agents_text()
    start = text.find(_PASSAGE_ANCHOR)
    if start < 0:
        return None
    end = text.find(_PASSAGE_END, start)
    passage = text[start:] if end < 0 else text[start:end]
    return " ".join(passage.split())


class TestTheGuidanceGatesTheCompositionRun:
    """The precondition is only actionable with its qualifiers beside it."""

    def test_the_instruction_is_present(self) -> None:
        # Context guard: every assertion below is positioned from this phrase, so
        # a silent rewording would move the pin rather than break it.
        assert _PASSAGE_ANCHOR in _agents_text(), (
            f"AGENTS.md no longer contains {_PASSAGE_ANCHOR!r}, which this class uses "
            "to locate the composition rule. If the instruction was deliberately "
            "reworded, update _PASSAGE_ANCHOR to match rather than deleting these "
            "tests - the point is that the rule and its precondition stay together."
        )

    def test_the_slice_is_the_passage_and_not_its_neighbour(self) -> None:
        # Non-vacuity for `_composition_passage`: an empty or runaway slice would
        # make every assertion below meaningless in opposite directions.
        passage = _composition_passage()
        assert passage is not None
        assert len(passage) > 1500, (
            f"the composition passage collapsed to {len(passage)} characters, so the "
            "pins below are reading a fragment rather than the passage."
        )
        assert "The converse happens too" not in passage, (
            "the slice ran past the composition passage into the "
            "require_last_push_approval one, so a qualifier moved out of this passage "
            "would still pass. Check _PASSAGE_END against AGENTS.md."
        )

    def test_the_guidance_names_the_field_that_decides_it(self) -> None:
        passage = _composition_passage()
        assert passage is not None and ".behind_by" in passage, (
            "AGENTS.md must name the compare field beside the instruction. 'Check "
            "whether the branch is behind' is advice that still costs a clone to act "
            "on; a named field on a response already being fetched does not. See #2014."
        )

    def test_the_guidance_states_which_direction_proves_something(self) -> None:
        passage = _composition_passage()
        assert passage is not None and "proves nothing needs composing" in passage, (
            "AGENTS.md must say that a zero proves the absence of a composition. "
            "Without it the field reads as one more signal to weigh rather than as a "
            "precondition that settles the question. See #2014."
        )

    def test_the_guidance_states_which_direction_proves_nothing(self) -> None:
        passage = _composition_passage()
        assert passage is not None and "does not prove a conflict exists" in passage, (
            "AGENTS.md must state the unsound direction explicitly. A precondition "
            "presented without it invites the reader to treat a non-zero as a reason "
            "to run and a zero as the only question, and to drop the overlap "
            "heuristic that selects which branches are worth a run. See #2014."
        )

    def test_the_guidance_keeps_the_overlap_heuristic(self) -> None:
        passage = _composition_passage()
        assert passage is not None and "overlap heuristic" in passage, (
            "AGENTS.md must keep the file-overlap read as the narrowing rather than "
            "replacing it: a semantic conflict need not share a file, so the "
            "precondition is necessary and not sufficient. See #2014."
        )

    def test_the_guidance_says_the_counts_are_totals(self) -> None:
        passage = _composition_passage()
        assert passage is not None and "totals rather than page counts" in passage, (
            "AGENTS.md must say the counts are totals. Otherwise a reader who knows "
            "the commits array is capped has reason to distrust the field at exactly "
            "the distances where it matters most. See #2014."
        )

    def test_the_guidance_says_an_unreadable_comparison_is_not_a_zero(self) -> None:
        passage = _composition_passage()
        assert passage is not None and "run the composition" in passage, (
            "AGENTS.md must say that a comparison it cannot make is not a zero. "
            "Without it the precondition fails open on a force-pushed fork sha, "
            "which is the common case on a contributor branch. See #2014."
        )

    def test_the_guidance_offers_tree_equality_as_the_no_clone_after_check(self) -> None:
        passage = _composition_passage()
        assert passage is not None and ".commit.tree.sha" in passage, (
            "AGENTS.md must name the tree-sha field beside the path-scoped diff. The "
            "diff needs the local composition to still exist, so on a batch it is "
            "often unavailable exactly when it is the sole evidence. See #2014."
        )

    def test_the_guidance_says_why_tree_equality_is_the_stronger_claim(self) -> None:
        passage = _composition_passage()
        assert passage is not None and "rather than two prefixes" in passage, (
            "AGENTS.md must say the tree comparison is whole-tree where the diff is "
            "path-scoped. Presented as merely more convenient, a reader has no reason "
            "to prefer it, and the paths it misses - changelog.d/, pyproject.toml, a "
            "workflow - are ones squash-time edits land in. See #2014."
        )
