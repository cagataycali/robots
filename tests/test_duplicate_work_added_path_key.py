"""Pins the added-path key against the claim-free duplicate pairs this repository shipped.

``scripts/check_duplicate_claim.py`` was built around one key,
``closingIssuesReferences``, and **249 of the last 300 pull requests (#2345
through #2708) link no issue at all** -- so for most of the traffic both
claim-keyed modes have nothing to collide on and report a unique claim while
looking straight at a duplicate pair. Issue #2709 is the third recorded instance.

:data:`_CLAIM_FREE_PAIRS` fixes the two measured ones in place, so the sweep is
tested against real shapes rather than invented ones. Both were reconstructed
from ``pulls/<n>/files``, which outlives the state change that closed one half of
each.

Three pins carry design decisions rather than behaviour:

``TestTheTwoKeysAreComplementary``
    The window holds four duplicate pairs: two reachable from a claim, two only
    from an added path, and none from both. So this is a second key rather than a
    replacement, and neither relation may be deleted in favour of the other.

``TestOnlyACreatedPathCollides``
    Widening the relation from ``ADDED`` to every ``changeType`` turns 2 selected
    pairs of 1802 into 117, which is a composition question owned by
    ``scripts/check_merge_base_overlap.py`` and not this one. The narrowness is
    the whole claim, so it is asserted from both sides.

``TestAnIncompleteAnswerIsNotAFinding``
    A truncated file list, a truncated open-pull-request list and an API error
    must all reach ``unknown-additions``. The failure mode guarded is not a false
    accusation but a silent no-op: a sweep that reports clean because it could not
    read the open set is worse than none, because it looks like one.

See scripts/check_duplicate_claim.py, issue #2709, and the "PR Workflow" section
of AGENTS.md.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "check_duplicate_claim.py"
_AGENTS = _ROOT / "AGENTS.md"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_duplicate_claim", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load()


#: The two duplicate pairs in #2345..#2708 that claimed no issue, as
#: ``(label, {number: added paths}, the path both create)``. Every path is the
#: real one the pull request added; the changelog fragment is carried too,
#: because it is the one added path that never collides -- its name embeds the
#: pull request number -- and a fixture without it would not show that.
_CLAIM_FREE_PAIRS = [
    (
        "#2388/#2389 - un-count frames a discarded episode never wrote",
        {
            2388: (
                "changelog.d/2388-recorder-uncount-discarded-frames.md",
                "tests/test_recorder_counters_track_on_disk_frames.py",
            ),
            2389: (
                "changelog.d/2389-recorder-uncount-discarded-frames.md",
                "tests/test_recorder_counters_track_on_disk_frames.py",
            ),
        },
        "tests/test_recorder_counters_track_on_disk_frames.py",
    ),
    (
        "#2707/#2708 - a value domain for TrainSpec.save_freq",
        {
            2707: (
                "changelog.d/2707-trainspec-save-freq-domain.md",
                "tests/training/test_checkpoint_cadence_domain.py",
            ),
            2708: ("tests/training/test_checkpoint_cadence_domain.py",),
        },
        "tests/training/test_checkpoint_cadence_domain.py",
    ),
]

#: The two duplicate pairs in the same window that *were* reachable from a claim,
#: as ``(left, right, the issue both claim, each side's added paths)``. Their
#: added-path sets are disjoint, which is what makes the two keys complementary.
_ISSUE_KEYED_PAIRS = [
    (
        2570,
        2571,
        2569,
        {
            2570: ("changelog.d/2570-all-open-refuses-a-local-checkout-flag.md",),
            2571: ("changelog.d/2571-sweep-refuses-a-flag-it-cannot-honour.md",),
        },
    ),
    (
        2480,
        2508,
        2466,
        {
            2480: ("changelog.d/2466-dataset-transform-surface.md", "strands_robots/transforms/base.py"),
            2508: ("changelog.d/2467-data-augmentation-notebook.md", "examples/notebooks/07_data_augmentation.ipynb"),
        },
    ),
]


def _node(number: int, files: list[dict[str, str]], total: int | None = None) -> dict[str, Any]:
    """Build one pull-request node the way the GraphQL response shapes it."""
    return {
        "number": number,
        "files": {"totalCount": len(files) if total is None else total, "nodes": files},
    }


def _added(*paths: str) -> list[dict[str, str]]:
    return [{"path": path, "changeType": check.ADDED_CHANGE_TYPE} for path in paths]


class TestTheMeasuredClaimFreePairsAreReported:
    """Each measured pair is the finding, and names the file both branches create."""

    @pytest.mark.parametrize(("label", "additions", "shared"), _CLAIM_FREE_PAIRS, ids=lambda value: str(value)[:40])
    def test_the_pair_is_the_finding(self, label: str, additions: dict[int, tuple[str, ...]], shared: str) -> None:
        verdict = check.classify_additions(additions)
        assert verdict.outcome == check.DUPLICATE_ADDITION, label
        assert verdict.is_finding

    @pytest.mark.parametrize(("label", "additions", "shared"), _CLAIM_FREE_PAIRS, ids=lambda value: str(value)[:40])
    def test_only_the_shared_path_is_reported(
        self, label: str, additions: dict[int, tuple[str, ...]], shared: str
    ) -> None:
        left, right = sorted(additions)
        assert check.classify_additions(additions).collisions == ((left, right, (shared,)),)

    @pytest.mark.parametrize(("label", "additions", "shared"), _CLAIM_FREE_PAIRS, ids=lambda value: str(value)[:40])
    def test_both_pull_requests_are_named(
        self, label: str, additions: dict[int, tuple[str, ...]], shared: str
    ) -> None:
        summary = check.classify_additions(additions).summary
        for number in additions:
            assert f"#{number}" in summary
        assert shared in summary

    def test_a_changelog_fragment_is_never_the_shared_path(self) -> None:
        """Every branch adds one, and its name embeds the number, so it cannot collide.

        The premise of the fixtures: without this, a relation over added paths
        would fire on every pair in the queue.
        """
        for label, additions, _ in _CLAIM_FREE_PAIRS:
            fragments = {path for paths in additions.values() for path in paths if path.startswith("changelog.d/")}
            assert fragments, label
            collisions = check.classify_additions(additions).collisions
            reported = {path for _, _, paths in collisions for path in paths}
            assert not (reported & fragments), label


class TestTheTwoKeysAreComplementary:
    """Four duplicate pairs, two reachable from each key, none from both."""

    @pytest.mark.parametrize(("label", "additions", "shared"), _CLAIM_FREE_PAIRS, ids=lambda value: str(value)[:40])
    def test_a_claim_free_pair_is_invisible_to_the_claim_key(
        self, label: str, additions: dict[int, tuple[str, ...]], shared: str
    ) -> None:
        """Neither half links an issue, so the claim-keyed verdict is ``no-claim``.

        Not a defect being pinned in place -- requiring a claim is what 18 of the
        last 30 merges would fail. It is why a second key had to exist.
        """
        left, right = sorted(additions)
        verdict = check.classify(claimed=(), others={right: ()})
        assert verdict.outcome == check.NO_CLAIM
        assert not verdict.is_finding

    @pytest.mark.parametrize(("left", "right", "issue", "additions"), _ISSUE_KEYED_PAIRS)
    def test_an_issue_keyed_pair_is_invisible_to_the_added_path_key(
        self, left: int, right: int, issue: int, additions: dict[int, tuple[str, ...]]
    ) -> None:
        assert check.classify_additions(additions).outcome == check.UNIQUE_ADDITIONS

    @pytest.mark.parametrize(("left", "right", "issue", "additions"), _ISSUE_KEYED_PAIRS)
    def test_the_claim_key_still_reports_that_pair(
        self, left: int, right: int, issue: int, additions: dict[int, tuple[str, ...]]
    ) -> None:
        """The non-vacuity half: each key must still catch what it was built for."""
        verdict = check.classify(claimed=(issue,), others={right: (issue,)})
        assert verdict.outcome == check.DUPLICATE_CLAIM
        assert verdict.collisions == ((issue, (right,)),)

    def test_neither_outcome_vocabulary_reuses_the_others_names(self) -> None:
        """A reader has to be able to tell which relation produced a finding."""
        claim = {check.NO_CLAIM, check.UNIQUE_CLAIM, check.DUPLICATE_CLAIM, check.UNKNOWN_CLAIMS}
        addition = {check.UNIQUE_ADDITIONS, check.DUPLICATE_ADDITION, check.UNKNOWN_ADDITIONS}
        assert not claim & addition
        assert len(claim) == 4
        assert len(addition) == 3


class TestOnlyACreatedPathCollides:
    """The narrowness is the claim: a path that already exists is the sibling's question."""

    @pytest.mark.parametrize("change_type", ["MODIFIED", "REMOVED", "RENAMED", "COPIED", "CHANGED"])
    def test_a_path_both_branches_merely_touch_is_not_a_finding(self, change_type: str) -> None:
        shared = [{"path": "strands_robots/utils.py", "changeType": change_type}]
        additions = {n: check.added_paths(_node(n, shared)) for n in (11, 12)}
        assert check.classify_additions(additions).outcome == check.UNIQUE_ADDITIONS

    def test_the_same_path_created_by_both_is_the_finding(self) -> None:
        """The control for the row above: only ``changeType`` differs."""
        shared = _added("strands_robots/utils.py")
        additions = {n: check.added_paths(_node(n, shared)) for n in (11, 12)}
        assert check.classify_additions(additions).outcome == check.DUPLICATE_ADDITION

    def test_one_branch_creating_what_the_other_edits_is_not_a_finding(self) -> None:
        """Impossible on one base, and if the API says it the sibling sweep owns it."""
        creator = check.added_paths(_node(11, _added("docs/new.md")))
        editor = check.added_paths(_node(12, [{"path": "docs/new.md", "changeType": "MODIFIED"}]))
        assert check.classify_additions({11: creator, 12: editor}).outcome == check.UNIQUE_ADDITIONS

    def test_prose_is_not_exempt(self) -> None:
        """The sibling sweep's prose exemption does not transfer to this question.

        There it holds because a shared ``.md`` edit cannot change what the suite
        does and git reports the conflict anyway. Two branches each *writing* one
        new page is duplicated authoring whatever the suffix.
        """
        additions = {n: check.added_paths(_node(n, _added("docs/guide.md"))) for n in (11, 12)}
        verdict = check.classify_additions(additions)
        assert verdict.outcome == check.DUPLICATE_ADDITION
        assert verdict.collisions == ((11, 12, ("docs/guide.md",)),)


class TestEveryPairIsComparedAndTheReportIsStable:
    """Determinism in all three axes, and no reliance on adjacent numbers."""

    def test_pull_requests_that_are_not_adjacent_are_still_compared(self) -> None:
        additions = {
            11: ("tests/test_a.py",),
            12: ("tests/test_unrelated.py",),
            99: ("tests/test_a.py",),
        }
        assert check.find_addition_collisions(additions) == ((11, 99, ("tests/test_a.py",)),)

    def test_pairs_and_paths_are_sorted(self) -> None:
        additions = {
            99: ("tests/test_b.py", "tests/test_a.py"),
            11: ("tests/test_b.py", "tests/test_a.py"),
        }
        assert check.find_addition_collisions(additions) == (
            (11, 99, ("tests/test_a.py", "tests/test_b.py")),
        )

    def test_three_branches_creating_one_file_report_all_three_pairs(self) -> None:
        additions = {n: ("tests/test_a.py",) for n in (11, 12, 13)}
        assert check.find_addition_collisions(additions) == (
            (11, 12, ("tests/test_a.py",)),
            (11, 13, ("tests/test_a.py",)),
            (12, 13, ("tests/test_a.py",)),
        )
        assert check.classify_additions(additions).implicated == (11, 12, 13)

    def test_an_empty_open_set_is_clean_and_says_so(self) -> None:
        verdict = check.classify_additions({})
        assert verdict.outcome == check.UNIQUE_ADDITIONS
        assert verdict.compared == 0

    def test_the_pair_count_is_the_number_of_comparisons(self) -> None:
        assert check.classify_additions({n: () for n in range(1, 5)}).compared == 6


class TestAnIncompleteAnswerIsNotAFinding:
    """An unread file list is neither a pass nor a finding."""

    def test_an_unreadable_set_is_its_own_outcome(self) -> None:
        verdict = check.classify_additions(None, "the API returned errors.")
        assert verdict.outcome == check.UNKNOWN_ADDITIONS
        assert not verdict.is_finding
        assert "the API returned errors." in verdict.summary

    def test_a_truncated_file_list_is_refused_rather_than_read_short(self) -> None:
        node = _node(11, _added("tests/test_a.py"), total=check.FILE_PAGE_SIZE + 1)
        with pytest.raises(check.ClaimSetUnreadable, match="file list is truncated"):
            check.added_paths(node)

    def test_a_complete_file_list_is_read(self) -> None:
        assert check.added_paths(_node(11, _added("b", "a"))) == ("a", "b")

    def test_a_node_without_a_file_list_is_unreadable(self) -> None:
        with pytest.raises(check.ClaimSetUnreadable, match="carried no file list"):
            check.added_paths({"number": 11, "files": "not a list"})

    def test_a_repository_that_is_not_in_owner_name_form_is_refused(self) -> None:
        with pytest.raises(check.ClaimSetUnreadable, match="owner/name form"):
            check.resolve_open_additions("robots", "token")

    def test_an_api_error_is_unreadable_rather_than_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(check, "_post", lambda *a, **k: {"errors": [{"message": "nope"}]})
        with pytest.raises(check.ClaimSetUnreadable, match="the API returned errors"):
            check.resolve_open_additions("owner/name", "token")

    def test_a_page_without_a_cursor_is_unreadable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "data": {
                "repository": {
                    "pullRequests": {"pageInfo": {"hasNextPage": True, "endCursor": None}, "nodes": []}
                }
            }
        }
        monkeypatch.setattr(check, "_post", lambda *a, **k: payload)
        with pytest.raises(check.ClaimSetUnreadable, match="carried no cursor"):
            check.resolve_open_additions("owner/name", "token")

    def test_a_list_longer_than_the_page_bound_is_unreadable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "data": {
                "repository": {
                    "pullRequests": {"pageInfo": {"hasNextPage": True, "endCursor": "next"}, "nodes": []}
                }
            }
        }
        monkeypatch.setattr(check, "_post", lambda *a, **k: payload)
        with pytest.raises(check.ClaimSetUnreadable, match="the list was truncated"):
            check.resolve_open_additions("owner/name", "token")


class TestTheOpenSetIsReadLiveAndWholeAndIncludesDrafts:
    """Every page, every open pull request, from the repository rather than search."""

    def test_every_page_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pages = [
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
                            "nodes": [_node(11, _added("tests/test_a.py"))],
                        }
                    }
                }
            },
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [_node(12, _added("tests/test_a.py"))],
                        }
                    }
                }
            },
        ]
        seen: list[object] = []

        def fake_post(query: str, variables: dict[str, object], token: str) -> object:
            seen.append(variables.get("after"))
            return pages[len(seen) - 1]

        monkeypatch.setattr(check, "_post", fake_post)
        additions = check.resolve_open_additions("owner/name", "token")
        assert seen == [None, "c1"]
        # The finding only exists across the page boundary, so a single-page read
        # would report clean here.
        assert check.classify_additions(additions).outcome == check.DUPLICATE_ADDITION

    def test_the_open_set_is_read_from_the_repository_and_not_from_search(self) -> None:
        """Search is eventually consistent, and a pull request opened seconds ago
        is exactly the row this sweep exists to find."""
        assert "pullRequests(states: OPEN" in check._ADDITIONS_QUERY
        assert "search(" not in check._ADDITIONS_QUERY

    def test_no_pull_request_is_excluded_for_being_a_draft(self) -> None:
        """This file's claim-keyed policy, not the sibling sweep's.

        A draft's new file is authored work whatever its merge state, so excluding
        one would hide a collision for as long as either side stayed a draft.
        """
        source = _SCRIPT.read_text(encoding="utf-8")
        sweep = source[source.index("def resolve_open_additions") : source.index("def render_additions")]
        assert "draft" not in sweep.lower()
        assert "draft" not in check._ADDITIONS_QUERY.lower()

    def test_the_whole_set_is_compared_with_nothing_under_test(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No pull request is "the subject", so none is left out of the comparison."""
        payload = {
            "data": {
                "repository": {
                    "pullRequests": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [_node(11, _added("a.py")), _node(12, _added("b.py"))],
                    }
                }
            }
        }
        monkeypatch.setattr(check, "_post", lambda *a, **k: payload)
        assert sorted(check.resolve_open_additions("owner/name", "token")) == [11, 12]


class TestTheReportNamesBothPullRequestsAndTheRemedy:
    """What a reader has to be able to act on."""

    @staticmethod
    def _finding() -> str:
        _, additions, _ = _CLAIM_FREE_PAIRS[1]
        return check.render_additions(check.classify_additions(additions), "strands-labs/robots")

    def test_the_finding_names_both_pull_requests_and_the_shared_file(self) -> None:
        report = self._finding()
        assert "#2707 + #2708" in report
        assert "tests/training/test_checkpoint_cadence_domain.py" in report

    def test_the_finding_offers_the_remedy_and_says_no_push_settles_it(self) -> None:
        report = self._finding()
        assert "What clears this" in report
        assert "Close whichever of the two is redundant" in report
        assert "no push by one of them settles it" in report

    def test_the_finding_separates_itself_from_the_composition_question(self) -> None:
        assert "not a merge order to decide" in self._finding()

    def test_a_clean_report_carries_no_remedy_section(self) -> None:
        report = check.render_additions(check.classify_additions({11: ("a.py",)}), "owner/name")
        assert check.UNIQUE_ADDITIONS in report
        assert "What clears this" not in report

    def test_a_clean_report_states_what_it_looked_at(self) -> None:
        report = check.render_additions(check.classify_additions({n: () for n in (11, 12, 13)}), "owner/name")
        assert "| open pull requests read | 3 |" in report
        assert "| pairs compared | 3 |" in report

    def test_one_row_per_colliding_pair(self) -> None:
        additions = {n: ("tests/test_a.py",) for n in (11, 12, 13)}
        report = check.render_additions(check.classify_additions(additions), "owner/name")
        for pair in ("#11 + #12", "#11 + #13", "#12 + #13"):
            assert pair in report


class TestTheExitStatusIsOneOnlyForTheFinding:
    """The contract the sibling gate scripts share: 1 is a finding, 2 is a usage error."""

    @pytest.mark.parametrize(
        ("additions", "expected"),
        [
            ({11: ("a.py",), 12: ("a.py",)}, 1),
            ({11: ("a.py",), 12: ("b.py",)}, 0),
        ],
    )
    def test_the_exit_status(
        self, monkeypatch: pytest.MonkeyPatch, additions: dict[int, tuple[str, ...]], expected: int
    ) -> None:
        monkeypatch.setattr(check, "resolve_open_additions", lambda *a, **k: additions)
        argv = ["--repo", "owner/name", "--all-open", "--token", "t"]
        assert check.main(argv) == expected

    def test_a_lookup_failure_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> dict[int, tuple[str, ...]]:
            raise check.ClaimSetUnreadable("the response carried no repository.")

        monkeypatch.setattr(check, "resolve_open_additions", boom)
        assert check.main(["--repo", "owner/name", "--all-open", "--token", "t"]) == 0

    @pytest.mark.parametrize(
        "argv",
        [
            ["--repo", "owner/name", "--token", "t"],
            ["--repo", "owner/name", "--all-open", "--pr", "5", "--token", "t"],
            ["--repo", "owner/name", "--all-open", "--issue", "5", "--token", "t"],
            ["--repo", "owner/name", "--all-open", "--pr", "5", "--issue", "6", "--token", "t"],
        ],
    )
    def test_exactly_one_subject_is_required(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit) as raised:
            check.main(argv)
        assert raised.value.code == 2

    def test_the_sweep_keeps_the_inferred_repository_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Its caller is a workflow running where the pull requests live, and a
        sweep of the wrong repository is visible in its own report."""
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/name")
        monkeypatch.setattr(check, "resolve_open_additions", lambda *a, **k: {})
        assert check.main(["--all-open", "--token", "t"]) == 0

    def test_the_sweep_reads_no_claim_and_no_single_pull_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The two keys are independent: neither lookup of the other runs here."""

        def forbidden(*args: object, **kwargs: object) -> object:
            raise AssertionError("the added-path sweep must not read a claim")

        monkeypatch.setattr(check, "resolve_claim", forbidden)
        monkeypatch.setattr(check, "resolve_open_claims", forbidden)
        monkeypatch.setattr(check, "resolve_open_additions", lambda *a, **k: {11: ("a.py",)})
        assert check.main(["--repo", "owner/name", "--all-open", "--token", "t"]) == 0


class TestTheGuidanceRecordsTheSecondKey:
    """AGENTS.md step 1 owns the duplicate-work convention, both keys of it."""

    @staticmethod
    def _step_one() -> str:
        text = _AGENTS.read_text(encoding="utf-8")
        start = text.index("1. Create the feature branch")
        end = text.index("2. Make changes", start)
        return " ".join(text[start:end].split())

    @pytest.mark.parametrize(
        "phrase",
        [
            "check_duplicate_claim.py --repo strands-labs/robots --all-open",
            "link no issue at all",
            "create the same path",
            "a path set is a property of a pushed branch",
            "check_merge_base_overlap.py",
        ],
    )
    def test_step_one_carries_the_second_key(self, phrase: str) -> None:
        assert phrase in self._step_one()

    def test_the_guidance_keeps_the_intake_check_it_already_had(self) -> None:
        """The second key is additive: the claim-keyed intake question stays."""
        step_one = self._step_one()
        assert "check_duplicate_claim.py --repo strands-labs/robots --issue" in step_one
        assert "check that no open pull request already claims the issue" in step_one


class TestTheModuleStatesTheMeasurementBehindTheRelation:
    """A relation over paths is only worth wiring up if its precision is stated."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "1802 pairs",
            "both add a path",
            "two reachable from each key, none from both",
            "a path set is a property of a *pushed",
        ],
    )
    def test_the_docstring_carries_the_measurement(self, phrase: str) -> None:
        assert check.__doc__ is not None
        assert phrase in check.__doc__

    def test_the_stale_scope_bullet_no_longer_calls_the_pair_unreachable(self) -> None:
        """The 18-of-30 measurement stands; the conclusion drawn from it was wider.

        It rules out *requiring* a claim, which neither claim-keyed mode does. It
        never ruled out colliding the pair on a different key.
        """
        assert check.__doc__ is not None
        assert "18 of the last 30 merges" in check.__doc__
        assert "collides it on an added path instead" in check.__doc__
