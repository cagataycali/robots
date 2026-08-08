"""Pins the push-time re-read of the duplicate-claim question.

Step 1 of the PR Workflow asks whether an open pull request already claims the
issue. That read is a claim about **minute 0**, and authoring a tested change takes
longer than the window in which a collision becomes observable, so the intake
answer can be stale by the time the push happens. This module pins the second read
that AGENTS.md step 5 now asks for, and -- more usefully -- pins *why* neither of
the two obvious cheap forms is sufficient on its own.

Both executable classes are recorded-payload comparisons of **two competing checks
disagreeing about one moment**, driven through the production classifier where one
of them is production code:

``TestTheOpenSetNarrowsOnceTheRivalMerges``
    Re-running step 1's command is only correct while the rival is still *open*.
    :func:`check_duplicate_claim.classify` reads
    ``repository.pullRequests(states: OPEN)``, so a rival that merges leaves the
    set and the verdict returns to ``unique-claim``. The issue's own state does not
    move, which is what makes it the read worth adding.

``TestABranchComparisonIsBlindToAMerge``
    Comparing the branch against a sha recorded at the start catches a sibling
    push. It cannot catch a merge, because a squash merge writes a new commit onto
    the base and never moves the head ref -- so the two checks disagree, and the
    recorded case is one where the branch comparison passed and a round was
    orphaned.

The point of both is the *reassuring direction*: a stale answer here says "no
duplicate", which is indistinguishable from a fresh one and only misleads toward
doing work that is already done.

See AGENTS.md step 5, scripts/check_duplicate_claim.py, and issues #2017 and #2031.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
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


# Reached through importlib because scripts/ is not an importable package, so its
# members are module attributes at runtime rather than names mypy can resolve.
mod = _load()


#: The #2029 / #2030 collision, as the two reads saw it. A cycle took #2029 at
#: intake with a clean answer, authored a fix, and re-asked before pushing.
_RIVAL: dict[str, Any] = {
    "issue": 2029,
    "rival": 2030,
    "rival_opened": "2026-08-08T07:24:45Z",
    "rival_merged": "2026-08-08T07:43:54Z",
    "reread": "2026-08-08T07:56:00Z",
    # The open-pull-request link sets the command compares, at each moment.
    "open_links_while_rival_open": {2030: (2029,), 2028: ()},
    "open_links_after_rival_merged": {2028: ()},
    # The issue's own fields, read at the same re-read moment.
    "issue_state": "CLOSED",
    "issue_state_reason": "COMPLETED",
    "closer": 2030,
}

#: The #2015 merge race. A round was pushed at ~23:12 and the squash merge fired
#: at 23:13:13 against the head recorded before it, so the round never landed.
_MERGE_RACE: dict[str, Any] = {
    "pull_request": 2015,
    "state": "MERGED",
    "merged_at": "2026-08-07T23:13:13Z",
    "head_ref_oid_at_merge": "ea5e3ff84aa2c05e24d86de352c0e6948634f82b",
    "merge_commit": "1026088270e42d714afe6e85de1c9cdafd1a5a11",
    "round_pushed_to_fork": "e7ab4d5b6c78d5648e09c718323345e57c97ed38",
    "round_is_ancestor_of_main": False,
    "recovery": 2018,
}


def _at(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _open_claim_verdict(links: dict[int, tuple[int, ...]]) -> str:
    """What step 1's command answers, through the production classifier."""
    return str(mod.classify([_RIVAL["issue"]], links).outcome)


def _issue_state_says_landed(payload: dict[str, Any]) -> bool:
    """What the issue's own state answers: has this work already landed?"""
    return bool(payload["issue_state"] == "CLOSED" and payload["issue_state_reason"] == "COMPLETED")


def _branch_comparison_diverged(recorded_sha: str, current_sha: str) -> bool:
    """The local form: did the branch move since the sha recorded at the start?"""
    return recorded_sha != current_sha


def _pull_request_is_gone(state: str) -> bool:
    """The remote form: is this pull request no longer awaiting my push?"""
    return state in {"MERGED", "CLOSED"}


class TestTheOpenSetNarrowsOnceTheRivalMerges:
    """Re-running the intake command answers about the *open* set, which shrinks.

    The command is still worth re-running -- it is the read that names the rival --
    but it is not the read that stays true, so the guidance asks for both.
    """

    def test_the_command_names_the_rival_while_it_is_open(self) -> None:
        verdict = mod.classify([_RIVAL["issue"]], _RIVAL["open_links_while_rival_open"])
        assert verdict.outcome == mod.DUPLICATE_CLAIM
        assert verdict.rivals == (_RIVAL["rival"],)

    def test_the_same_command_reads_clean_once_the_rival_merges(self) -> None:
        """The reassuring direction: the rival left the set, so nothing collides."""
        assert _open_claim_verdict(_RIVAL["open_links_after_rival_merged"]) == mod.UNIQUE_CLAIM

    def test_the_issues_own_state_still_names_the_work_as_landed(self) -> None:
        assert _issue_state_says_landed(_RIVAL) is True
        assert _RIVAL["closer"] == _RIVAL["rival"]

    def test_the_two_reads_disagree_at_the_moment_the_push_would_happen(self) -> None:
        """The crux: one read clears the work, the other says it is already done."""
        command_says_clear = _open_claim_verdict(_RIVAL["open_links_after_rival_merged"]) == mod.UNIQUE_CLAIM
        state_says_landed = _issue_state_says_landed(_RIVAL)
        assert command_says_clear and state_says_landed

    def test_the_observability_window_is_shorter_than_the_authoring_one(self) -> None:
        """19 minutes of visibility inside a ~40-minute authoring window."""
        window = _at(_RIVAL["rival_merged"]) - _at(_RIVAL["rival_opened"])
        assert window.total_seconds() / 60 == pytest.approx(19.15, abs=0.1)
        assert _at(_RIVAL["reread"]) > _at(_RIVAL["rival_merged"])


class TestABranchComparisonIsBlindToAMerge:
    """A squash merge writes a new commit onto the base; it never moves the head ref.

    So the cheap local check -- compare the branch against the sha recorded at the
    start -- passes across a merge. It answers "did a sibling push?", which is a
    different question from "is this pull request still waiting for me?".
    """

    def test_the_squash_merge_is_a_new_commit_and_not_the_head(self) -> None:
        assert _MERGE_RACE["merge_commit"] != _MERGE_RACE["head_ref_oid_at_merge"]

    def test_the_branch_comparison_passes_across_the_merge(self) -> None:
        head = _MERGE_RACE["head_ref_oid_at_merge"]
        assert _branch_comparison_diverged(head, head) is False

    def test_the_pull_request_state_is_the_read_that_sees_it(self) -> None:
        assert _pull_request_is_gone(_MERGE_RACE["state"]) is True

    def test_the_two_checks_disagree_on_the_recorded_race(self) -> None:
        head = _MERGE_RACE["head_ref_oid_at_merge"]
        assert _branch_comparison_diverged(head, head) is not _pull_request_is_gone(_MERGE_RACE["state"])

    def test_the_pushed_round_did_not_reach_main(self) -> None:
        """What the passing comparison cost: the round was orphaned on the fork."""
        assert _MERGE_RACE["round_is_ancestor_of_main"] is False
        assert _MERGE_RACE["round_pushed_to_fork"] != _MERGE_RACE["head_ref_oid_at_merge"]
        assert _MERGE_RACE["recovery"] != _MERGE_RACE["pull_request"]

    def test_a_sibling_push_is_what_the_comparison_does_catch(self) -> None:
        """Not an argument for dropping it -- it answers a question the state does not."""
        assert _branch_comparison_diverged(_MERGE_RACE["head_ref_oid_at_merge"], _MERGE_RACE["round_pushed_to_fork"])


class TestTheGuidanceRecordsThePushTimeReRead:
    """The rule is guidance by necessity: no workflow can see an unpushed tree."""

    @staticmethod
    def _step_five() -> str:
        """Return step 5 of the PR Workflow, whitespace-collapsed.

        Bounded at step 6 so a qualifier reworded down into another step leaves the
        slice and fails, and collapsed so a reflow cannot.
        """
        text = _AGENTS.read_text(encoding="utf-8")
        start = text.index("5. Open PR from your fork")
        end = text.index("6. Track follow-up items", start)
        return " ".join(text[start:end].split())

    def test_the_slice_is_step_five_and_nothing_else(self) -> None:
        """A collapsed or runaway slice would make every phrase pin meaningless.

        The bound is below step 5's length *without* this change (2659 collapsed
        characters against 5263 with it), so this guards the slicing rather than
        the content -- it passes on both trees, and the phrase pins are what fail
        when the guidance is missing.
        """
        step_five = self._step_five()
        assert len(step_five) > 2000, len(step_five)
        assert "Track follow-up items" not in step_five

    @pytest.mark.parametrize(
        "phrase",
        [
            "Ask the intake question again before the first push",
            "a claim about minute 0",
            "repository.pullRequests(states: OPEN)",
            "`state` and `stateReason`",
            "pullRequest { state mergedAt }",
            "never moves the head ref",
            "orphaned on the fork",
            "The same read carries `reviewThreads`",
            "Guidance rather than a check, by necessity",
            "same mode split as its `--repo` default",
        ],
    )
    def test_step_five_carries_the_push_time_re_read(self, phrase: str) -> None:
        assert phrase in self._step_five()

    @pytest.mark.parametrize(
        "figure",
        [
            _RIVAL["rival_opened"][11:19],
            _RIVAL["rival_merged"][11:19],
            _MERGE_RACE["head_ref_oid_at_merge"][:8],
            _MERGE_RACE["merge_commit"][:7],
            _MERGE_RACE["round_pushed_to_fork"][:8],
        ],
    )
    def test_the_guidance_quotes_the_recorded_measurement(self, figure: str) -> None:
        """The prose and the pins cite one measurement, not two."""
        assert figure in self._step_five()

    def test_the_guidance_names_both_reads_rather_than_one(self) -> None:
        step_five = self._step_five()
        assert "Two reads" in step_five
        assert "Unpushed work claiming an issue" in step_five
        assert "A review-round push on an existing pull request" in step_five
