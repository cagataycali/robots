"""Contract pins for reading a filtered ``timelineItems`` connection.

``AGENTS.md`` > PR Workflow > step 8 opens with the one read that has to happen
*before* a mutation: query
``timelineItems(itemTypes: [CLOSED_EVENT, REOPENED_EVENT])`` before closing or
reopening a pull request, because "a lone ``CLOSED_EVENT`` is safe to re-apply;
an alternating run means something is undoing you".

The field a reader reaches for to answer "how many" is ``totalCount``, and on a
filtered connection **``totalCount`` ignores ``itemTypes`` entirely**. It is the
count of the pull request's whole timeline - commits, reviews, comments, project
status changes. The argument narrows ``nodes`` and nothing else.

So the cheap read reports a close/reopen history that does not exist. Four pull
requests in this repository, one query each::

    pull request  state                         totalCount   nodes
    #2144         OPEN, never closed once        2            0
    #2143         MERGED                         13           1  (the merge)
    #1987         MERGED, one deliberate flip    25           3
    #1667         CLOSED, the flip war           119          45

#2144 is the sharpest of the four: it had never been closed in its life, and a
connection filtered to ``[CLOSED_EVENT, REOPENED_EVENT]`` answered ``2``. The
mechanism is settled by asking for a type that cannot be present at all -
``itemTypes: [CONVERT_TO_DRAFT_EVENT]`` on #2143 returns ``totalCount: 13``
beside an empty ``nodes`` - so this is not a recompute lag that a second read
would settle, and no amount of re-reading it helps. It answers a different
question than the one the argument asks.

It is not even stable. #1667 was read twice twenty minutes apart, its 45
close/reopen events unchanged and the pull request closed and retired since
2026-07-30, and the filtered count moved ``119`` -> ``120``. The 120th item is a
``CrossReferencedEvent`` from #2146 - the issue reporting this defect - so
writing about #1667 raised its apparent flip count by one. An agent that cached
``119`` and re-read it later would see a number that had grown and conclude
someone had flipped the pull request again.

What makes it worth a module rather than a footnote is the direction of the
error and what the misread refuses. The number *overstates*, so it invents
history rather than hiding it, and the flip it argues against is not optional:
step 8 prescribes a close/reopen as the only remedy for the #1988 condition, a
head commit written through the API under the Actions token spawning zero check
suites, where the required check never reports and ``BLOCKED`` is terminal.
There is no suite to re-run and re-pushing costs a re-approval round. An agent
that reads ``totalCount`` on such a pull request sees a two-digit alternating run
where the truth is zero, correctly applies the rule as written, and declines -
after which the pull request sits at ``APPROVED`` / ``BLOCKED`` forever and is
reported as waiting on a reviewer, which is the presentation #1905 and #1917
both record for their own separate causes.

That is the same shape as the advisory-``CodeQL`` lesson in the same file: the
wrong action does not look like a mistake, it looks like diligence, and nothing
else in the payload contradicts it. Both fields were fetched in the read that
found this. Only one of them was believed.

Four classes, and the first three are what keep the last honest.

``TestTheFilteredCountIsTheUnfilteredTotal`` executes the arithmetic over the
recorded payloads, so the pin says *why* the count is wrong rather than only
that the prose mentions it.

``TestTheVerdictKeyedOnNodesIsTheRightOne`` runs both derivations - the one step
8 now asks for, and the one it used to permit - and asserts they disagree on the
pull request where it matters and agree on #1667. The agreement row is the point:
on a genuine flip war both answers are "do not flip", so agreement proves
nothing, and the two can only be told apart on a pull request that was safe.

``TestTheTailIsWhatMakesAFlipUnsafe`` pins the second, opposite-direction trap in
the same read. ``nodes`` is ordered oldest-first while the judgement is about
what happened *last*, and on #1667 ``first: 3`` and ``last: 3`` are disjoint
windows five days apart.

``TestTheGuidanceNamesTheHonestField`` pins the prose, because the prose is the
deliverable: an agent reads ``AGENTS.md``, not this module. What is asserted is
*adjacency* - the correction has to stay in the same breath as the instruction it
qualifies, since the bare instruction reads perfectly well on its own and is
exactly what a later tidy-up would leave behind. That follows the shape
``tests/test_merge_gate_viewer_scope.py`` and
``tests/test_graphql_node_id_targeting.py`` established for step 8's other two
reading-discipline corrections.

Negative control: with ``origin/main``'s ``AGENTS.md`` restored, 5 of the 6 tests
in ``TestTheGuidanceNamesTheHonestField`` fail - the sixth is the context guard,
which locates the passage and so passes on both trees - and all 23 in the four
offline classes pass unchanged - the API's behaviour is a property of GitHub rather than
of this change; only the guidance is new.

See #2146.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"

#: The timeline item types step 8 filters for.
_WANTED = ("ClosedEvent", "ReopenedEvent")

#: ``repository.pullRequest.timelineItems(itemTypes: [CLOSED_EVENT,
#: REOPENED_EVENT], first: 100)`` recorded 2026-08-11 with ``PAT_TOKEN``. Node
#: lists are abridged to their ``__typename`` sequence, which is everything the
#: derivation reads; ``#1667``'s timestamps are kept because the tail class needs
#: them. ``total_count`` is the value the API returned *beside* those nodes.
_RECORDED: dict[int, dict[str, Any]] = {
    2144: {
        "state": "OPEN",
        "total_count": 2,
        "nodes": [],
        "truth": "never closed once",
    },
    2143: {
        "state": "MERGED",
        "total_count": 13,
        "nodes": ["ClosedEvent"],
        "truth": "one close, and it is the squash merge",
    },
    1987: {
        "state": "MERGED",
        "total_count": 25,
        "nodes": ["ClosedEvent", "ReopenedEvent", "ClosedEvent"],
        "truth": "the one deliberate flip step 8 prescribes",
    },
    1667: {
        "state": "CLOSED",
        "total_count": 119,
        "nodes": ["ClosedEvent", "ReopenedEvent"] * 22 + ["ClosedEvent"],
        "truth": "closed and reopened ten times in under fourteen hours",
    },
}

#: #1667 read twice, twenty minutes apart, with its 45 close/reopen events
#: unchanged in between - the pull request has been closed and retired since
#: 2026-07-30. What moved the count is recorded in ``_DRIFT_CAUSE``.
_DRIFT_1667 = {"first_read": 119, "second_read": 120, "node_count": 45}

#: The 120th item, from ``timelineItems(last: 4)``. Filing #2146 - the issue
#: reporting this defect - cross-referenced #1667, and that is the entire
#: difference between the two reads above.
_DRIFT_CAUSE = {
    "typename": "CrossReferencedEvent",
    "created_at": "2026-08-11T18:23:12Z",
    "source": 2146,
}

#: The same pull request read three ways, to isolate the mechanism. Recorded on
#: #2143: the filter narrows ``nodes`` and leaves ``totalCount`` alone, including
#: when it selects a type the pull request cannot carry.
_MECHANISM: dict[str, dict[str, Any]] = {
    "filtered_to_close_and_reopen": {"total_count": 13, "node_count": 1},
    "no_item_types_argument": {"total_count": 13, "node_count": 13},
    "filtered_to_a_type_not_present": {"total_count": 13, "node_count": 0},
}

#: ``timelineItems(itemTypes: [CLOSED_EVENT, REOPENED_EVENT], first: 3)`` and the
#: same query with ``last: 3``, on #1667. Recorded as ``(typename, createdAt)``.
_HEAD_WINDOW_1667 = [
    ("ClosedEvent", "2026-07-27T22:27:00Z"),
    ("ReopenedEvent", "2026-07-27T22:36:01Z"),
    ("ClosedEvent", "2026-07-28T17:36:03Z"),
]
_TAIL_WINDOW_1667 = [
    ("ClosedEvent", "2026-07-30T05:04:07Z"),
    ("ReopenedEvent", "2026-07-30T05:26:01Z"),
    ("ClosedEvent", "2026-07-30T20:17:18Z"),
]

#: The phrase locating step 8's *Before* bullet. Everything in the prose class is
#: positioned from it, so its absence fails rather than making the rest vacuous.
_ANCHOR = "do not flip #1667 again"

#: How far the correction may sit from the instruction it qualifies while still
#: reading as one bullet. Generous enough to survive rewording, tight enough that
#: moving it out of the *Before* bullet fails.
_ADJACENCY_WINDOW = 2600


def _agents_text() -> str:
    return _AGENTS_PATH.read_text(encoding="utf-8")


def _window_after(text: str, anchor: str) -> str | None:
    """The ``_ADJACENCY_WINDOW`` characters following ``anchor``, or ``None``."""
    position = text.find(anchor)
    if position < 0:
        return None
    return text[position : position + _ADJACENCY_WINDOW]


def _flip_events(payload: dict[str, Any]) -> list[str]:
    """The close/reopen events actually present, which is what ``nodes`` carries."""
    return [typename for typename in payload["nodes"] if typename in _WANTED]


def _verdict_from_nodes(payload: dict[str, Any]) -> str:
    """The judgement step 8 asks for, derived from the node list.

    ``never-closed`` and ``lone-close`` are both safe to flip; an alternating run
    means something is undoing you. A merge writes a ``ClosedEvent`` too, so a
    single close is not evidence anyone closed the pull request by hand - which
    is a distinction the node list can draw and a count cannot.
    """
    events = _flip_events(payload)
    if not events:
        return "never-closed"
    reopens = events.count("ReopenedEvent")
    if reopens == 0:
        return "lone-close"
    return "alternating-run"


def _verdict_from_total_count(payload: dict[str, Any]) -> str:
    """The same judgement derived from ``totalCount``, as step 8 used to permit.

    ``totalCount`` cannot distinguish a close from a reopen - it does not count
    either of them - so any reading of it treats a non-trivial number as history.
    """
    total = payload["total_count"]
    if total == 0:
        return "never-closed"
    if total == 1:
        return "lone-close"
    return "alternating-run"


class TestTheFilteredCountIsTheUnfilteredTotal:
    """``itemTypes`` narrows ``nodes``. It does not narrow ``totalCount``."""

    def test_a_pull_request_never_closed_reports_close_events(self) -> None:
        recorded = _RECORDED[2144]
        assert recorded["state"] == "OPEN"
        assert _flip_events(recorded) == [], (
            "#2144 was open and had never been closed, so the filtered node list is empty. "
            "If this payload is being updated because the pull request has since been "
            "closed, pick another never-closed pull request rather than weakening the row - "
            "it is the one that shows the count is not merely imprecise but unrelated."
        )
        assert recorded["total_count"] == 2, (
            "The recorded totalCount for a connection filtered to [CLOSED_EVENT, "
            "REOPENED_EVENT] on a pull request that was never closed is 2, not 0."
        )

    def test_the_filter_does_not_change_the_count_it_reports(self) -> None:
        filtered = _MECHANISM["filtered_to_close_and_reopen"]["total_count"]
        unfiltered = _MECHANISM["no_item_types_argument"]["total_count"]
        assert filtered == unfiltered == 13, (
            "The same pull request read with and without an itemTypes argument reported "
            "the same totalCount, which is what identifies that number as the whole "
            "timeline rather than the selection."
        )

    def test_a_filter_matching_nothing_still_reports_the_total(self) -> None:
        absent = _MECHANISM["filtered_to_a_type_not_present"]
        assert absent["node_count"] == 0
        assert absent["total_count"] == 13, (
            "Filtering #2143 to a type it cannot carry (CONVERT_TO_DRAFT_EVENT) returned "
            "an empty node list beside totalCount 13. This is the row that rules out a "
            "recompute lag: a stale count would not survive a filter selecting nothing."
        )

    @pytest.mark.parametrize("number", sorted(_RECORDED))
    def test_the_count_never_equals_the_number_of_matching_events(self, number: int) -> None:
        recorded = _RECORDED[number]
        assert recorded["total_count"] != len(_flip_events(recorded)), (
            f"On #{number} ({recorded['truth']}) the filtered totalCount happens to equal "
            "the number of matching events, which would make the two derivations "
            "indistinguishable on this payload. Keep a payload where they differ."
        )

    @pytest.mark.parametrize("number", sorted(_RECORDED))
    def test_the_count_overstates_and_never_understates(self, number: int) -> None:
        recorded = _RECORDED[number]
        assert recorded["total_count"] > len(_flip_events(recorded)), (
            f"On #{number} the count must be larger than the matching-event count. The "
            "direction is the whole cost: an overstated flip history refuses a flip that "
            "was safe, which is a decision that looks like caution and leaves a BLOCKED "
            "pull request stuck forever (#1988). An understating count would instead be "
            "caught by the flip failing to help."
        )


class TestTheCountIsNotEvenStable:
    """The number moves in response to events the filter excludes."""

    def test_the_count_moved_while_the_close_history_did_not(self) -> None:
        assert _DRIFT_1667["first_read"] != _DRIFT_1667["second_read"], (
            "Two reads of #1667 twenty minutes apart returned different filtered "
            "totalCounts. #1667 was closed and retired on 2026-07-30, so nothing about "
            "its close/reopen history can still be changing."
        )
        assert _DRIFT_1667["node_count"] == len(_flip_events(_RECORDED[1667])), (
            "The node list is the same 45 events across both reads, which is what makes "
            "the moving count attributable to the filter being ignored rather than to the "
            "pull request changing."
        )

    def test_the_event_that_moved_it_is_of_an_excluded_type(self) -> None:
        assert _DRIFT_CAUSE["typename"] not in _WANTED, (
            "The item that incremented the count is a CrossReferencedEvent, which is "
            "neither of the two types the query asked for. That is the mechanism stated "
            "as an event rather than as a comparison of two numbers."
        )

    def test_the_reference_that_moved_it_was_this_defect_being_filed(self) -> None:
        assert _DRIFT_CAUSE["source"] == 2146, (
            "The 120th item is a cross-reference from #2146, the issue reporting this "
            "defect - so writing about #1667 raised its apparent flip count by one. This "
            "is the practical consequence worth pinning: a cached count is not merely "
            "wrong once, it drifts, so an agent that recorded 119 and re-read it later "
            "would see 120 and conclude someone had flipped the pull request again."
        )


class TestTheVerdictKeyedOnNodesIsTheRightOne:
    """Both derivations, run over the same recorded payloads."""

    def test_the_node_verdict_clears_the_pull_request_that_was_safe(self) -> None:
        assert _verdict_from_nodes(_RECORDED[2144]) == "never-closed"

    def test_the_count_verdict_refuses_the_pull_request_that_was_safe(self) -> None:
        assert _verdict_from_total_count(_RECORDED[2144]) == "alternating-run", (
            "Reading totalCount on a pull request that had never been closed yields "
            "'something is undoing you'. This is the failure this module exists for: the "
            "flip is the only remedy for the #1988 zero-check-suite condition, so the "
            "misread does not merely delay the merge, it forecloses it."
        )

    def test_the_two_derivations_disagree_where_it_matters(self) -> None:
        safe = _RECORDED[2144]
        assert _verdict_from_nodes(safe) != _verdict_from_total_count(safe)

    def test_the_two_derivations_agree_on_the_genuine_flip_war(self) -> None:
        war = _RECORDED[1667]
        assert _verdict_from_nodes(war) == _verdict_from_total_count(war) == "alternating-run", (
            "On #1667 both derivations say 'do not flip', so their agreement is not "
            "evidence the count is sound - it is what makes the defect invisible on "
            "exactly the pull requests where the rule is doing real work."
        )

    def test_a_merge_close_is_not_read_as_a_flip(self) -> None:
        assert _verdict_from_nodes(_RECORDED[2143]) == "lone-close", (
            "#2143's single filtered event is its own squash merge. A lone close is safe "
            "to re-apply, and distinguishing it from a close someone applied to undo you "
            "needs the node list; the count cannot tell a close from a reopen because it "
            "counts neither."
        )

    def test_the_deliberate_single_flip_is_not_a_run(self) -> None:
        recorded = _RECORDED[1987]
        assert _flip_events(recorded) == ["ClosedEvent", "ReopenedEvent", "ClosedEvent"]
        assert recorded["total_count"] == 25, (
            "#1987 is the pull request step 8's flip remedy was written for. Its filtered "
            "count reads 25 against three real events, so an agent applying that remedy "
            "today and checking its own work with totalCount would read its single "
            "deliberate flip as a war it had joined."
        )


class TestTheTailIsWhatMakesAFlipUnsafe:
    """``nodes`` is oldest-first; the judgement is about what happened last."""

    def test_the_head_and_tail_windows_are_disjoint(self) -> None:
        assert not set(_HEAD_WINDOW_1667) & set(_TAIL_WINDOW_1667), (
            "On #1667 first: 3 and last: 3 select different events entirely, so the "
            "pagination argument decides which question the read answers."
        )

    def test_the_head_window_is_the_older_one(self) -> None:
        newest_head = max(timestamp for _, timestamp in _HEAD_WINDOW_1667)
        oldest_tail = min(timestamp for _, timestamp in _TAIL_WINDOW_1667)
        assert newest_head < oldest_tail, (
            "nodes is ordered oldest-first, which is why a truncated first: N answers "
            "'how did this pull request begin' rather than 'what happened to it last'."
        )

    def test_reading_the_head_can_be_arbitrarily_stale(self) -> None:
        newest_head = max(timestamp for _, timestamp in _HEAD_WINDOW_1667)
        newest_tail = max(timestamp for _, timestamp in _TAIL_WINDOW_1667)
        assert newest_head[:10] != newest_tail[:10], (
            "The two windows on #1667 are five days apart. A first: N read is not a "
            "slightly stale view of the tail, it is a different period of the pull "
            "request's life, so use last: N."
        )


class TestTheGuidanceNamesTheHonestField:
    """The correction has to stay beside the instruction it qualifies."""

    def test_the_anchor_is_still_present(self) -> None:
        # Context guard: every assertion below is positioned from this phrase, so a
        # silent rewording would move the pin rather than break it.
        assert _ANCHOR in _agents_text(), (
            f"AGENTS.md no longer contains {_ANCHOR!r}, which this class uses to locate "
            "step 8's *Before* bullet. If the passage was deliberately reworded, update "
            "_ANCHOR to match rather than deleting these tests - the point is that the "
            "instruction and its correction stay together."
        )

    def test_the_guidance_names_the_field_not_to_read(self) -> None:
        window = _window_after(_agents_text(), _ANCHOR)
        assert window is not None and "totalCount" in window, (
            "AGENTS.md tells a contributor to read a filtered timelineItems connection "
            "before a close or reopen without saying that totalCount ignores the filter. "
            "The bare instruction reads perfectly well, which is what makes the omission "
            "silent: the natural way to answer 'how many' returns the whole timeline's "
            "count. See #2146."
        )

    def test_the_guidance_names_the_field_to_read_instead(self) -> None:
        window = _window_after(_agents_text(), _ANCHOR)
        assert window is not None and "nodes" in window, (
            "AGENTS.md must say which field is honest, not only which one is not - "
            "otherwise the reader knows the read is unreliable and not what to do "
            "instead. See #2146."
        )

    def test_the_guidance_carries_the_never_closed_measurement(self) -> None:
        window = _window_after(_agents_text(), _ANCHOR)
        assert window is not None and "#2144" in window, (
            "AGENTS.md must carry the measurement, because the claim is counter-intuitive "
            "enough to be tidied away as a redundant caveat: #2144 had never been closed "
            "and its filtered count read 2. Without a number the correction reads as "
            "pedantry about two equivalent fields. See #2146."
        )

    def test_the_guidance_states_what_the_misread_costs(self) -> None:
        window = _window_after(_agents_text(), _ANCHOR)
        assert window is not None and "BLOCKED" in window, (
            "AGENTS.md must say that the refused flip is the only remedy for a pull "
            "request held at BLOCKED with no check suite. Without the consequence the "
            "correction is a curiosity about an API field, and the action it changes - "
            "flip anyway - looks less cautious than declining. See #2146 and #1988."
        )

    def test_the_guidance_tells_the_reader_to_read_the_tail(self) -> None:
        window = _window_after(_agents_text(), _ANCHOR)
        assert window is not None and "last:" in window, (
            "AGENTS.md must name the pagination argument. nodes is oldest-first and the "
            "judgement is about the most recent events, so a first: N read answers a "
            "different question - and that one errs in the opposite, reassuring "
            "direction. See #2146."
        )
