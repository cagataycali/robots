### Quality: a filtered timeline read counts the events it filtered out

`AGENTS.md` > PR Workflow step 8 tells a contributor to query
`timelineItems(itemTypes: [CLOSED_EVENT, REOPENED_EVENT])` before closing or
reopening a pull request, because "a lone `CLOSED_EVENT` is safe to re-apply; an
alternating run means something is undoing you". The field a reader reaches for
to answer "how many" is `totalCount`, and on a filtered connection that is the
count of the *whole* timeline - commits, reviews, comments, project status
changes. The `itemTypes` argument narrows `nodes` and nothing else.

So the cheap read invents a flip history. #2144 had never been closed once in its
life and its close/reopen-filtered count read `2`; #1667's read `119` against 45
real events. Filtering #2143 to a type it cannot carry
(`CONVERT_TO_DRAFT_EVENT`) returns `totalCount: 13` beside an empty node list,
which rules out a recompute lag - the number is answering a different question,
and no re-read helps. It is not even stable: #1667 has been closed and retired
since 2026-07-30, and two reads twenty minutes apart returned `119` then `120`
with its 45 close/reopen events unchanged, the new item being a cross-reference
from the issue reporting this defect.

The direction is the cost. An overstated flip history refuses a flip that was
safe, and step 8 prescribes a close/reopen as the only remedy for a head commit
that spawned no check suite, where `BLOCKED` is terminal and both re-running and
re-pushing are unavailable. Declining then looks exactly like following the rule
while leaving the pull request stuck and presenting as reviewer bandwidth. On a
genuine flip war both readings say "do not flip", so their agreement is never
evidence the count is sound.

Step 8 now says to count the `nodes`, to read the tail with `last: N` because
`nodes` is oldest-first while the judgement is about what happened last, and that
a merge writes a `ClosedEvent` too - so a lone close may be the squash rather
than someone undoing you.
`tests/test_timeline_filter_count_is_unfiltered.py` executes the arithmetic over
the recorded payloads and pins the prose adjacency separately, so the correction
cannot drift away from the instruction it qualifies.
